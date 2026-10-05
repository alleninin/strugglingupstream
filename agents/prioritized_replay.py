import numpy as np


class SumTree:
    def __init__(self, capacity: int):
        if capacity < 1:
            raise ValueError("capacity must be positive")
        self.capacity = int(capacity)
        self.tree = np.zeros(2 * self.capacity, dtype=np.float64)
        self.data = [None] * self.capacity
        self.pos = 0
        self.size = 0

    def _propagate(self, leaf: int, change: float) -> None:
        parent = leaf // 2
        while parent >= 1:
            self.tree[parent] += change
            parent //= 2

    def update(self, data_idx: int, priority: float) -> None:
        leaf = data_idx + self.capacity
        change = priority - self.tree[leaf]
        self.tree[leaf] = priority
        self._propagate(leaf, change)

    def add(self, data, priority: float) -> int:
        data_idx = self.pos
        self.data[data_idx] = data
        self.update(data_idx, priority)
        self.pos = (self.pos + 1) % self.capacity
        if self.size < self.capacity:
            self.size += 1
        return data_idx

    def total(self) -> float:
        return float(self.tree[1])

    def get(self, s: float):
        idx = 1
        while idx < self.capacity:
            left = 2 * idx
            right = left + 1
            if s < self.tree[left]:
                idx = left
            else:
                s -= self.tree[left]
                idx = right
        data_idx = idx - self.capacity
        return data_idx, self.data[data_idx], float(self.tree[idx])


class PrioritizedReplayBuffer:
    def __init__(
        self,
        capacity: int = 20000,
        alpha: float = 0.6,
        beta: float = 0.4,
        beta_anneal_steps: int = 100000,
        epsilon: float = 1e-6,
        rng=None,
    ):
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.beta_start = float(beta)
        self.beta_anneal_steps = max(1, int(beta_anneal_steps))
        self.beta_increment = (1.0 - self.beta) / self.beta_anneal_steps
        self.epsilon = float(epsilon)
        self.tree = SumTree(capacity)
        self.max_priority = 1.0
        self.rng = rng if rng is not None else np.random.default_rng()

    def push(self, transition, priority: float = None) -> None:
        if priority is None:
            p = self.max_priority
        else:
            p = (abs(float(priority)) + self.epsilon) ** self.alpha
        self.tree.add(transition, p)
        if p > self.max_priority:
            self.max_priority = p

    def sample(self, batch_size: int):
        total = self.tree.total()
        if total <= 0 or self.tree.size == 0:
            return None
        idxs, data, priorities = [], [], []
        for _ in range(batch_size):
            s = self.rng.random() * total
            di, d, p = self.tree.get(s)
            idxs.append(di)
            data.append(d)
            priorities.append(p)
        priorities = np.asarray(priorities, dtype=np.float64)
        probs = priorities / total
        n = self.tree.size
        weights = (n * probs) ** (-self.beta)
        weights = weights / weights.max()
        return data, idxs, weights.astype(np.float32)

    def update_priorities(self, idxs, priorities) -> None:
        for i, p in zip(idxs, priorities):
            pp = (abs(float(p)) + self.epsilon) ** self.alpha
            self.tree.update(i, pp)
            if pp > self.max_priority:
                self.max_priority = pp

    def anneal_beta(self) -> None:
        self.beta = min(1.0, self.beta + self.beta_increment)

    def __len__(self) -> int:
        return self.tree.size
