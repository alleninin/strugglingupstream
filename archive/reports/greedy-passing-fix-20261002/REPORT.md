# Greedy passing correction

The reported symptom was reproduced with a constructed deal, not a recording of the user’s game. The player led singles 3 through 10 while holding two bombs; all three bots passed despite legal replies. The previous policy first contested after 11 cards had been shed. The corrected policy contests the first single with a spare 2 and leaves its pairs intact.

## Cause and correction

- The old zero-control-cost filter excluded intact aces, 2s, and jokers. These are now legitimate replies; control cost ranks choices rather than forbidding them.
- The old five-card threat threshold was too late for opponents who could finish in two bombs. Pressure now includes a player reaching the final third of a starting hand or gaining a substantial card-count lead.
- Requiring every reply to immediately reduce the estimated number of remaining plays caused additional passes. Under pressure, the bot can accept up to one extra estimated future play and spend whole bombs. It still preserves combinations when there is no pressure and never invents an illegal reply.
- Bomb-only responses now use the actual table combination size when detecting an immediate threat.

## Validation

All 59 unit/regression tests and the existing rule checks passed, including 50 full games. Eight new tests cover spare high cards, racing an opponent, bombs, unavailable replies, large table combinations, and the reproduced sequence.

The simulation below uses 50 held-out deals (seeds 310000–310049), rotated through all four player seats, for 200 games per policy/version, 800 total. The focal policy faces three copies of the old or corrected Greedy. The two probe policies deliberately lead low singles or favor large combinations; they are not human players.

“Early pass” counts only replies with a legal non-pass option while the focal player still has more than eight cards. Forced passes and leading turns are excluded. The hands evolve differently after decisions diverge, so this is a before/after behavioral comparison, not an independent per-action causal estimate.

| Probe policy | Version | Early voluntary pass rate | Probe first-place rate |
|---|---|---:|---:|
| low_singles | before | 82.8% | 21.5% |
| low_singles | after | 19.1% | 2.0% |
| large_combos | before | 78.5% | 15.0% |
| large_combos | after | 13.2% | 9.5% |

Reproduce with `python3.12 -B reports/greedy-passing-fix-20261002/validate.py`. The saved `greedy_before.py` is the exact pre-fix policy and `results.json` contains every result. No learned checkpoints were changed or retrained. Restart `scripts/play.py --greedy` to use the updated fixed policy.
