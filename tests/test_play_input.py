import contextlib
import io
import unittest
from collections import Counter
from unittest.mock import patch

from test_regressions import game_for

from game.moves import PASS_MOVE, MoveType
from scripts.play import human_choose, parse_play, show_hand


class PlayInputTests(unittest.TestCase):
    def test_rank_input_uses_owned_cards_and_preserves_duplicates(self):
        game = game_for([3, 3, 4, 4, 5, 5, 9], [7])
        move = parse_play("5, 3 4, 3 5 4", game.hands[0], game.legal_moves())
        self.assertEqual(move.type, MoveType.CONSEC_PAIRS)
        self.assertEqual(
            Counter(card.rank for card in move.cards), Counter([3, 3, 4, 4, 5, 5])
        )
        game.apply_move(0, move)
        self.assertEqual(show_hand(game.hands[0]), "9")

    def test_single_number_is_a_rank_not_an_index(self):
        game = game_for([3, 7, 9, 15], [4])
        self.assertEqual(parse_play("3", game.hands[0], game.legal_moves()).rank, 3)
        self.assertEqual(parse_play("2", game.hands[0], game.legal_moves()).rank, 15)

    def test_face_cards_and_jokers_are_case_insensitive(self):
        game = game_for([10, 11, 11, 12, 13, 14, 15, 16, 17], [3])
        for text, ranks in (
            ("j J", [11, 11]),
            ("q", [12]),
            ("k", [13]),
            ("a", [14]),
            ("10", [10]),
            ("bj", [16]),
            ("RJ", [17]),
        ):
            with self.subTest(text=text):
                move = parse_play(text, game.hands[0], game.legal_moves())
                self.assertEqual([card.rank for card in move.cards], ranks)

    def test_invalid_inputs_do_not_mutate_the_hand(self):
        game = game_for([3, 3, 4, 5], [7])
        before = list(game.hands[0])
        for text, message in (
            ("", "Enter cards"),
            ("0", "Unknown card"),
            ("3 3 3", "don't have"),
            ("3 4", "valid combination"),
            ("p", "when leading"),
        ):
            with self.subTest(text=text), self.assertRaisesRegex(ValueError, message):
                parse_play(text, game.hands[0], game.legal_moves())
            self.assertEqual(game.hands[0], before)

    def test_following_rejects_low_cards_but_allows_pass_and_bombs(self):
        game = game_for([3, 3, 3, 3, 5], [10, 14])
        game.current_player = 1
        game.apply_move(1, next(move for move in game.legal_moves() if move.rank == 10))
        with self.assertRaisesRegex(ValueError, "doesn't beat"):
            parse_play("5", game.hands[0], game.legal_moves())
        self.assertEqual(
            parse_play("PASS", game.hands[0], game.legal_moves()), PASS_MOVE
        )
        move = parse_play("3 3 3 3", game.hands[0], game.legal_moves())
        self.assertEqual(move.type, MoveType.BOMB)
        game.apply_move(0, move)

    def test_normal_turn_is_compact_and_retries_bad_input(self):
        game = game_for([3, 3, 7], [9])
        output = io.StringIO()
        with (
            contextlib.redirect_stdout(output),
            patch("builtins.input", side_effect=["3 3 3", "3 3"]),
        ):
            move = human_choose(game, game.legal_moves())
        self.assertEqual(move.type, MoveType.PAIR)
        self.assertIn("don't have", output.getvalue())
        self.assertNotIn("Legal plays:", output.getvalue())
        self.assertIn("P1: 1 left", output.getvalue())

    def test_help_and_move_listing_are_available_on_request(self):
        game = game_for([3, 3, 7], [9])
        output = io.StringIO()
        with (
            contextlib.redirect_stdout(output),
            patch("builtins.input", side_effect=["?", "moves", "3"]),
        ):
            move = human_choose(game, game.legal_moves())
        self.assertEqual(move.rank, 3)
        self.assertIn("quit: end game", output.getvalue())
        self.assertIn("Legal plays:", output.getvalue())

    def test_lowercase_q_plays_a_queen_instead_of_quitting(self):
        game = game_for([3, 12], [9])
        with (
            contextlib.redirect_stdout(io.StringIO()),
            patch("builtins.input", return_value="q"),
        ):
            move = human_choose(game, game.legal_moves())
        self.assertEqual(move.rank, 12)

    def test_quit_exits_cleanly_without_playing(self):
        game = game_for([3, 3, 7], [9])
        with (
            contextlib.redirect_stdout(io.StringIO()),
            patch("builtins.input", return_value="quit"),
        ):
            with self.assertRaises(SystemExit) as stopped:
                human_choose(game, game.legal_moves())
        self.assertEqual(stopped.exception.code, 0)
        self.assertEqual(len(game.hands[0]), 3)


if __name__ == "__main__":
    unittest.main()
