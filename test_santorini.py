
import unittest

from santorini.SantoriniLogic import *
from santorini.SantoriniGame import SantoriniGame
import numpy as np


class TestAllGames(unittest.TestCase):

    def test_santorini(self):
        board = np.dstack([
           [[0,-1, 0,+1, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0,-1, 0, 0],
            [0, 0, 0, 0,+1]],
           [[1,  0,  1,  0,  4],
            [2,  4,  4,  4,  4],
            [0,  1,  4,  4,  4],
            [2,  3,  0,  0,  4],
            [4,  2,  4,  4,  0]]]).astype(np.int32)
        b = Board(5, 4, board)

        p1_reached_top = b.reached_top(+1)
        p2_reached_top = b.reached_top(-1)
        p1_has_legal_actions = b.has_legal_actions(+1)
        p2_has_legal_actions = b.has_legal_actions(-1)

        g = SantoriniGame(b.n, b.max_h)
        p1_game_ended = g.getGameEnded(b.pieces, +1)
        p2_game_ended = g.getGameEnded(b.pieces, -1)
        print('?')

if __name__ == '__main__':
    unittest.main()