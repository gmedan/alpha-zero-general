from __future__ import print_function
import sys
sys.path.append('..')
from Game import Game
from .SantoriniLogic import Board
import numpy as np


class SantoriniGame(Game):

    square_content = {
        -1: "-",
        +0: " ",
        +1: "+"
    }

    @staticmethod
    def get_square_piece(color: int):
        return SantoriniGame.square_content[color]

    def __init__(self, n: int = 5, max_h: int = 4):
        super(SantoriniGame).__init__()
        self.n = n
        self.max_h = max_h

    def getInitBoard(self):
        # return initial board (numpy board)
        b = Board(self.n, self.max_h)
        return np.array(b.pieces)

    def getBoardSize(self):
        # (a,b) tuple
        return self.n, self.n

    def getActionSize(self):
        # return number of actions
        return self.n**2 * len(Board.directions)**2

    def getNextState(self, board, player, action_ind: int):
        b = Board(self.n, self.max_h)
        b.pieces = np.copy(board)
        action = Board.action_from_linear_ind(self.n, action_ind)
        b.execute_action(action, player)
        return b.pieces, -player

    def getValidMoves(self, board, player):
        # return a fixed size binary vector
        valids = [0]*self.getActionSize()
        b = Board(self.n, self.max_h)
        b.pieces = np.copy(board)
        legal_actions = b.get_legal_actions(player)
        if len(legal_actions) == 0:
            return np.array(valids)
        for action in legal_actions:
            valids[int(Board.linear_ind_from_action(self.n, action))] = 1
        return np.array(valids)

    def getGameEnded(self, board, player):
        # return 0 if not ended, 1 if player 1 won, -1 if player 1 lost
        b = Board(self.n, self.max_h)
        b.pieces = np.copy(board)
        if b.reached_top(-player):
            return -1
        if not b.has_legal_actions(player):
            return -1
        return 0

    def getCanonicalForm(self, board, player):
        new_board = np.copy(board)
        new_board[:, :, 0] *= player
        return new_board

    def getSymmetries(self, board, pi):
        # mirror, rotational
        assert(len(pi) == self.getActionSize())
        pi_board = np.reshape(pi, (self.n, self.n, len(Board.directions), len(Board.directions)))
        l = [(board, pi)]
        return l  # TODO
        #
        # for i in range(1, 5):
        #     for j in [True, False]:
        #         newB = np.rot90(board, i)
        #         newPi = np.rot90(pi_board, i)
        #         if j:
        #             newB = np.fliplr(newB)
        #             newPi = np.fliplr(newPi)
        #         l += [(newB, list(newPi.ravel()))]
        # return l

    def stringRepresentation(self, board):
        return board.tostring()

    def stringRepresentationReadable(self, board):
        board_s = "".join(self.square_content[square] for row in board for square in row)
        return board_s

    @staticmethod
    def display(board):
        n = board.shape[0]
        print("   ", end="")
        for y in range(n):
            print(" {0} ".format(y), end=" ")
        print("")
        print("-----------------------")
        for y in range(n):
            print(y, "|", end="")    # print the row #
            for x in range(n):
                color, height = board[y][x]    # get the piece to print
                print("{1}{0}{1}".format(height, SantoriniGame.square_content[color]), end=" ")
            print("|")

        print("-----------------------")
