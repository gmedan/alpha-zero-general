"""
Author: Eric P. Nichols
Date: Feb 8, 2008.
Board class.
Board data:
  1=white, -1=black, 0=empty
  first dim is column , 2nd is row:
     pieces[1][7] is the square in column 2,
     at the opposite end of the board in row 8.
Squares are stored and manipulated as (x,y) tuples.
x is the column, y is the row.
"""
from typing import List, Tuple
import numpy as np

Position = Tuple[int, int]
MoveDirection = Tuple[int, int]
BuildDirection = Tuple[int, int]
GameAction = Tuple[Position, MoveDirection, BuildDirection]


class Board:

    # list of all 8 directions on the board, as (x,y) offsets
    directions = [(1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1), (0, 1)]

    @staticmethod
    def action_from_linear_ind(n: int, action_ind: int) -> GameAction:
        shp = (n, n, len(Board.directions), len(Board.directions))
        y, x, move, build = np.unravel_index(action_ind, shp, order='C')
        origin, move_dir, build_dir = (x, y), Board.directions[move], Board.directions[build]
        action = (origin, move_dir, build_dir)
        return action

    @staticmethod
    def linear_ind_from_action(n: int, action: GameAction) -> int:
        shp = (n, n, len(Board.directions), len(Board.directions))
        origin, move_dir, build_dir = action
        x, y = origin
        move_ind = np.argwhere([move_dir == direction for direction in Board.directions])
        build_ind = np.argwhere([build_dir == direction for direction in Board.directions])
        return np.ravel_multi_index((y, x, move_ind, build_ind), shp, order='C')[0]

    def __init__(self, n: int = 5, max_h: int = 4):
        """Set up initial board configuration."""
        self.max_h = max_h
        self.n = n
        # Create the empty board array.
        self.pieces = np.zeros((self.n, self.n, 2), dtype=np.int32)

        # Set up the initial 4 pieces.
        self.pieces[-2][-2][0] = 1
        self.pieces[1][1][0] = 1
        self.pieces[1][-2][0] = -1
        self.pieces[-2][1][0] = -1

    def get_legal_actions(self, color: int) -> List[GameAction]:
        actions = set()  # stores the legal moves.

        # Get all the squares with pieces of the given color.
        for y in range(self.n):
            for x in range(self.n):
                if self.pieces[y][x][0] == color:
                    sq_actions = self.get_actions_from_square((x, y))
                    actions.update(sq_actions)
        return list(actions)

    def has_legal_actions(self, color) -> bool:
        for y in range(self.n):
            for x in range(self.n):
                if self.pieces[y][x][0] == color:
                    sq_actions = self.get_actions_from_square((x, y))
                    if len(sq_actions) > 0:
                        return True
        return False

    def get_actions_from_square(self, origin: Position) -> List[GameAction]:
        # search all possible move directions.
        actions = []
        for move_dir in self.directions:
            actions_in_move_dir = self._get_actions_from_square_in_move_direction(origin, move_dir)
            for act in actions_in_move_dir:
                if act:
                    actions.append(act)

        # return the generated move list
        return actions

    def _get_actions_from_square_in_move_direction(self, origin: Position, move_dir: MoveDirection) -> List[GameAction]:
        x, y = origin
        color, height = self.pieces[y][x]

        dest = list(map(sum, zip(origin, move_dir)))
        if not all(map(lambda t: 0 <= t < self.n, dest)):
            return []

        dest_x, dest_y = dest
        dest_color, dest_height = self.pieces[dest_y][dest_x]

        if dest_color != 0 or dest_height > height + 1 or dest_height >= self.max_h:
            return []

        actions = []
        for build_dir in self.directions:
            build = list(map(sum, zip(dest, build_dir)))
            build_x, build_y = build
            if not all(map(lambda t: 0 <= t < self.n, build)):
                continue
            build_color, build_height = self.pieces[build_y][build_x]
            if (build != origin and build_color != 0) or build == origin or build_height >= self.max_h:
                continue
            actions.append((origin, move_dir, build_dir))
        return actions

    def execute_action(self, action: GameAction, color: int):
        origin, move_dir, build_dir = action
        dest = list(map(sum, zip(origin, move_dir)))
        build = list(map(sum, zip(dest, build_dir)))
        x, y = origin
        dest_x, dest_y = dest
        build_x, build_y = build
        origin_color, origin_height = self.pieces[y][x]
        assert color == origin_color
        assert origin_height < self.max_h
        self.pieces[y][x][0] = 0
        self.pieces[dest_y][dest_x][0] = color
        self.pieces[build_y][build_x][1] += 1

    def reached_top(self, player: int):
        b = np.copy(self.pieces)
        b = b[:, :, 0] * b[:, :, 1] == player * (self.max_h - 1)
        ans = np.any(b, keepdims=False)
        return ans
