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

    def __init__(self, n: int = 5, max_h: int = 4, pieces=None):
        self.max_h = max_h
        self.n = n
        # Create the empty board array.
        self.players_locations = {+1: {(self.n-2, self.n-2), (1, 1)},
                                  -1: {(self.n-2, 1), (self.n-2, 1)}}
        self.height_map = np.zeros((self.n, self.n), dtype=np.int32)

        if pieces is not None:
            assert pieces.shape == (self.n, self.n, 2)
            self.height_map = np.copy(pieces[:, :, 1])
            self.players_locations = {color: set([(x, y) for (y, x) in np.argwhere(pieces[:, :, 0] == color)]) for color in [-1, 1]}

    @property
    def pieces(self):
        pcs = np.zeros((self.n, self.n, 2), dtype=np.int32)
        for color, locs in self.players_locations.items():
            for (x, y) in locs:
                pcs[y][x][0] = color
        pcs[:, :, 1] = np.copy(self.height_map)
        return pcs

    def get_legal_actions(self, color: int) -> List[GameAction]:
        actions = set()  # stores the legal moves.

        # Get all the squares with pieces of the given color.
        for (x, y) in self.players_locations[color]:
            sq_actions = self.get_actions_from_square((x, y))
            actions.update(sq_actions)
        return list(actions)

    def has_legal_actions(self, color) -> bool:
        for (x, y) in self.players_locations[color]:
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
        pieces = self.pieces
        color, height = pieces[y][x]

        dest = tuple(map(sum, zip(origin, move_dir)))
        if not all(map(lambda t: 0 <= t < self.n, dest)):
            return []

        dest_x, dest_y = dest
        dest_color, dest_height = pieces[dest_y][dest_x]

        if dest_color != 0 or dest_height > height + 1 or dest_height >= self.max_h:
            return []

        actions = []
        for build_dir in self.directions:
            build = tuple(map(sum, zip(dest, build_dir)))
            build_x, build_y = build
            if not all(map(lambda t: 0 <= t < self.n, build)):
                continue
            build_color, build_height = pieces[build_y][build_x]
            if (build != origin and build_color != 0) or build_height >= self.max_h:
                continue
            actions.append((origin, move_dir, build_dir))
        return actions

    def execute_action(self, action: GameAction, color: int):
        origin, move_dir, build_dir = action
        dest = tuple(map(sum, zip(origin, move_dir)))
        build = tuple(map(sum, zip(dest, build_dir)))
        x, y = origin
        build_x, build_y = build
        self.players_locations[color].remove(origin)
        self.players_locations[color].add(dest)
        self.height_map[build_y][build_x] += 1

    def reached_top(self, player: int):
        for (x, y) in self.players_locations[player]:
            if self.height_map[y][x] == self.max_h - 1:
                return True
        return False
