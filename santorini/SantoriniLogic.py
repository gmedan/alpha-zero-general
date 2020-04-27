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

Position = Tuple[int, int, int]
MoveDirection = Tuple[int, int, int]
BuildDirection = Tuple[int, int, int]
GameAction = Tuple[int, MoveDirection, BuildDirection]


class Board:

    # list of all 8 directions on the board, as (x,y) offsets
    directions = [(1, 1, 0), (1, 0, 0), (1, -1, 0), (0, -1, 0), (-1, -1, 0), (-1, 0, 0), (-1, 1, 0), (0, 1, 0)]

    @staticmethod
    def get_action_size():
        return 2 * len(Board.directions)**2

    @staticmethod
    def action_from_linear_ind(action_ind: int) -> GameAction:
        shp = (2, len(Board.directions), len(Board.directions))
        z, move, build = np.unravel_index(action_ind, shp, order='C')
        move_dir, build_dir = Board.directions[move], Board.directions[build]
        action = (z, move_dir, build_dir)
        return action

    @staticmethod
    def linear_ind_from_action(action: GameAction) -> int:
        shp = (2, len(Board.directions), len(Board.directions))
        z, move_dir, build_dir = action
        move_ind = np.argwhere([move_dir == direction for direction in Board.directions])
        build_ind = np.argwhere([build_dir == direction for direction in Board.directions])
        return np.ravel_multi_index((z, move_ind, build_ind), shp, order='C')[0]

    # @staticmethod
    # def flip_action_lr(n: int, action: GameAction) -> GameAction:
    #     origin, move_dir, build_dir = action
    #     return (n-1-origin[0], origin[1]), (-move_dir[0], move_dir[1]), (-build_dir[0], build_dir[1])

    # @staticmethod
    # def rot_action_90ccw(n: int, action: GameAction) -> GameAction:
    #     origin, move_dir, build_dir = action
    #     return (n-1-origin[0], origin[1]), (-move_dir[0], move_dir[1]), (-build_dir[0], build_dir[1])

    def __init__(self, n: int = 5, max_h: int = 4, pieces=None):
        self.max_h = max_h
        self.n = n
        # Create the empty board array.
        self.players_locations = {+1: {(self.n-2, self.n-2, 0), (1, 1, 1)},
                                  -1: {(self.n-2, 1, 0), (1, self.n-2, 1)}}
        self.height_map = np.zeros((self.n, self.n), dtype=np.int32)

        if pieces is not None:
            assert pieces.shape == (self.n, self.n, 5)
            self.height_map = np.copy(pieces[:, :, -1])
            self.players_locations = {color: set([(x, y, z) for (y, x, z) in np.argwhere((
                                                                             pieces[:, :, slice(0, 2) if color == 1 else slice(2, 4)]))])
                                      for color in [-1, 1]}

    @property
    def pieces(self):
        pcs = np.zeros((self.n, self.n, 5), dtype=np.int32)
        for color, locs in self.players_locations.items():
            for (x, y, z) in locs:
                actual_z = z + (0 if color == 1 else 2)
                pcs[y][x][actual_z] = 1
        pcs[:, :, -1] = np.copy(self.height_map)
        return pcs

    def get_legal_actions(self, color: int) -> List[GameAction]:
        actions = set()  # stores the legal moves.

        # Get all the squares with pieces of the given color.
        for loc in self.players_locations[color]:
            sq_actions = self.get_actions_from_square(loc, color)
            actions.update(sq_actions)
        return list(actions)

    def has_legal_actions(self, color) -> bool:
        for loc in self.players_locations[color]:
            sq_actions = self.get_actions_from_square(loc, color)
            if len(sq_actions) > 0:
                return True
        return False

    def get_actions_from_square(self, origin: Position, color: int) -> List[GameAction]:
        # search all possible move directions.
        actions = []
        for move_dir in self.directions:
            actions_in_move_dir = self._get_actions_from_square_in_move_direction(origin, color, move_dir)
            for act in actions_in_move_dir:
                if act:
                    actions.append(act)

        # return the generated move list
        return actions

    def get_square_color(self, x: int, y: int):
        if (x, y) in [(x, y) for (x, y, _) in self.players_locations[1]]:
            return 1
        if (x, y) in [(x, y) for (x, y, _) in self.players_locations[-1]]:
            return -1
        else:
            return 0

    def get_square_height(self,  x: int, y: int):
        return self.height_map[y][x]

    def _get_actions_from_square_in_move_direction(self, origin: Position, color: int, move_dir: MoveDirection) -> List[GameAction]:
        x, y, z = origin
        pieces = self.pieces
        height = pieces[y][x][-1]

        dest = tuple(map(sum, zip(origin, move_dir)))
        if not all(map(lambda t: 0 <= t < self.n, dest[:2])):
            return []

        dest_x, dest_y, dest_z = dest
        dest_color = self.get_square_color(dest_x, dest_y)
        dest_height = pieces[dest_y][dest_x][-1]

        if dest_color != 0 or dest_height > height + 1 or dest_height >= self.max_h:
            return []

        actions = []
        for build_dir in self.directions:
            build = tuple(map(sum, zip(dest, build_dir)))
            build_x, build_y, build_z = build
            if not all(map(lambda t: 0 <= t < self.n, build[:2])):
                continue
            build_height = pieces[build_y][build_x][-1]
            build_color = self.get_square_color(build_x, build_y)
            if (build != origin and build_color != 0) or build_height >= self.max_h:
                continue
            actions.append((z, move_dir, build_dir))
        return actions

    def execute_action(self, action: GameAction, color: int):
        z, move_dir, build_dir = action
        origin = [(x, y, z) for (x, y, z_) in self.players_locations[color] if z_ == z][0]
        dest = tuple(map(sum, zip(origin, move_dir)))
        build = tuple(map(sum, zip(dest, build_dir)))
        build_x, build_y, _ = build
        self.players_locations[color].remove(origin)
        self.players_locations[color].add(dest)
        self.height_map[build_y][build_x] += 1

    def reached_top(self, player: int):
        for (x, y, z) in self.players_locations[player]:
            if self.height_map[y][x] == self.max_h - 1:
                return True
        return False
