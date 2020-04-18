
import Arena
from MCTS import MCTS

from santorini.SantoriniGame import SantoriniGame
from santorini.pytorch.NNet import NNetWrapper as SantoriniPytorchNNet
from santorini.SantoriniPlayers import *
import numpy as np
from utils import *

game = SantoriniGame()
rp = RandomPlayer(game).play
arena = Arena.Arena(rp, rp, game, display=SantoriniGame.display)
print("Player 1 won: {}, lost: {}, draws: {}".format(*arena.playGames(50, verbose=False)))