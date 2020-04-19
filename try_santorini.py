
import Arena
from MCTS import MCTS

from santorini.SantoriniGame import SantoriniGame
from santorini.pytorch.NNet import NNetWrapper as SantoriniPytorchNNet
from santorini.SantoriniPlayers import *
import numpy as np
from utils import *

game = SantoriniGame()
rp = RandomPlayer(game).play
gp = GreedySantoriniPlayer(game).play
neural_net = SantoriniPytorchNNet
nnet = neural_net(game)
# nnet.load_checkpoint(folder='./temp/', filename='checkpoint_13.pth.tar')
mcts = MCTS(game, nnet, dotdict({'numMCTSSims': 25, 'cpuct': 1.0}))
n1p = lambda x: np.argmax(mcts.getActionProb(x, temp=0))

arena = Arena.Arena(n1p, gp, game, display=SantoriniGame.display)
print("Player 1 won: {}, lost: {}, draws: {}".format(*arena.playGames(20, verbose=True)))