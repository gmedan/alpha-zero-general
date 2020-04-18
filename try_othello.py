
import Arena
from MCTS import MCTS

from othello.OthelloGame import OthelloGame
from othello.pytorch.NNet import NNetWrapper as OthelloPytorchNNet
from othello.OthelloPlayers import *
import numpy as np
from utils import *

game = OthelloGame(6)
neural_net = OthelloPytorchNNet

nnet = neural_net(game)
nnet.load_checkpoint('./pretrained_models/othello/pytorch/', '6x100x25_best.pth.tar')
mcts1 = MCTS(game, nnet, dotdict({'numMCTSSims': 25, 'cpuct': 1.0}))
mcts2 = MCTS(game, nnet, dotdict({'numMCTSSims': 35, 'cpuct': 1.0}))
n1p = lambda x: np.argmax(mcts1.getActionProb(x, temp=0))
n2p = lambda x: np.argmax(mcts2.getActionProb(x, temp=0))
arena = Arena.Arena(n1p, n2p, game, display=OthelloGame.display)
print(arena.playGames(100, verbose=False))