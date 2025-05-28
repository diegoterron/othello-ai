
import sys
import os

# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.board.othello_board import OthelloBoard
from src.mcts.UCT_tree import Node
import numpy as np

node = Node(OthelloBoard())

print("Initial node")
print(node)
print("UCT search aplication for low iters")
node.UCT_search(10)
for child in node.children:
    print(child, sep='\t')
