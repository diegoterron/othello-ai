
import sys
import os

# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.board.othello_board import OthelloBoard
from src.mcts.UCT_tree import Node
import numpy as np
from src.mcts.tree_visualization import draw_tree

node = Node(OthelloBoard())


print("Drawing tree...")
node.UCT_search(10) # KEEP ITERS LOW FOR TESTING
draw_tree(node)




