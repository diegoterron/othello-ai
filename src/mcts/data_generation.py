import sys
import os

# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import numpy as np
from src.board.othello_board import OthelloBoard
from src.mcts.UCT_tree import Node
from tensorflow import keras

MODEL_PATH = 'models/untrained_model.keras'

def generate_data(num_games=100,UCT_depth=100,useModel=False):
    """
    Generate training data for the Othello game using MCTS.
    num_games: Number of samples to generate.
    Returns a list of tuples (board,winner).
    """
    model = keras.models.load_model(MODEL_PATH) if useModel else None
    data = []
    print(f'Model {"loaded" if useModel else "usage is not active"}. Generating data...')
    for _ in range(num_games):
        board = OthelloBoard()
        acum= []
        node = Node(board, model=model)

        # Perform MCTS to get a sample tree
        while node.board.is_game_over() is False:
            bestChild = node.UCT_search(UCT_depth)
            if bestChild is None: ## The player has no legal moves
                board = OthelloBoard(board=node.board.board, turn=1 if node.board.get_turn() == 2 else 2) 
                node = Node(board, parent=node, in_action=None, model=model)
            else:
                node = bestChild
                acum.append(node.board.board)  # Store the board state for this node
        node.board.check_game_over()
        print(f"Game finished:{node.board}. Winner: {node.board.get_winner()}")
        data.extend([(board,node.board.get_winner()) for board in acum])

    return data

def save_data(data, filename='data/othello_train_data.npz'):
    """
    Save the generated data to a file.
    data: List of tuples (board, winner).
    filename: Name of the file to save the data.
    """
    inputs = np.array([board for board, _ in data])
    labels = np.array([label for _, label in data])
    np.savez(filename, X=inputs, y=labels)
    print(f"Data saved to {filename}")