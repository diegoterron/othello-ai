import numpy as np
from src.board.othello_board import OthelloBoard
from src.mcts.UCT_tree import Node
from tensorflow import keras

MODEL_PATH = 'models/untrained_model.keras'

def _generate_data(root,data):
    if root.children:
        acum = [] #Create an acum for each no-leaf node 
        for child in root.children: #For each childrenof the node
            acum += _generate_data(child,data) #recursively generates the labels
        data.extend([(root.board,winner) for winner in acum]) 
        """Adds a pair (node,label) to the data. If the board is in a draw position, the list will most likely have roughly the same number of 1s and -1s, so it will be a balanced dataset. Else, it will have a lot of 1s or -1s, depending on the winner. This is desirable because it will help the model to learn to predict the winner of the game from an early state."""
        return acum #returns the acumulated labels to add them to the father node
    else: # Leaf node, no children
        root.board.check_game_over() #Updates the winner of the game
        data.append((root.board, root.board.get_winner()))  # Add the board and its winner to the data
        return [root.board.get_winner()] #return a 1 element list to be added to the father acum.

def generate_data(num_samples=100):
    """
    Generate training data for the Othello game using MCTS.
    num_samples: Number of samples to generate.
    Returns a list of tuples (board,winner).
    """
    model = keras.models.load_model(MODEL_PATH)
    data = []

    for _ in range(num_samples):
        board = OthelloBoard()
        node = Node(board, model=model)

        # Perform MCTS to get a sample tree
        node.UCT_search(1000)  

        _generate_data(node, data)

    return data

def save_data(data, filename='data/othello_train_data.npz'):
    """
    Save the generated data to a file.
    data: List of tuples (board, winner).
    filename: Name of the file to save the data.
    """
    inputs = np.array([board.board for board, _ in data])
    labels = np.array([label for _, label in data])
    np.savez(filename, X=inputs, y=labels)
    print(f"Data saved to {filename}")