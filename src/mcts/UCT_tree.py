import numpy as np
import sys
import os

# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.board.othello_board import OthelloBoard


EXPLORATION_WEIGHT = 0.7
NITER = 1000  # Number of iterations for UCT search

class Node():
    def __init__(self, board,parent=None,in_action=None):
        """
        Initialize a node in the MCTS tree.
        board: The board state represented by this node. It's an instace of OthelloBoard.
        parent: The parent node in the tree.
        in_action: The action that led to this state from the parent node.
        """
        self.board = board 
        self.parent = parent
        self.children = []
        self.visits = 0
        self.reward = 0
        self.in_action = in_action



    def is_fully_expanded(self):
        return len(self.children) == len(self.board.get_available_moves())



    def best_child(self,ew=EXPLORATION_WEIGHT):
        if not self.children:
            return None
        return max(self.children, key=lambda child: child.reward / child.visits + ew * np.sqrt(2 * np.log(self.visits) / child.visits))



    def tree_policy(self):
        """
        Traverse the tree according to the UCT policy until a leaf node is reached.
        """
        current_node = self
        while not current_node.board.is_game_over():
            if not current_node.is_fully_expanded():
                return current_node.expand()
            current_node = current_node.best_child()
        return current_node



    def expand(self):
        """
        Expand the node by adding a new child node for an unvisited action.
        Returns the newly created child node.
        """
        available_moves = self.board.get_available_moves()
        already_used_moves = [child.in_action for child in self.children]
        selected_move = None

        while selected_move is None:
            selected_move = available_moves.pop() ## Should never run out of moves, since it would mean it's fully expanded
            if selected_move in already_used_moves:
                selected_move = None

        new_board_state = self.board.simulate_move(self.board.board ,selected_move[0], selected_move[1],self.board.get_turn())
        new_board = OthelloBoard(board=new_board_state, turn=1 if self.board.get_turn() == 2 else 2, winner=self.board.get_winner())
        new_node = Node(new_board, parent=self, in_action=selected_move)
        self.children.append(new_node)
        return new_node
    
    def default_policy(self):
        """
        This will be the neural network evaluation of the board state.
        For now, we will simply return a random value.
        """
        return np.random.rand()
    
    def backup(self, reward):
        current_node = self
        while current_node is not None:
            current_node.visits += 1
            current_node.reward += reward
            reward = -reward  # In a zero-sum game, the reward for the parent is the negative of the child's reward
            current_node = current_node.parent

    def UCT_search(self,iters=NITER):
        """
        Perform UCT search for a given number of iterations.
        """
        for _ in range(iters):
            leaf = self.tree_policy()
            reward = leaf.default_policy()
            leaf.backup(reward)
        return self.best_child(0) ## Return the best child without exploration weight for the final decision

    def __str__(self):
        return f"Node(board={self.board} \n visits={self.visits}, reward={self.reward}, in_action={self.in_action})"