import numpy as np
import sys
import os

# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from board.othello_board import OthelloBoard


EXPLORATION_WEIGHT = 0.5
NITER = 1000  # Number of iterations for UCT search

class Node():
    def __init__(self, board,parent=None,in_action=None,model = None):
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
        self.model = model
        self.prior_probs = {}



    def is_fully_expanded(self):
        self.board.update_available_moves()
        return len(self.children) == len(self.board.get_available_moves())



    def best_child(self, ew=EXPLORATION_WEIGHT):
        if not self.children:
            return None

        best_score = float('-inf')
        best = None

        total_visits = self.visits
        for child in self.children:
            s1 = child.reward / child.visits if child.visits > 0 else 0

            # Modify the standard UCT formula as they did in alpha-zero
            # p is the prior probability of the action leading to this child. If in the root node, noise has been applied
            p = self.prior_probs.get(child.in_action, 1 / len(self.children))
            s2 = ew * p * np.sqrt(total_visits) / (1 + child.visits)
            score = s1 + s2

            if score > best_score:
                best_score = score
                best = child

        return best



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
        new_board = OthelloBoard(board=new_board_state, turn=1 if self.board.get_turn() == 2 else 2)
        new_node = Node(new_board, parent=self, in_action=selected_move, model=self.model)
        self.children.append(new_node)
        return new_node
    
    def default_policy(self):
        """
        Use a neural network to evaluate the board state or return a random value. If no Neural Network is provided, use the traditional default policy implementation
        """
        if self.model is not None:
            self.prediction =  self.model.predict(np.expand_dims(self.board.get_board(), axis=0),verbose = 0)[0][0]
            return self.prediction
        else:
            current_board = OthelloBoard(board=self.board.board, turn=self.board.get_turn())
            while current_board.is_game_over() is False:
                available_moves = current_board.get_available_moves()
                if not available_moves:
                    new_board = OthelloBoard(board=current_board.board, turn=1 if current_board.get_turn() == 2 else 2)
                else:
                    move = available_moves[np.random.choice([i for i,_ in enumerate(available_moves)])]
                    new_board_state = current_board.simulate_move(current_board.board, move[0], move[1], current_board.get_turn())
                    new_board = OthelloBoard(board=new_board_state, turn=1 if current_board.get_turn() == 2 else 2)
                current_board = new_board
            current_board.check_game_over()
            return current_board.get_winner()


    
    def backup(self, reward):
        current_node = self
        while current_node is not None:
            current_node.visits += 1
            current_node.reward += reward
            reward = -reward  # In a zero-sum game, the reward for the parent is the negative of the child's reward
            current_node = current_node.parent

    def apply_dirichlet_noise(self, epsilon=0.25, alpha=0.3):
        available_moves = self.board.get_available_moves()
        noise = np.random.dirichlet([alpha] * len(available_moves))
        self.prior_probs = {
            move: (1 - epsilon) * (1 / len(available_moves)) + epsilon * noise[i]
            for i, move in enumerate(available_moves)
        }

    def UCT_search(self,iters=NITER,noisy=False):
        """
        Perform UCT search for a given number of iterations.
        """
        self.board.update_available_moves()
        if self.board.get_available_moves() and noisy:
            self.apply_dirichlet_noise()

        for _ in range(iters):
            leaf = self.tree_policy()
            reward = leaf.default_policy()
            leaf.backup(reward)
        return self.best_child(0) ## Return the best child without exploration weight for the final decision

    def __str__(self):
        return f"Node(board={self.board} \n visits={self.visits}, reward={self.reward}, in_action={self.in_action})"