import numpy as np
from colorama import Fore, Style

class OthelloBoard:
    def __init__(self, size=8,board = None, turn = None, winner = None):
        if size % 2 != 0:
            raise ValueError("Board size must be even.")
        self.size = size
        self.board = board if board is not None else np.zeros((size, size), dtype=int)
        self.turn = turn if turn else 2  # 1 for white, 2 for black
        self.winner = winner if winner else None
        if board is  None:
            self._initialize_starting_position()

        self.update_available_moves()

    def __str__(self):
        return str(self.board)

    def get_board(self):
        return self.board.copy()

    def reset_board(self):
        self.board.fill(0)
        self.turn = 2
        self.winner = 0
        self._initialize_starting_position()
        self.update_available_moves()

    def _initialize_starting_position(self):
        mid = self.size // 2
        self.board[mid-1, mid-1] = 1
        self.board[mid, mid] = 1
        self.board[mid-1, mid] = 2
        self.board[mid, mid-1] = 2

    def simulate_move(self, board_state, row, col, color):
        opponent = 2 if color == 1 else 1
        new_board = board_state.copy()
        new_board[new_board == 3] = 0  # Clear old available move hints
        new_board[row, col] = color

        directions = [(-1, -1), (-1, 0), (-1, 1),
                      (0, -1),          (0, 1),
                      (1, -1), (1, 0), (1, 1)]

        for dr, dc in directions:
            r, c = row + dr, col + dc
            path = []

            while 0 <= r < self.size and 0 <= c < self.size:
                if new_board[r, c] == opponent:
                    path.append((r, c))
                elif new_board[r, c] == color:
                    for pr, pc in path:
                        new_board[pr, pc] = color
                    break
                else:
                    break
                r += dr
                c += dc

        # Recalculate available moves for the next player
        next_turn = 2 if color == 1 else 1
        new_board = self.update_available_moves_for_board(new_board, next_turn)

        return new_board

    def place_disc(self, row, col, color):
        if color not in (1, 2):
            raise ValueError("Color must be 1 (white) or 2 (black).")
        if color != self.turn:
            raise ValueError(f"It's not player {color}'s turn.")
        if not (0 <= row < self.size and 0 <= col < self.size):
            raise IndexError("Position out of bounds.")
        if self.board[row, col] != 3:
            raise ValueError("Invalid move. Not a legal position.")

        self.board = self.simulate_move(self.board, row, col, color)
        self.turn = 2 if self.turn == 1 else 1
        self.check_game_over()

    def get_turn(self):
        return self.turn

    def get_winner(self):
        return self.winner

    def get_available_moves(self):
        return [(r, c) for r in range(self.size) for c in range(self.size) if self.board[r, c] == 3]

    def update_available_moves(self):
        self.board = self.update_available_moves_for_board(self.board, self.turn)

    def update_available_moves_for_board(self, board, turn):
        board = board.copy()
        board[board == 3] = 0  # Clear old available move hints
        opponent = 2 if turn == 1 else 1
        directions = [(-1, -1), (-1, 0), (-1, 1),
                      (0, -1),          (0, 1),
                      (1, -1), (1, 0), (1, 1)]

        for row in range(self.size):
            for col in range(self.size):
                if board[row, col] != 0:
                    continue
                for dr, dc in directions:
                    r, c = row + dr, col + dc
                    found_opponent = False
                    while 0 <= r < self.size and 0 <= c < self.size:
                        if board[r, c] == opponent:
                            found_opponent = True
                        elif board[r, c] == turn:
                            if found_opponent:
                                board[row, col] = 3  # Valid move
                            break
                        else:
                            break
                        r += dr
                        c += dc
        return board

    def is_game_over(self):
        return not np.any(self.board == 3)
    

    def check_game_over(self):
        if self.is_game_over():
            result = self._calculate_winner()
            if result == 0:
                self.winner = 0
            elif result == self.turn:
                self.winner = -1  # Opponent won
            else:
                self.winner = 1   # Current player won
            

    def _calculate_winner(self):
        white_count = np.count_nonzero(self.board == 1)
        black_count = np.count_nonzero(self.board == 2)
        if white_count > black_count:
            return 1
        elif black_count > white_count:
            return 2
        else:
            return 0  # Draw

    def print_board(self, board_state=None, title="Current Board"):
        board_state = board_state if board_state is not None else self.board
        color_map = {
            0: Fore.LIGHTBLACK_EX + "0",
            1: Fore.BLUE + "1",
            2: Fore.RED + "2",
            3: Fore.GREEN + "3",
        }
        print(title)
        print(f"\nPlayer {self.get_turn()}'s turn:")
        for row in board_state:
            print(" ".join(color_map[val] for val in row))
        print(Style.RESET_ALL)
