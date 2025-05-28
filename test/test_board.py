from colorama import init, Fore, Style

import sys
import os

# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.board.othello_board import OthelloBoard

# Initialize colorama
init(autoreset=True)

def print_board_state(board_obj,title):
    color_map = {
        0: Fore.LIGHTBLACK_EX + "0",  # Grey
        1: Fore.BLUE + "1",           # Blue
        2: Fore.RED + "2",            # Red
        3: Fore.GREEN + "3",          # Green
    }
    print(title)
    print(f"\nPlayer {board_obj.get_turn()}'s turn:")
    for row in board_obj.get_board():
        print(" ".join(color_map[val] for val in row))
    print(Style.RESET_ALL)

# Initialize the board
board = OthelloBoard()

# Print initial state
print_board_state(board, "Initial")

board.print_board(board.get_board(), "After Initial Setup")

# Find and play one available move
available_moves = list(zip(*((board.board == 3).nonzero())))
if available_moves:
    row, col = available_moves[0]
    board.place_disc(row, col, board.get_turn())
    board.print_board(board.get_board(),f"After Player {2 if board.get_turn() == 1 else 1} Move at ({row}, {col})")

available_moves = list(zip(*((board.board == 3).nonzero())))
if available_moves:
    row, col = available_moves[2]
    board.place_disc(row, col, board.get_turn())
    board.print_board(board.get_board(),f"After Player {2 if board.get_turn() == 1 else 1} Move at ({row}, {col})")

available_moves = list(zip(*((board.board == 3).nonzero())))
if available_moves:
    row, col = available_moves[2]
    board.place_disc(row, col, board.get_turn())
    board.print_board(board.get_board(),f"After Player {2 if board.get_turn() == 1 else 1} Move at ({row}, {col})")

available_moves = list(zip(*((board.board == 3).nonzero())))
if available_moves:
    row, col = available_moves[2]
    board.place_disc(row, col, board.get_turn())
    board.print_board(board.get_board(),f"After Player {2 if board.get_turn() == 1 else 1} Move at ({row}, {col})")

available_moves = list(zip(*((board.board == 3).nonzero())))
if available_moves:
    row, col = available_moves[2]
    board.place_disc(row, col, board.get_turn())
    board.print_board(board.get_board(),f"After Player {2 if board.get_turn() == 1 else 1} Move at ({row}, {col})")

available_moves = list(zip(*((board.board == 3).nonzero())))
if available_moves:
    row, col = available_moves[2]
    board.place_disc(row, col, board.get_turn())
    board.print_board(board.get_board(),f"After Player {2 if board.get_turn() == 1 else 1} Move at ({row}, {col})")

# Print winner (should be 0 until board is full)
print("\nWinner:", board.get_winner())
