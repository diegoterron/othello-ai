import random
import pygame
import sys
import os
import numpy as np
import tensorflow as tf


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow warnings

from src.board.othello_board import OthelloBoard
from src.mcts.UCT_tree import Node, EXPLORATION_WEIGHT

import numpy as np
from src.board.othello_board import OthelloBoard
from src.mcts.UCT_tree import Node
from tensorflow import keras

MODEL_PATH = 'models/model_v5.keras'

# replace with None to test interface, since model is very laggy
model = tf.keras.models.load_model(MODEL_PATH) 

SIZE = 8
WIDTH, HEIGHT = 640, 640
CELL_SIZE = WIDTH // SIZE
WHITE = (220, 232, 221)
BLACK = (6, 15, 6)
GREEN = (64, 156, 68)
DARK_GREEN = (61, 148, 65)
GRAY = (25, 79, 25)
BLUE = (0, 0, 255)
LIGHT_BLUE = (100, 100, 255)
RED = (255, 0, 0)

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Othello: Human vs AI")
font = pygame.font.SysFont(None, 36)


def get_best_move(board: OthelloBoard, model) -> tuple:
    """
    Return the best move (row, col) using MCTS or fallback to a random move.
    """
    moves = board.get_available_moves()
    if not moves:
        return None

    try:
        root = Node(board, model=model)
        best_child = root.UCT_search(iters=100)
        return best_child.in_action if best_child else random.choice(moves)
    except Exception as e:
        print("MCTS failed:", e)
        return random.choice(moves)

def update_cursor(board: OthelloBoard):
    mouse_x, mouse_y = pygame.mouse.get_pos()
    row = mouse_y // CELL_SIZE
    col = mouse_x // CELL_SIZE
    if (row, col) in board.get_available_moves():
        pygame.mouse.set_cursor(pygame.SYSTEM_CURSOR_HAND)
    else:
        pygame.mouse.set_cursor(pygame.SYSTEM_CURSOR_ARROW)

def draw_board(board: OthelloBoard, show_moves=True):
    for x in range(SIZE):
        for y in range(SIZE):
            rect = pygame.Rect(y * CELL_SIZE, x * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            color = GREEN if (x + y) % 2 == 0 else DARK_GREEN
            pygame.draw.rect(screen, color, rect)  
            pygame.draw.rect(screen, BLACK, rect, 1)  

            val = board.board[x][y]
            if val == 1:
                pygame.draw.circle(screen, WHITE, rect.center, CELL_SIZE // 2 - 5)
            elif val == 2:
                pygame.draw.circle(screen, BLACK, rect.center, CELL_SIZE // 2 - 5)
            elif val == 3 and show_moves:
                pygame.draw.circle(screen, GRAY, rect.center, 5)
    
    pygame.display.flip()

def get_cell(pos):
    x, y = pos
    return y // CELL_SIZE, x // CELL_SIZE  # row, col

def human_move(board: OthelloBoard):
    waiting = True
    while waiting:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                row, col = get_cell(pygame.mouse.get_pos())
                if (row, col) in board.get_available_moves():
                    board.place_disc(row, col, board.get_turn())
                    waiting = False
        draw_board(board)

def ai_move(board: OthelloBoard):
    move = get_best_move(board, model)
    if move:
        board.place_disc(move[0], move[1], board.get_turn())

def render_text_with_outline(font, text, text_color, outline_color, outline_width=2):
    base = font.render(text, True, text_color)
    size = (base.get_width() + 2*outline_width, base.get_height() + 2*outline_width)
    img = pygame.Surface(size, pygame.SRCALPHA)

    # Draw outline by rendering text shifted in 8 directions
    for dx in range(-outline_width, outline_width+1):
        for dy in range(-outline_width, outline_width+1):
            if dx == 0 and dy == 0:
                continue
            offset_pos = (dx + outline_width, dy + outline_width)
            img.blit(font.render(text, True, outline_color), offset_pos)

    # Draw main text in center
    img.blit(base, (outline_width, outline_width))
    return img

def show_winner(winner, board):
    
    if winner == 0:
        text = "Draw!"
    elif winner == 1:
        if board.turn == 1:
            text = "Black (You) win!"
        else:
            text = "White (AI) wins!"
    elif winner == -1:
        if board.turn == 1:
            text = "White (AI) wins!"
        else:
            text = "Black (You) win!"
    else:
        text = "Game over."

    img = render_text_with_outline(font, text, WHITE, BLACK, outline_width=2)
    screen.blit(img, (WIDTH // 2 - img.get_width() // 2, HEIGHT // 2 - img.get_height() // 2))
    pygame.display.flip()

def main():
    board = OthelloBoard()
    human_color = 2
    ai_color = 1
    running = True

    while running:
        if board.get_turn() == ai_color and not board.is_game_over():
            if board.get_available_moves():
                ai_move(board)
            else:
                board.turn = human_color

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                

            if board.get_turn() == human_color and event.type == pygame.MOUSEBUTTONDOWN:
                row, col = get_cell(pygame.mouse.get_pos())
                if (row, col) in board.get_available_moves():
                    board.place_disc(row, col, human_color)

        if not board.get_available_moves():
            board.turn = 1 if board.get_turn() == 2 else 2

        update_cursor(board)

        show_moves = board.get_turn() == human_color
        draw_board(board, show_moves=show_moves)

        if board.is_game_over():
            board.check_game_over()
            show_winner(board.get_winner(), board)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
            
    
    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
