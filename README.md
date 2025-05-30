## Installation

```bash
pip install -r requirements.txt
```

## Usage

```python
from othello import OthelloBoard

board = OthelloBoard()

board.print_board()  # Show current board
print(board.get_turn())  # Get current player
print(board.get_available_moves())  # Get valid moves for current player

# Make a move
row, col = board.get_available_moves()[0]
board.place_disc(row, col, board.get_turn())
board.print_board()

# Check winner after game ends
print(board.get_winner())
```

## Public Methods

### `get_board()`
Returns a copy of the current board state as a numpy array.

### `get_turn()`
Returns the current player's turn (`1` for white, `2` for black).

### `get_winner()`
Returns the result relative to the current turn:
- `1` if current player wins
- `-1` if opponent wins
- `0` for a tie or ongoing game

### `place_disc(row, col, color)`
Attempts to place a disc at the given position and flips affected discs. Updates the board and changes turn if move is valid.

### `simulate_move(board_state, row, col, color)`
Returns a simulated board state as a numpy array that reflects what would happen if a disc is placed at the given position.

### `get_available_moves()`
Returns a list of `(row, col)` tuples where the current player can legally place a disc.

### `check_game_over()`
Checks if the game has ended and updates the `winner` attribute accordingly.

### `print_board(board_state=None, title=\"Current Board\")`
Prints the current board (or a specified board) to the console with colors:
- Grey: `0` (empty)
- Blue: `1` (white disc)
- Red: `2` (black disc)
- Green: `3` (available move)
