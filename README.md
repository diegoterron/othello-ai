## Installation

```bash
pip install -r requirements.txt
```

## Usage

Run the AI-driven Othello game by executing the main script after installing dependencies:

```bash
python src/game/main.py
```

## Public Methods

### From `OthelloBoard` class (`src/board/othello_board.py`)

#### `get_board()`
Returns a copy of the current board state as a numpy array.

#### `get_turn()`
Returns the current player's turn (`1` for white, `2` for black).

#### `get_winner()`
Returns the result relative to the current turn:
- `1` if current player wins
- `-1` if opponent wins
- `0` for a tie or ongoing game
- `None` if the game has not ended yet

#### `place_disc(row, col, color)`
Attempts to place a disc at the given position. Validates turn, updates the board, flips discs, and switches turns. Raises errors for invalid moves.

#### `simulate_move(board_state, row, col, color)`
Returns a simulated board state as a numpy array, showing what the board would look like if a disc is placed at the specified position for the given color.

#### `get_available_moves()`
Returns a list of `(row, col)` tuples where the current player can legally place a disc.

#### `check_game_over()`
Checks if the game has ended. If so, sets the `winner` attribute relative to the current player:
- `1` if current player wins
- `-1` if opponent wins
- `0` for a tie

#### `print_board(board_state=None, title="Current Board")`
Prints the current board (or a specified board) to the console with colored values:
- Grey: `0` (empty)
- Blue: `1` (white disc)
- Red: `2` (black disc)
- Green: `3` (available move)

---

### From `Node` class (`src/mcts/UCT_tree.py`)

#### `UCT_search(iters=1000, noisy=False)`
Performs a Monte Carlo Tree Search (MCTS) using Upper Confidence bounds applied to Trees (UCT) for a specified number of iterations (`iters`).

- Runs simulations to expand and evaluate the game tree.
- Optionally applies Dirichlet noise at the root node for exploration (`noisy=True`).
- Returns the best child node representing the best next move without exploration weight.

---

## Folder Structure

```
src/
├── board/
│   └── othello_board.py          # OthelloBoard implementation
├── game/
│   └── main.py                   # Entry point to run the AI Othello game
└── mcts/
    └── UCT_tree.py               # MCTS Node class with UCT_search method
```