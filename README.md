# torchchess-elo800
torchchess-elo800 is an entry-level chess engine implemented in Python using the PyTorch library. The engine is designed to achieve an estimated Elo rating of around 800, making it suitable for beginners and educational purposes. The engine utilizes neural networks powered by PyTorch for move generation, position evaluation, and decision-making.

## Key Features:
- Entry-level chess engine targeting an Elo rating of 800.
- PyTorch-based neural networks for move generation and position evaluation.
- Implements the UCI (Universal Chess Interface) protocol for compatibility with chess GUIs.
- Simple search algorithm with basic evaluation functions.
- Designed for educational purposes, providing insights into chess engine development.

## How to Run:
1. **Clone the Repository:**
   ```bash
   git clone https://github.com/yourusername/torchchess-elo800.git
   cd torchchess-elo800
   ```

2. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download CuteChess:**
   Download and install [CuteChess](https://github.com/cutechess/cutechess/releases) on your machine.

4. **Configure CuteChess:**
   - Open CuteChess and go to the "Engines" tab.
   - Click on "Add" and provide the following information:
     - Name: torchchess-elo800
     - Command: `{python_path} {path_for_engine.py}`
     - For example: `python3 ./src/engine.py`
   - Save the configuration.

5. **Start a Game:**
   - Create a new chess game in CuteChess.
   - Select the "torchchess-elo800" engine as one of the opponents.
   - Start the game and enjoy playing against the torchchess-elo800 engine.

### Contributions:
Contributions are welcome! Feel free to submit pull requests, report issues, or contribute to the project's improvement. Whether you're a beginner or an experienced developer, your contributions can help enhance the engine's features and performance.

### Disclaimer:
This chess engine is primarily intended for educational purposes and is not intended to compete with advanced engines like Stockfish. It serves as a starting point for those interested in understanding the basics of chess engine development using PyTorch.

### Note:
Please refer to the documentation for detailed instructions, examples, and information about the engine's internals.
