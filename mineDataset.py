import pickle
from tqdm import tqdm

import chess
import chess.engine
import chess.pgn
import chess.svg

from IPython.display import SVG
import datetime 
import random

import numpy as np

# Mapping of piece types to channel indices
piece_to_channel = {'p': 0, 'n': 1, 'b': 2, 'r': 3, 'q': 4, 'k': 5,
                    'P': 6, 'N': 7, 'B': 8, 'R': 9, 'Q': 10, 'K': 11}

def board2vec(board:chess.Board)->(np.ndarray, np.ndarray):
    # Initialize an empty tensor with shape (14, 8, 8)
    board_tensor = np.zeros((12, 8, 8), dtype=np.float32)
    move_tensor = np.zeros((12, 8, 8), dtype=np.float32)
    # Iterate over the squares of the board
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is not None:
            # Get the channel index for the piece type
            channel_index = piece_to_channel[piece.symbol()]
            # Set the corresponding entry in the tensor to 1.0
            board_tensor[channel_index, chess.square_rank(square), chess.square_file(square)] = 1.0
            
            # Get legal moves for the piece
            legal_moves = [move for move in board.legal_moves if move.from_square == square]
            for move in legal_moves:
                # Set the corresponding entry in the move tensor to 1.0
                move_tensor[channel_index, chess.square_rank(move.to_square), chess.square_file(move.to_square)] = 1.0

    return board_tensor, move_tensor


engine = chess.engine.SimpleEngine.popen_uci(r"path_to_\stockfish\stockfish-windows-x86-64-avx2.exe")
board = chess.Board()


def random_board(board:chess.Board, n_random_moves=5, n_random_moves_error=1)->chess.Board:
    board.reset()
    for i in range(random.randint(int(n_random_moves*(1-n_random_moves_error)), int(n_random_moves*(1+n_random_moves_error)))):
        if board.is_game_over(): break
        move = random.choice(list(board.legal_moves))
        board.push(move)
    return board

def encode_square(square):
    encoded_square = np.zeros(64)
    encoded_square[square] = 1
    return encoded_square



X_board_tensor = []
X_move_tensor = []
Y_who = []
Y_where = []

n_random_moves = 0
n_random_moves_error = 1

GAMES = range(90000, 100000)
for i in tqdm(GAMES):
    if i==100: 
        n_random_moves = 2
        np.savez('chess_dataset_100_night.npz', X_board_tensor=np.array(X_board_tensor), X_move_tensor=np.array(X_move_tensor),
                                       Y_who=np.array(Y_who), Y_where=np.array(Y_where))
    elif i==1000: 
        n_random_moves = 5
        n_random_moves_error=.9
        # Save the dataset
        np.savez('chess_dataset_1000_night.npz', X_board_tensor=np.array(X_board_tensor), X_move_tensor=np.array(X_move_tensor),
                                       Y_who=np.array(Y_who), Y_where=np.array(Y_where))
    elif i==10000: 
        n_random_moves = 10
        n_random_moves_error=.8
        # Save the dataset
        np.savez('chess_dataset_10000_night.npz', X_board_tensor=np.array(X_board_tensor), X_move_tensor=np.array(X_move_tensor),
                                       Y_who=np.array(Y_who), Y_where=np.array(Y_where))
    elif i==30000: 
        n_random_moves = 20
        n_random_moves_error=.75
        # Save the dataset
        np.savez('chess_dataset_30000_night.npz', X_board_tensor=np.array(X_board_tensor), X_move_tensor=np.array(X_move_tensor),
                                       Y_who=np.array(Y_who), Y_where=np.array(Y_where))
    elif i==50000: 
        n_random_moves = 25
        n_random_moves_error=.7
        # Save the dataset
        np.savez('chess_dataset_50000_night.npz', X_board_tensor=np.array(X_board_tensor), X_move_tensor=np.array(X_move_tensor),
                                       Y_who=np.array(Y_who), Y_where=np.array(Y_where))
    elif i==75000: 
        n_random_moves = 30
        n_random_moves_error=.6
        # Save the dataset
        np.savez('chess_dataset_75000_night.npz', X_board_tensor=np.array(X_board_tensor), X_move_tensor=np.array(X_move_tensor),
                                       Y_who=np.array(Y_who), Y_where=np.array(Y_where))
    elif i==85000: 
        n_random_moves = 50
        n_random_moves_error=.5
        # Save the dataset
        np.savez('chess_dataset_85000_night.npz', X_board_tensor=np.array(X_board_tensor), X_move_tensor=np.array(X_move_tensor),
                                       Y_who=np.array(Y_who), Y_where=np.array(Y_where))
    
    random_board(board, n_random_moves, n_random_moves_error)
    while not board.is_game_over():
        
        move = engine.play(board, chess.engine.Limit(time=0.001)).move
        
        board_tensor, move_tensor = board2vec(board)
        X_move_tensor.append(move_tensor)
        X_board_tensor.append(board_tensor)
        
        Y_who.append(encode_square(move.from_square))
        Y_where.append(encode_square(move.to_square))
        
        board.push(move)

np.savez('chess_dataset_90000-100000_night.npz', X_board_tensor=np.array(X_board_tensor), X_move_tensor=np.array(X_move_tensor),
                                       Y_who=np.array(Y_who), Y_where=np.array(Y_where))
