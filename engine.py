# packages
import chess
from copy import deepcopy
import sys
import random
import time
import os
os.chdir(os.path.dirname(__file__))

CACHE_SIZE = 200000
MINTIME = 0.1
TIMEDIV = 25.0
NODES = 800
C = 3.0


import numpy as np
import chess
import chess.engine
import chess.pgn
import chess.svg
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import os



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

def load_checkpoint(model, optimizer=None, filename="my_checkpoint.pth.tar"):
   
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint["state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
    
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ChessModelWho(nn.Module):
    def __init__(self):
        super(ChessModelWho, self).__init__()
        
        self.conv1_board = nn.Conv2d(12, 32, 3, padding=1)
        self.conv2_board = nn.Conv2d(32, 18, 2)
        self.fc0_board = nn.Linear(18*7*7, 512)
        
        self.conv1_move = nn.Conv2d(12, 32, 3, padding=1)
        self.conv2_move = nn.Conv2d(32, 18, 2)
        self.fc0_move = nn.Linear(18*7*7, 512)
        
        self.fc1_board_move = nn.Linear(2*512, 512)
        self.fc2_who = nn.Linear(512, 256)
        self.fc3_who = nn.Linear(256, 8*8)
        self.relu = F.relu # lambda x:x
    
    def forward(self, board_tensor, move_tensor):
        N = board_tensor.size(0)
         
        x1 = self.relu(self.conv1_board(board_tensor))
        x1 = self.relu(self.conv2_board(x1))
        x1 = self.relu(self.fc0_board(x1.view(N, -1)))
        
        x2 = self.relu(self.conv1_move(move_tensor))
        x2 = self.relu(self.conv2_move(x2))
        x2 = self.relu(self.fc0_move(x2.view(N, -1)))
        
        x = torch.concat((x1, x2), dim=1).view(N, -1)
        x = self.relu(self.fc1_board_move(x))
        x = self.relu(self.fc2_who(x))
        
        return self.fc3_who(x)
    
class ChessModelWhere(nn.Module):
    def __init__(self):
        super(ChessModelWhere, self).__init__()
        
        self.pool = nn.MaxPool2d(kernel_size=2)
        
        self.conv1_board = nn.Conv2d(12, 32, 3, padding=1)
        self.conv2_board = nn.Conv2d(32, 18, 2)
        
        self.conv1_move = nn.Conv2d(12, 32, 3, padding=1)
        self.conv2_move = nn.Conv2d(32, 18, 2)
        
        self.fc1_board_move = nn.Linear(2*18*7*7+64, 512)
        self.fc2_who = nn.Linear(512, 8*8)
        self.relu = F.relu # lambda x:x
    
    def forward(self, board_tensor, move_tensor, who): # who is target_who Nx64
        N = board_tensor.size(0)
        x1 = self.relu(self.conv1_board(board_tensor))
        # x1 = self.pool(x1)
        x1 = self.relu(self.conv2_board(x1))
        x2 = self.relu(self.conv1_move(move_tensor))
        # x2 = self.pool(x2)
        x2 = self.relu(self.conv2_move(x2))
        x = torch.concat((x1, x2), dim=1).view(N, -1)
        x = torch.concat((x, who), dim=1)
        x = self.relu(self.fc1_board_move(x))
        return self.fc2_who(x)
last_move = None
last_last_move = None
def predict(modelwho, modelwhere, board:chess.Board, forceRight=True, random_threshold=.9):
    global last_move, last_last_move
    legal_moves = list(board.legal_moves)
    # assert(len(legal_moves)!=0)
    board_tensor, move_tensor = board2vec(board)
    board_tensor = torch.from_numpy(board_tensor).unsqueeze(0).to(device)
    move_tensor = torch.from_numpy(move_tensor).unsqueeze(0).to(device)
    
    with torch.no_grad():
        who = modelwho(board_tensor, move_tensor)
    who = who.squeeze().cpu().numpy()
    if forceRight:
        who_map = [i.from_square for i in legal_moves]
        who = sorted([(i, idx) for idx, i in enumerate(who) if idx in who_map], key=lambda x: x[0], reverse=True)
    else:
        who = sorted([(i, idx) for idx, i in enumerate(who)], key=lambda x: x[0], reverse=True)

    who_prob = who
    who = random.choices([i[1] for i in who_prob if i[0]>=who_prob[0][0]*random_threshold], 
                             [i[0] for i in [i for i in who_prob if i[0]>=who_prob[0][0]*random_threshold]])[0]

    # assert who in [i.from_square for i in legal_moves]
    who_one_hot = np.zeros((64), dtype=np.float32)
    who_one_hot[who] = 1
    who_one_hot = torch.from_numpy(who_one_hot).unsqueeze(0).to(device)
    
    with torch.no_grad():
        where = modelwhere(board_tensor, move_tensor, who_one_hot)
    where = where.squeeze().cpu().numpy()
    
    if forceRight:
        where_map = [i.to_square for i in legal_moves if i.from_square == who]
        where = sorted([(i, idx) for idx, i in enumerate(where) if idx in where_map], key=lambda x: x[0], reverse=True)
    else:
        where = sorted([(i, idx) for idx, i in enumerate(where)], key=lambda x: x[0], reverse=True)

    where_prob = where
    where = random.choices([i[1] for i in where_prob if i[0]>=where_prob[0][0]*random_threshold], 
                               [i[0] for i in [i for i in where_prob if i[0]>=where_prob[0][0]*random_threshold]])[0]
    
    f,t = chess.square_name(who), chess.square_name(where)
    move = chess.Move.from_uci(f + t)
    if t in ['a1', 'b1', 'c1', 'd1', 'e1', 'f1', 'g1', 'h1', 'a8', 'b8', 'c8', 'd8', 'e8', 'f8', 'g8', 'h8']:
        if (board.piece_at(who) == chess.Piece(chess.PAWN, chess.WHITE)):
            move = chess.Move.from_uci(f + t + 'Q')
        elif (board.piece_at(who) == chess.Piece(chess.PAWN, chess.BLACK)):
            move = chess.Move.from_uci(f + t + 'q')
    
    

    #TODO: add force three move probabilities
    # if move == last_last_move:
    #     if (len(who_prob)==1):
    #         who = where_prob
    #     else:
    #         where = where_prob[1][1]
    #     f,t = chess.square_name(who), chess.square_name(where)
    #     move = chess.Move.from_uci(f + t)
    #     if t in ['a1', 'b1', 'c1', 'd1', 'e1', 'f1', 'g1', 'h1', 'a8', 'b8', 'c8', 'd8', 'e8', 'f8', 'g8', 'h8']:
    #         if (board.piece_at(who) == chess.Piece(chess.PAWN, chess.WHITE)):
    #             move = chess.Move.from_uci(f + t + 'Q')
    #         elif (board.piece_at(who) == chess.Piece(chess.PAWN, chess.BLACK)):
    #             move = chess.Move.from_uci(f + t + 'q')
    
    # assert move in legal_moves
    
    last_last_move = last_move
    last_move = move
    
    return move
modelWHO = ChessModelWho().to(device)
modelWhere = ChessModelWhere().to(device)

# load_checkpoint(modelWHO,  filename='models\\modelWho_1000-10000_epoch_100_lr_002_accuracy_valid_35_ckpt.pth.tar')
# load_checkpoint(modelWHO,  filename='models\\modelWho_10000-20000_epoch_100_lr_002_accuracy_valid_31_ckpt.pth.tar')
# load_checkpoint(modelWhere, filename='models\\modelWhere_1000-10000_epoch_100_lr_002_accuracy_valid_60_ckpt.pth.tar')
load_checkpoint(modelWHO,  filename='models\\modelWho_100000_epoch_100_lr_002_accuracy_valid__ckpt.pth.tar')
load_checkpoint(modelWhere, filename='models\\modelWhere_100000_epoch_100_lr_002_accuracy_valid__ckpt.pth.tar')

# board state class
class State:
    # init board state instance
    def __init__(self, opponent=False):
        # assign chess board instance to current state from scratch
        self.board = chess.Board()
        
        # assign chess board instance to current state from existing position
        if opponent:
            self.board = deepcopy(opponent.board)
    
    # get whether the game is in the terminal state (win/draw/loss) or not
    def is_terminal(self):
        return self.board.is_game_over()
    
    # generate states (generate legal moves)
    def generate_states(self):
        # legal actions (moves) to consider in current position
        actions = []
        
        # generate legal moves
        moves = self.board.legal_moves
        
        # loop over legal moves
        for move in moves:
            # append move to action list
            actions.append(str(move))
        
        # return list of available actions (moves)
        return actions
    
    # take action (make move on board)
    def take_action(self, action):
        # create new state instance from the current state
        new_state = State(self)
        
        # take action (make move on board)
        new_state.board.push(chess.Move.from_uci(action))
        
        # return new state with action naken on board
        return new_state

    # output current state's board position
    def __str__(self):
        # for windows users
        #return self.board.__str__()
        return self.board.unicode().replace('⭘', '.')

def send(str):
    with open('log.txt', 'a') as f: f.write(f"{str}\n")
    sys.stdout.write(str)
    sys.stdout.write("\n")
    sys.stdout.flush()

def process_position(tokens):
    board = chess.Board()

    offset = 0

    if tokens[1] ==  'startpos':
        offset = 2
    elif tokens[1] == 'fen':
        fen = " ".join(tokens[2:8])
        board = chess.Board(fen=fen)
        offset = 8

    if offset >= len(tokens):
        return board

    if tokens[offset] == 'moves':
        for i in range(offset+1, len(tokens)):
            board.push_uci(tokens[i])

    # deal with cutechess bug where a drawn positions is passed in
    if board.can_claim_draw():
        board.clear_stack()

    return board

def uci_loop(state):
    while True:
        try:
            line = sys.stdin.readline()
            line = line.rstrip()
            tokens = line.split()
            if len(tokens) == 0:
                continue
            
            with open('log.txt', 'a') as f: f.write(f"{tokens}\n")
            if tokens[0] == "uci":
                send('id name ThefCraft Chess')
                send('id author ThefCraft')
                send('uciok')
            elif tokens[0] == "quit": exit(0)
            elif tokens[0] == "isready": send("readyok")
            elif tokens[0] == "ucinewgame":
                state.board = chess.Board()
                # print(state)

            elif tokens[0] == 'position':
                state.board = process_position(tokens)
                # print(state)

            elif tokens[0] == 'go':
                my_nodes = NODES
                my_time = None
                if (len(tokens) == 3) and (tokens[1] == 'nodes'): my_nodes = int(tokens[2])
                if (len(tokens) == 3) and (tokens[1] == 'movetime'):
                    my_time = int(tokens[2])
                    if my_time < MINTIME:
                        my_time = MINTIME
                if (len(tokens) == 9) and (tokens[1] == 'wtime'):
                    wtime = int(tokens[2])
                    btime = int(tokens[4])
                    winc = int(tokens[6])
                    binc = int(tokens[8])
                    if (wtime > 5*winc):
                        wtime += 5*winc
                    else:
                        wtime += winc
                    if (btime > 5*binc):
                        btime += 5*binc
                    else:
                        btime += binc
                    if state.board.turn:
                        my_time = wtime/(TIMEDIV*1000.0)
                    else:
                        my_time = btime/(TIMEDIV*1000.0)
                    if my_time < MINTIME:
                        my_time = MINTIME


                if my_time != None:
                    # search with time limit per move
                    # mcts = MCTS(timeLimit=my_time)
                    best_move, score = (random.choice(list(state.board.legal_moves)), random.randint(-999, 999))#mcts.search(state)
    
                else:          
                    # search placeholder for various time controls
                    # mcts = MCTS(timeLimit=1000)
                    best_move, score = (random.choice(list(state.board.legal_moves)), random.randint(-999, 999))#mcts.search(state)
                    
                    
                time.sleep(.1)
                best_move = predict(modelWHO, modelWhere, state.board, forceRight=True)
                score = 0

                # return best move to the GUI
                send('info score cp %s' % score)
                send('bestmove %s' % best_move)

        except Exception as e: 
            with open('log.txt', 'a') as f: f.write(f"\tERROR!!! : {e}\n")

state = State()
uci_loop(state)