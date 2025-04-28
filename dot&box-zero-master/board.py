import math
import copy
import numpy as np

import config


# initiallize board with shape (2*N-1) * (2*N-1)
def initialize_board():
    board = np.zeros((config.N_data, config.N_data), dtype=float)

    for i in range(config.N_data):
        for j in range(config.N_data):
            if i % 2 == 0 and j % 2 == 0:
                board[i][j] = config.vertex
            elif i % 2 == 1 and j % 2 == 1:
                board[i][j] = config.empty_box

    return board


class Board:
    def __init__(self, board=initialize_board(), blue=initialize_board(), red=initialize_board()):
        self.board = board
        self.blue = blue
        self.red = red
        self.next = False

    # update the state of the box
    def UpdateBox(self, player, board):
        self.next = False

        for i in range(config.N - 1):
            for j in range(config.N - 1):
                box_i = 2*i + 1
                box_j = 2*j + 1

                # check if box's four edges are legal edges
                if (board[box_i][box_j] == config.empty_box):

                    if (board[box_i-1][box_j] != config.legal_edge and
                        board[box_i+1][box_j] != config.legal_edge and
                        board[box_i][box_j-1] != config.legal_edge and
                            board[box_i][box_j+1] != config.legal_edge):

                        # update the state of each box
                        board[box_i][box_j] = config.blue_box if player == config.blue else config.red_box
                        self.next = True

        return board

    # split board into blue board, and red board
    def split_board(self, board):
        blue = initialize_board()
        red = initialize_board()

        for i in range(config.N_data):
            for j in range(config.N_data):
                if board[i][j] == config.blue_edge or board[i][j] == config.blue_box:
                    blue[i][j] = board[i][j]
                if board[i][j] == config.red_edge or board[i][j] == config.red_box:
                    red[i][j] = board[i][j]

        return blue, red

    # make the move, and update state of the board
    def make_move(self, player, move):
        if move == config.pass_move:
            return Board(self.board, self.blue, self.red)

        row = math.floor(move/(config.N_data))
        col = move % (config.N_data)
        board = copy.deepcopy(self.board)

        board[row][col] = config.blue_edge if player == config.blue else config.red_edge

        board = self.UpdateBox(player, board)
        blue, red = self.split_board(board)

        return Board(board, blue, red)

    # return legal moves
    def get_legal_moves(self):
        legal_moves = np.zeros(config.board_length, dtype=float)
        for i in range(config.N_data):
            for j in range(config.N_data):
                if self.board[i][j] == config.legal_edge:
                    legal_moves[i*(config.N_data)+j] = 1

        if np.sum(legal_moves) == 0:
            return np.concatenate((legal_moves, [1]))

        else:
            return np.concatenate((legal_moves, [0]))

    # return self's board, and enemy's board
    def get_own_and_enemy(self, player):
        if player == config.blue:
            return self.blue, self.red

        else:
            return self.red, self.blue

    def count_score(self, player):
        score = 0
        box = config.blue_box if player == config.blue else config.red_box

        for i in range(config.N_data):
            for j in range(config.N_data):
                if self.board[i][j] == box:
                    score = score+1

        return score
