import random
from Dots_and_Box import *
import numpy as np
import copy

def check_box(next_board):
    box_filled = False
    m = (len(next_board)+1)//2
    n = (len(next_board[0])+1)//2
    box_pos = None
    for i in range(m - 1):
        for j in range(n - 1):
            box_i = 2*i + 1
            box_j = 2*j + 1
            # 檢查該方格的四條邊是否都不為 0
            if (next_board[box_i][box_j] == 8 and
                next_board[box_i-1][box_j] != 0 and
                next_board[box_i+1][box_j] != 0 and
                next_board[box_i][box_j-1] != 0 and
                next_board[box_i][box_j+1] != 0):
                box_filled = True
                box_pos = (box_i, box_j)
    return box_filled, box_pos

def check_suicide(next_board):
    box_filled = False
    box_filled = False
    m = (len(next_board)+1)//2
    n = (len(next_board[0])+1)//2
    for i in range(m - 1):
        for j in range(n - 1):
            box_i = 2*i + 1
            box_j = 2*j + 1
            # 檢查該方格的四條邊是否都不為 0
            if (next_board[box_i][box_j] == 8 and
                next_board[box_i-1][box_j] == 0 and
                next_board[box_i+1][box_j] != 0 and
                next_board[box_i][box_j-1] != 0 and
                next_board[box_i][box_j+1] != 0):
                box_filled = True
            if (next_board[box_i][box_j] == 8 and
                next_board[box_i-1][box_j] != 0 and
                next_board[box_i+1][box_j] == 0 and
                next_board[box_i][box_j-1] != 0 and
                next_board[box_i][box_j+1] != 0):
                box_filled = True
            
            if (next_board[box_i][box_j] == 8 and
                next_board[box_i-1][box_j] != 0 and
                next_board[box_i+1][box_j] != 0 and
                next_board[box_i][box_j-1] == 0 and
                next_board[box_i][box_j+1] != 0):
                box_filled = True
            
            if (next_board[box_i][box_j] == 8 and
                next_board[box_i-1][box_j] != 0 and
                next_board[box_i+1][box_j] != 0 and
                next_board[box_i][box_j-1] != 0 and
                next_board[box_i][box_j+1] == 0):
                box_filled = True

    return box_filled

def find_chain(board, r, c, board_rows, board_cols, player):
    """遞迴尋找從 (i, j) 出發的鏈長度s"""
    n_board = copy.deepcopy(board)
    stack = [(r, c)]
    length = 0
    while stack:
        print(stack)
        x, y = stack.pop()
        if not isValid(n_board, x, y):
            continue
        n_board[x][y] = player
        filled, score = checkBox(n_board, player)
        # if not filled:
        #     continue
        length += score

        print_board(n_board,board_rows,board_cols)
        # 檢查相鄰格子（四周）
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ni, nj = x + dx, y + dy
            if 0 <= ni < board_rows and 0 <= nj < board_cols:
                stack.append((ni, nj))
    return length
    

def GreedAlg(board,ValidMoves,verbose = False):
    for ValidMove in ValidMoves:
        r,c = ValidMove
        next_board = board
        next_board[r][c] = 1

        if check_box(next_board)[0]:
            if verbose:
                print(ANSI_string("greedy","green",None,True))
            next_board[r][c] = 0
            return r,c
        else:
            next_board[r][c] = 0
    
    while ValidMoves:
        r,c = random.choice(ValidMoves)
        next_board[r][c] = 1
        if check_suicide(next_board):
            ValidMoves.remove((r, c))
            next_board[r][c] = 0
        else:
            if verbose:
                print("not bad move")
            next_board[r][c] = 0
            return r,c
    if verbose:
        print(ANSI_string("bad move","red",None,True))
    return None

class Greedy_Bot():
    def __init__(self, m, n, verbose = False):
        self.verbose = verbose
        self.board_rows = 2*m - 1
        self.board_cols = 2*n - 1

    def get_move(self, board, player):
        ValidMoves = getValidMoves(board)
        greedy_move = GreedAlg(board,ValidMoves)
        if not greedy_move:
            greedy_move = random.choice(getValidMoves(board))

        r,c = greedy_move

        one_d_len = self.board_rows * self.board_cols

        position = r*self.board_cols+c
        tmp=np.zeros(one_d_len)
        tmp[position] = 1.0
        copy_board = copy.deepcopy(board)

        return greedy_move, [copy_board, tmp, player]

class Greedy_Bot_2():
    def __init__(self, m, n, verbose = False):
        self.verbose = verbose
        self.board_rows = 2*m - 1
        self.board_cols = 2*n - 1

    def get_vertical_horizontal_validmoves(self, board):
        vertical_moves = []
        horizontal_moves = []
        for j in range(self.board_rows):
            for i in range(self.board_cols):
                if board[i][j] != 0:
                    continue
                if i % 2 == 0:
                    horizontal_moves.append((i, j))
                else:
                    vertical_moves.append((i, j))
        return vertical_moves + horizontal_moves

    def Greedy(self,board, ValidMoves, verbose = False):
        for ValidMove in ValidMoves:
            r,c = ValidMove
            next_board = board
            next_board[r][c] = 1

            if check_box(next_board)[0]:
                if verbose:
                    print(ANSI_string("greedy","green",None,True))
                next_board[r][c] = 0
                return r,c
            else:
                next_board[r][c] = 0


        for r,c in ValidMoves:
            next_board[r][c] = 1
            if check_suicide(next_board):
                ValidMoves.remove((r, c))
                next_board[r][c] = 0
            else:
                if verbose:
                    print("not bad move")
                next_board[r][c] = 0
                return r,c

        if verbose:
            print(ANSI_string("bad move","red",None,True))
        return None

    def get_move(self, board, player):
        ValidMoves = self.get_vertical_horizontal_validmoves(board)
        greedy_move = self.Greedy(board, ValidMoves)
        if not greedy_move:
            greedy_move = random.choice(getValidMoves(board))

        r,c = greedy_move
        print(find_chain(r,c, False, self.board_rows, self.board_cols))

        one_d_len = self.board_rows * self.board_cols
        position = r*self.board_cols+c

        tmp=np.zeros(one_d_len)
        tmp[position] = 1.0
        copy_board = copy.deepcopy(board)

        return greedy_move, [copy_board, tmp, player]

class Random_Bot():
    def __init__(self, m, n, verbose = False):
        self.verbose = verbose
        self.board_rows = 2*m - 1
        self.board_cols = 2*n - 1

    def get_move(self, board, player, verbose = True):
        ValidMoves = getValidMoves(board)
        result = random.choice(ValidMoves)

        one_d_len = self.board_rows * self.board_cols
        r,c = result
        position = r*self.board_cols+c
        tmp=np.zeros(one_d_len)
        tmp[position] = 1.0
        copy_board = copy.deepcopy(board)
        if verbose:
            print(f"Radom: {result}")
        return result, [copy_board, tmp, player]

