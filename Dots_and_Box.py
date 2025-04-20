# -1 or 1 : 先手(1)or後手(-1)走的邊
# 5 : 頂點
# 8 : 未被佔領的格子
# 7 or 9 : 被先手or後手佔領的格子，7->8+(-1)，9->8+(1)
# p1 -> -1(blue)，p2 -> 1(red)
from DaB_Utils import *
from dataclasses import dataclass, field
from typing import List
def ANSI_string(s="", color=None, background=None, bold=False):
    colors = {
        'black': '\033[30m',
        'red': '\033[31m',
        'green': '\033[32m',
        'yellow': '\033[33m',
        'blue': '\033[34m',
        'magenta': '\033[35m',
        'cyan': '\033[36m',
        'white': '\033[37m',
        'reset': '\033[0m'
    }

    background_colors = {
        'black': '\033[40m',
        'red': '\033[41m',
        'green': '\033[42m',
        'yellow': '\033[43m',
        'blue': '\033[44m',
        'magenta': '\033[45m',
        'cyan': '\033[46m',
        'white': '\033[47m',
        'reset': '\033[0m',
        'gray': '\033[100m',  # 新增的灰色背景
        'light_gray': '\033[47m'  # 新增的淺灰色背景
    }

    styles = {
        'bold': '\033[1m',
        'reset': '\033[0m'
    }
    color_code = colors[color] if color in colors else ''
    background_code = background_colors[background] if background in colors else ''
    bold_code = styles['bold'] if bold else ''

    return f"{color_code}{background_code}{bold_code}{s}{styles['reset']}"


@dataclass
class STATE:
    p1_p2_scores: list
    board: list
    m: int
    n: int
    board_rows: int = field(init=False)
    board_cols: int = field(init=False)
    current_player: int = -1

    def __post_init__(self):
        self.board_rows = 2 * self.m - 1
        self.board_cols = 2 * self.n - 1

class DotsAndBox():
    def initialize_board(self, m, n):
        # 初始化一個(2*m-1) * (2*n-1)的二維陣列
        board = [[0 for _ in range(2*n-1)] for _ in range(2*m-1)]
        # 填入頂點(5)和合法邊(0)，以及未被佔領的格子(8)
        for i in range(2*m-1):
            for j in range(2*n-1):
                if i % 2 == 0 and j % 2 == 0:
                    # 填入頂點 5
                    board[i][j] = 5
                elif i % 2 == 1 and j % 2 == 1:
                    # 填入格子 8
                    board[i][j] = 8
                else:
                    # 填入合法邊 0
                    board[i][j] = 0
        return board

    def __init__(self, state:STATE, collect_gaming_data = True):
        self.state = state
        self.state.board = self.initialize_board(self.state.m, self.state.n)
        self.state.current_player = -1
        self.state.p1_p2_scores = [0, 0]   # [0]=>player -1, [1]=>player 1

        self.collect_gaming_data = collect_gaming_data
        self.history = []

    def play(self, player1, player2, verbose=True, train = False):
        if verbose:
            self.print_board()

        while not isGameOver(self.state.board):
            # print(f"Valid moves: {self.getValidMoves()}")
            print(f"Current player: {self.state.current_player}")

            if self.state.current_player == -1:
                move_data = player1.get_move(self.state.board, self.state.current_player)
            else:
                move_data = player2.get_move(self.state.board, self.state.current_player)

            if move_data[0]:    #valid_postion
                row, col = move_data[0]
                if train:
                    self.history.append(move_data[1])

                # 進行落子並檢查換手
                self.state.current_player,score = make_move(self.state.board, row, col, self.state.current_player)
                if self.state.current_player == -1:
                    self.state.p1_p2_scores[0] += score
                elif self.state.current_player == 1:
                    self.state.p1_p2_scores[1] += score
                if verbose:
                    self.print_board()
        winner = GetWinner(self.state.board, self.state.p1_p2_scores)
        self.print_board()

        if (winner == 0):
            print("Tie!!!")
            return 0
        else:
            print(f"Player {winner} won!!!")
            return winner

    def NewGame(self):
        self.state.board = self.initialize_board(self.state.m, self.state.n)
        self.state.current_player = -1
        self.state.p1_p2_scores = [0, 0]
        self.history.clear()

    def print_board(self):
        print(f"Player -1: {self.state.p1_p2_scores[0]}")
        print(f"Player 1: {self.state.p1_p2_scores[1]}")
        for i in range(self.state.board_rows):
            for j in range(self.state.board_cols):
                value = self.state.board[i][j]
                if value == 0:
                    print(' ', end='')
                elif value in [-1, 1]:  # 處理先手和後手邊
                    color = 'blue' if value == -1 else 'red'
                    if i % 2 == 0:  # 偶數列 -> 水平線
                        print(ANSI_string(s='-', color=color), end='')
                    else:  # 奇數列 -> 垂直線
                        print(ANSI_string(s='|', color=color), end='')
                elif value == 5:  # 頂點
                    print('o', end='')
                elif value in [7, 9]:  # 處理先手和後手佔領
                    background = 'blue' if value == 7 else 'red'
                    print(ANSI_string(s=str(value), background=background), end='')
                elif value == 8:
                    print(' ', end='')
            print()
        print("="*(self.state.board_cols))
