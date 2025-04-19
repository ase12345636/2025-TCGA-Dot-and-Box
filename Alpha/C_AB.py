import math
import copy
import random
from Dots_and_Box import *
from RandomBot import *
import time
from Alpha.game_ai import PyState, py_alphabeta

"""
struct state {
    int p1_p2_scores[2];   // [0]=>player -1, [1]=>player 1
    vector<vector<int>> board;
    int m;
    int n;
    int board_rows;
    int board_cols;
    int current_player;
}
"""

class C_AB_player:
    def __init__(self, symbol, state: STATE, max_depth=10,method_bot = None):
        self.symbol = symbol
        self.state = state
        self.max_depth = max_depth
        self.select_meth_bot = Greedy_Bot_2(self.state.m, self.state.n, verbose=True)
        if method_bot:
            self.select_meth_bot = method_bot
    def get_move(self, board, player):
        best_value = -math.inf
        best_move = None

        visited_pos = []
        valid_moves = getValidMoves(board)

        non_think_moves = ((self.state.m-1)*(self.state.n) + (self.state.m)*(self.state.n-1))//2

        for _ in range(len(valid_moves)):
            move = self.select_meth_bot.get_move(board, player)[0]
            one_d_len = self.state.board_rows * self.state.board_cols

            if (len(valid_moves)>non_think_moves) and move:   #遊戲前期無需太多思考，直接return

                r,c = move
                position = r*self.state.board_cols+c
                tmp=np.zeros(one_d_len)
                tmp[position] = 1.0
                copyboard = copy.deepcopy(self.state.board)

                return move, [copyboard, tmp, self.state.current_player]

            attempt = 0
            while move in visited_pos and attempt<12:
                move = random.choice(valid_moves)
                attempt+=1

            visited_pos.append(move)
            if move:
                new_state = copy.deepcopy(self.state)
                r, c = move
                new_state.current_player, score = make_move(new_state.board, r, c, new_state.current_player)
                if new_state.current_player == -1:
                    new_state.p1_p2_scores[0] += score
                elif new_state.current_player == 1:
                    new_state.p1_p2_scores[1] += score

                start_time = time.time()
                C_new_state = PyState()
                C_new_state.set_dimensions(new_state.m, new_state.n, new_state.board_rows, new_state.board_cols)  # m, n, board_rows, board_cols
                C_new_state.set_board(new_state.board)
                C_new_state.set_scores(new_state.p1_p2_scores)
                C_new_state.set_current_player(new_state.current_player)

                if self.symbol != new_state.current_player: #有輪流
                    move_value = py_alphabeta(C_new_state, 0, self.max_depth, -999999, 999999, self.symbol, False)
                else:   #還是自己
                    move_value = py_alphabeta(C_new_state, 0, self.max_depth, -999999, 999999, self.symbol, True)

                end_time = time.time()
                print(f"C AlphaBeta move {move} took {end_time - start_time:.6f} seconds")

                if move_value > best_value:
                    best_value = move_value
                    best_move = move
        if not best_move:
            best_move =  random.choice(getValidMoves(board))
            print(f"random best_move {best_move} ")

        r,c = best_move
        position = r*self.state.board_cols+c
        tmp=np.zeros(one_d_len)
        tmp[position] = 1.0
        copyboard = copy.deepcopy(self.state.board)
        print(f"C best move: {best_move}, best value: {best_value}")
        return best_move, [copyboard, tmp, self.state.current_player]
