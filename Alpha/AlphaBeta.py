import math
import copy
import random
from Dots_and_Box import *
from RandomBot import *
import time

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

class AlphaBetaPlayer:
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

        non_think_moves = ((self.state.m-1)*(self.state.n) + (self.state.m)*(self.state.n-1))//2 + 5

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
                if self.symbol != new_state.current_player: #有輪流
                    move_value = self.alphabeta(new_state, 0, -math.inf, math.inf, False)
                else:   #還是自己
                    move_value = self.alphabeta(new_state, 0, -math.inf, math.inf, True)
                end_time = time.time()
                print(f"Py AlphaBeta move {move} took {end_time - start_time:.6f} seconds")

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
        print(f"py best move: {best_move}, best value: {best_value}")
        return best_move, [copyboard, tmp, self.state.current_player]

    def evaluate(self, state:STATE):
        """ 評估函數：根據當前棋盤返回數值評估 """
        if GetWinner(state.board, state.p1_p2_scores) == self.symbol:
            return 10000
        elif GetWinner(state.board, state.p1_p2_scores) == -self.symbol:
            return -10000
        else:
            score_diff = state.p1_p2_scores[0] - state.p1_p2_scores[1]
            if self.symbol == 1:
                score_diff*=-1
            
            # 增加先手玩家的分數
            if self.symbol == -1:  # 先手玩家
                score_diff += 5  # 可根据情况调整加分

            return score_diff*100
        
        return 0  # 平局或未結束時，評分為 0

    def alphabeta(self, state, depth, alpha, beta, maximizing):
        # 終止條件
        if depth >= self.max_depth:
            # print(f"max depth:{depth}")
            return self.evaluate(state)  # 返回棋盤評估值

        winner = GetWinner(state.board, state.p1_p2_scores)
        if winner is not None:
            return self.evaluate(state)  # 游戏结束，直接返回评估值


        if maximizing:
            max_eval = -math.inf
            valid_moves = getValidMoves(state.board)
            for move in valid_moves:
                new_state = copy.deepcopy(state)
                last_player = new_state.current_player
                r, c = move
                new_state.current_player, score = make_move(new_state.board, r, c, new_state.current_player)
                if new_state.current_player == -1:
                    new_state.p1_p2_scores[0] += score
                elif new_state.current_player == 1:
                    new_state.p1_p2_scores[1] += score

                if last_player != new_state.current_player: #有輪流
                    eval = self.alphabeta(new_state, depth + 1, alpha, beta, False)
                else:   #還是自己
                    eval = self.alphabeta(new_state, depth + 1, alpha, beta, True)

                max_eval = max(max_eval, eval)
                alpha = max(alpha, eval)

                if beta <= alpha:
                    # print(f"Beta剪枝 beta:{beta}, alpha:{alpha}, depth: {depth}")
                    break  # Beta 剪枝
            return max_eval
        else:
            min_eval = math.inf
            valid_moves = getValidMoves(state.board)
            for move in valid_moves:
                new_state = copy.deepcopy(state)
                last_player = new_state.current_player
                r, c = move
                new_state.current_player, score = make_move(new_state.board, r, c, new_state.current_player)
                if new_state.current_player == -1:
                    new_state.p1_p2_scores[0] += score
                elif new_state.current_player == 1:
                    new_state.p1_p2_scores[1] += score
                if last_player != new_state.current_player: #有輪流
                    eval = self.alphabeta(new_state, depth + 1, alpha, beta, True)
                else:   #還是自己
                    eval = self.alphabeta(new_state, depth + 1, alpha, beta, False)

                min_eval = min(min_eval, eval)
                beta = min(beta, eval)
                if beta <= alpha:
                    # print(f"Alpha剪枝 beta:{beta}, alpha:{alpha}, depth: {depth}")
                    break  # Alpha 剪枝
            return min_eval