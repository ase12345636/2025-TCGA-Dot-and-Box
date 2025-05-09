import math
import random
import time
import numpy as np
from copy import deepcopy
from Dots_and_Box import *
from RandomBot import *
from DeepLearning import ResnetBOT
from arg import *


args_Res['train'] = False
print(args_Res)


# MCTS 節點類別，代表蒙地卡羅樹搜索中的每個節點
class MCTSNode:
    # parent進行move達到self, player為誰進行了這項移動
    def __init__(self, game_state: STATE, parent=None, player=None, move=None, prior=0.0):
        self.game_state = deepcopy(game_state)  # 節點的遊戲狀態
        self.parent = parent  # 父節點
        self.move = move  # 對應的移動
        self.children = []  # 子節點列表
        self.visits = 0  # 訪問次數, N(s, a)
        self.score = 0  # 總得分
        self.Wsa = 0
        self.player = player    # player下了move走到此節點

        self.win_count = 0  # 走此節點會獲勝的次數

        self.prior = prior  # 新增：P(s, a)，由 policyNet 給出
        self.untried_moves = getValidMoves(self.game_state.board)
        random.shuffle(self.untried_moves)  # 隨機化未嘗試的移動順序


# MCTS 玩家類別，使用蒙地卡羅樹搜索進行決策
class AlphaGoMCTSPlayer:
    def __init__(self, num_simulations, game: DotsAndBox, symbol, exploration_weight=2):
        self.num_simulations = num_simulations  # 蒙地卡羅模擬次數
        self.exploration_weight = exploration_weight  # 探索權重
        self.root_state = game.state  # 遊戲狀態
        self.symbol = symbol
        self.select_meth_bot = ResnetBOT(
            input_size_m=m, input_size_n=n, game=game, args=args_Res)
        self.move_num = 0

    # 獲取下一步移動

    def get_move(self, board, player, verbose=True):
        current_time = time.time()

        total_moves = (self.root_state.m - 1) * self.root_state.n + \
            self.root_state.m * (self.root_state.n - 1)
        remaining_moves = len(getValidMoves(self.root_state.board))
        progress = 1 - remaining_moves / total_moves

        # 更新當前遊戲狀態
        self.root_state.board = board
        self.root_state.current_player = player

        # if progress < 0.45:  # 開局
        #     move = self.select_meth_bot.get_move(board, player)[0]
        #     return move, []
        # elif progress < 0.6:
        #     self.num_simulations = 300
        # elif progress < 0.7:
        #     self.num_simulations = 350
        # elif progress < 0.8:
        #     self.num_simulations = 350
        # elif progress < 0.9:
        #     self.num_simulations = 400
        # else:  # 終盤
        #     self.num_simulations = 500

        if verbose:
            print(f"Game progress: {progress}")
            print(f"Max num_simulations: {self.num_simulations}")

        if not self.root_state:
            raise ValueError("Game state not set")  # 若遊戲狀態未設置，則拋出錯誤
        root = MCTSNode(self.root_state)  # 根節點為當前遊戲狀態的複製

        # 進行多次模擬
        for _ in range(self.num_simulations):
            node = self.select(root)  # 選擇節點
            # 選擇節點直到不能選為止
            if not isGameOver(node.game_state.board) and node.untried_moves:
                node = self.expand(node)

            simulation_result = self.evaluate(deepcopy(node.game_state))
            self.backpropagate(node, simulation_result)

        # 如果根節點沒有子節點，則隨機選擇一個有效移動
        if not root.children:
            print("random choose")
            return self.select_meth_bot.get_move(board, player)

        # 選擇訪問次數最多的子節點
        def get_temperature(moves_num):
            if moves_num <= 10:
                return 1

            else:
                temperature_min = 1e-100
                temperature = 0.95 ** (moves_num - 10)
                return temperature if temperature > temperature_min else temperature_min

        self.move_num += 1
        n_with_temperature = np.array([
            i.visits for i in root.children], dtype=float)**(1 / get_temperature(self.move_num))
        sum_n_with_temperature = np.sum(n_with_temperature)
        best_move = deepcopy(root.children[np.argmax(
            n_with_temperature/sum_n_with_temperature)].move)
        # best_child = max(root.children, key=lambda c: c.score /
        #                  c.visits if c.visits > 0 else -float('inf'))
        # best_child = sorted(root.children, key = lambda c: float(c.score)/c.visits)[-1]
        # best_child = max(root.children, key=lambda c: c.score)

        end_time = time.time()

        if verbose:
            print(
                f"AlphaGo best move: {best_move}, take {end_time - current_time:.6f}s")

        self.del_tree(root)
        del root

        return best_move, []

    # 節點選擇過程

    def select(self, node: MCTSNode):
        while node.untried_moves == [] and node.children != []:
            node = self.puct_select(node)

        return node

    # 擴展節點

    def expand(self, node: MCTSNode):
        random.shuffle(node.untried_moves)
        # 從此節點狀態中可下且未被展開的move進行選擇
        move = node.untried_moves.pop()
        new_state = deepcopy(node.game_state)
        r, c = move

        # 預測當前狀態所有可行步的機率
        if self.select_meth_bot is not None:
            policy = self.select_meth_bot.PredictPolicy(
                new_state.board)    # 用policyNet得到每個位置的機率分布
            move_idx = r*new_state.board_cols + c
            new_prior = policy[move_idx]    # 把此步的機率標記進該節點中

        else:
            # 沒有 policy_net，則平均分配
            total_untried = len(node.untried_moves) + 1  # +1 是包括這個 move
            new_prior = 1.0 / total_untried

        # 執行移動
        new_node_player = new_state.current_player
        new_state.current_player, score = make_move(
            new_state.board, r, c, new_state.current_player)
        if new_state.current_player == -1:
            new_state.p1_p2_scores[0] += score
        elif new_state.current_player == 1:
            new_state.p1_p2_scores[1] += score
        new_state.history_8board.append(new_state.board)

        # 建立新節點
        new_node = MCTSNode(new_state, parent=node,
                            move=move, player=new_node_player)
        new_node.prior = new_prior
        node.children.append(new_node)

        return new_node

    # 評估遊戲狀態

    def evaluate(self, state: STATE):
        value = self.select_meth_bot.PredictValue(
            state.board, state.current_player)
        return value

    # 回傳模擬結果

    def backpropagate(self, node: MCTSNode, result):
        while node:
            node.visits += 1
            if node.parent:  # 非根節點
                node.score += result

            else:  # 根節點
                node.score += result

            node = node.parent

    def get_puct_score(self, node: MCTSNode):
        if node.parent is None:
            return float('inf')

         # Q(s,a): exploitation term
        if node.visits == 0:
            Q = 0  # 還沒被拜訪，Q值暫時設為0（或你也可以用其他初始化方式）
        else:
            # Q = node.Wsa / node.visits
            Q = node.score / node.visits * node.player

        # U(s,a): exploration term
        N = node.parent.visits  # 父節點的拜訪次數
        c_puct = self.exploration_weight
        U = c_puct * node.prior * math.sqrt(N) / (1 + node.visits)
        return Q+U

    def puct_select(self, node: MCTSNode):
        return max(node.children, key=lambda c: self.get_puct_score(c))

    def del_tree(self, current_node):
        for child in current_node.children:
            if child.children == []:
                del child

            else:
                self.del_tree(child)
