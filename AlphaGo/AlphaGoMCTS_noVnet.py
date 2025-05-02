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

        total_moves = (self.game_state.m - 1) * self.game_state.n + \
            self.game_state.m * (self.game_state.n - 1)
        remaining_moves = len(getValidMoves(self.game_state.board))
        self.progress = 1 - remaining_moves / total_moves


# MCTS 玩家類別，使用蒙地卡羅樹搜索進行決策
class AlphaGoMCTSPlayer_NoVNet:
    def __init__(self, num_simulations, game: DotsAndBox, symbol, exploration_weight=2, max_depth=20):
        self.num_simulations = num_simulations  # 蒙地卡羅模擬次數
        self.exploration_weight = exploration_weight  # 探索權重
        self.root_state = game.state  # 遊戲狀態
        self.symbol = symbol
        self.select_meth_bot = ResnetBOT(
            input_size_m=m, input_size_n=n, game=game, args=args_Res)
        self.move_num = 0
        self.max_depth = max_depth

    # 獲取下一步移動
    def get_move(self, board, player, verbose=True):
        current_time = time.time()

        # 更新當前遊戲狀態
        self.root_state.board = board
        self.root_state.current_player = player

        if not self.root_state:
            raise ValueError("Game state not set")  # 若遊戲狀態未設置，則拋出錯誤
        root = MCTSNode(self.root_state)  # 根節點為當前遊戲狀態的複製

        simulations=self.num_simulations
        if root.progress < 0.45:  # 開局
            simulations *= 1
        elif root.progress < 0.6:
            simulations *= 0.9
        elif root.progress < 0.7:
            simulations *= 0.8
        elif root.progress < 0.8:
            simulations *= 0.7
        elif root.progress < 0.9:
            simulations *= 0.6
        else:  # 終盤
            simulations *= 0.5

        # 進行多次模擬
        for _ in range(int(simulations)):
            # 選擇節點
            node = self.select(root)

            # 選擇節點直到不能選為止
            if not isGameOver(node.game_state.board) and node.untried_moves:
                node = self.expand(node)

            # simulation_result = self.evaluate(deepcopy(node.game_state))
            simulation_result = self.evaluate(node)
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
                f"AlphaGo takes {end_time - current_time:.6f}s")

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
        # 隨機挑選可行步
        move = node.untried_moves.pop()
        new_state = deepcopy(node.game_state)
        r, c = move

        # 得到此可行步的機率
        policy = self.select_meth_bot.PredictPolicy(
            new_state.board)
        move_idx = r*new_state.board_cols + c
        new_prior = policy[move_idx]

        # 執行移動
        new_state.current_player, score = make_move(
            new_state.board, r, c, new_state.current_player)
        new_node_player = new_state.current_player
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
    def evaluate(self, node: MCTSNode):
        # If node is root or node's player is different to node's parent's player, set false
        play_next = -1 if not node.parent or node.player != node.parent.player else 1

        # Score Difference
        _, own_score = checkBox(node.game_state.board, node.player)
        _, oppo_score = checkBox(node.game_state.board, node.player*-1)
        difference_score = own_score-oppo_score

        # Box chain
        next_chain = play_next*checkBoxChain(node.game_state.board)

        # number of box
        total_boxes = (node.game_state.m-1)*(node.game_state.n-1)

        total_score = difference_score+next_chain*100/(total_boxes*100)

        return total_score if node.player == self.symbol else total_score*-1

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
            self.del_tree(child)

        current_node.children.clear()
