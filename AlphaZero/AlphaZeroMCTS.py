import math
import random
from copy import deepcopy
from Dots_and_Box import *
from RandomBot import *
from DeepLearning import ResnetBOT
from arg import *

# MCTS 節點類別，代表蒙地卡羅樹搜索中的每個節點
class MCTSNode:
    def __init__(self, game_state:STATE, parent=None, player=None, move=None, prior=0.0):   # parent進行move達到self, player為誰進行了這項移動
        self.game_state = deepcopy(game_state)  # 節點的遊戲狀態
        self.parent = parent  # 父節點
        self.move = move  # 對應的移動
        self.children = []  # 子節點列表
        self.visits = 0  # 訪問次數, N(s, a)
        self.score = 0  # 總得分
        self.Wsa = 0.0  # W(s, a)
        self.player = player

        self.prior = prior  # 新增：P(s, a)，由 policyNet 給出
        self.untried_moves = getValidMoves(self.game_state.board)
        random.shuffle(self.untried_moves)  # 隨機化未嘗試的移動順序

# MCTS 玩家類別，使用蒙地卡羅樹搜索進行決策
class AlphaZeroMCTSPlayer:
    def __init__(self, num_simulations, game:DotsAndBox, symbol, exploration_weight=2, max_depth=1000):
        self.num_simulations = num_simulations  # 蒙地卡羅模擬次數
        self.exploration_weight = exploration_weight  # 探索權重
        self.max_depth = max_depth  # 模擬的最大深度
        self.root_state = game.state  # 遊戲狀態
        self.symbol = symbol
        # self.select_meth_bot = Greedy_Bot_2(state.m, state.n)
        self.policyNet = ResnetBOT(game.state.m, game.state.n, game,args_Res)

    # 獲取下一步移動
    def get_move(self, board, player):
        total_moves = (self.root_state.m - 1) * self.root_state.n + self.root_state.m * (self.root_state.n - 1)
        remaining_moves = len(getValidMoves(self.root_state.board))
        progress = 1 - remaining_moves / total_moves

        if progress < 0.44:  # 開局
            move = self.policyNet.get_move(board, player)[0]
            return move, []
        elif progress < 0.7:  # 中盤
            self.num_simulations = 17000
            self.max_depth = 20
        else:  # 終盤
            self.num_simulations = 25000
            self.max_depth = 50

        if not self.root_state:
            raise ValueError("Game state not set")  # 若遊戲狀態未設置，則拋出錯誤

        # root = MCTSNode(self.root_state)  # 根節點為當前遊戲狀態的複製
         # 更新當前遊戲狀態
        self.root_state.board = board
        root = MCTSNode(self.root_state, parent=None, player=self.root_state.current_player, move=None, prior=0.0)
        self.root_state.current_player = player
        # 進行多次模擬
        for _ in range(self.num_simulations):
            node = root

            # 選擇節點直到不能選為止
            while node.children and not isGameOver(node.game_state.board):
                node = self.select(node)

            # 擴展節點（如果還有未嘗試的移動）
            if node.untried_moves and not isGameOver(node.game_state.board):
                node = self.expand(node)

            simulation_result = self.simulate(node.game_state)
            self.backpropagate(node, simulation_result)


        # 如果根節點沒有子節點，則隨機選擇一個有效移動
        if not root.children:
            print("random choose")
            return self.policyNet.get_move(board, player)

        # 選擇訪問次數最多的子節點
        best_child = max(root.children, key=lambda c: c.score / c.visits if c.visits > 0 else -float('inf'))
        # best_child = sorted(root.children, key = lambda c: float(c.score)/c.visits)[-1]
        # best_child = max(root.children, key=lambda c: c.score)
        print(f"MCTS best move: {best_child.move}, score: {best_child.score}, visits: {best_child.visits}")
        return best_child.move, []

    # 節點選擇過程
    def select(self, node):
        return max(node.children, key=lambda c: self.puct_select(c))

    # 擴展節點
    def expand(self, node: MCTSNode):
        move = node.untried_moves.pop()
        new_state = deepcopy(node.game_state)
        r, c = move
        new_prior = 0.0
        # 預測當前狀態所有可行步的機率
        if self.policyNet is not None:
            policy = self.policyNet.model.predict(
                np.expand_dims(new_state.board, axis=0).astype(float))
            # Detect which move is valid
            valid_positions = getValidMoves(new_state.board)
            valids = np.zeros(
                (new_state.board_rows * new_state.board_cols,), dtype='int')
            for pos in valid_positions:
                idx = pos[0] * new_state.board_cols + pos[1]
                valids[idx] = 1

            # Filtered invalid move and avoid invalid loop
            policy = (policy+1e-30) * valids
            move_index = r*new_state.board_cols + c
            new_prior = policy[move_index]
        else:
            # 沒有 policy_net，則平均分配
            total_untried = len(node.untried_moves) + 1  # +1 是包括這個 move
            new_prior = 1.0 / total_untried

        # 執行移動
        new_node_player = new_state.current_player
        new_state.current_player, score = make_move(new_state.board, r, c, new_state.current_player)
        if new_state.current_player == -1:
            new_state.p1_p2_scores[0] += score
        elif new_state.current_player == 1:
            new_state.p1_p2_scores[1] += score

        # 建立新節點
        new_node = MCTSNode(new_state, parent=node, move=move, player=new_node_player)
        new_node.prior = new_prior


        node.children.append(new_node)
        return new_node

    # 模擬遊戲進行
    def simulate(self, game_state:STATE):
        state = deepcopy(game_state)  # 複製遊戲狀態
        depth = 0

        # 以state為遊戲狀態往下模擬，以self.policyNet進行雙方對弈模擬
        while not isGameOver(state.board):  # 直到遊戲結束或達到最大深度
            move = self.choose_simulation_move(state)  # 選擇模擬中的移動
            if move is None:
                break
            # 執行該移動
            r, c = move
            state.current_player, score = make_move(state.board,r,c,state.current_player)
            if state.current_player == -1:
                state.p1_p2_scores[0] += score
            elif state.current_player == 1:
                state.p1_p2_scores[1] += score

        return self.evaluate(state)  # 對遊戲結束或是達終止條件的state進行評分

    # 選擇模擬中的移動
    def choose_simulation_move(self, game_state:STATE):
        rand = random.random()
        if rand < 0.8:
            return self.select_meth_bot.get_move(game_state.board,game_state.current_player)[0]
        elif rand < 0.9:
            return Greedy_Bot(game_state.m, game_state.n).get_move(game_state.board,game_state.current_player)[0]
        else:
            return Random_Bot(game_state.m, game_state.n).get_move(game_state.board,game_state.current_player,verbose=False)[0]

    # 評估遊戲狀態
    def evaluate(self, state:STATE):
        # if isGameOver(state.board):
        #     winner = GetWinner(state.board, state.p1_p2_scores)
        #     return 1.0 if winner == self.symbol else -1.0 if winner != 0 else 0.0
        # else:
        total_boxes = (state.m-1)*(state.n-1)
        my_score = state.p1_p2_scores[0] if self.symbol == -1 else state.p1_p2_scores[1]
        opp_score = state.p1_p2_scores[1] if self.symbol == -1 else state.p1_p2_scores[0]
        return (my_score - opp_score) / total_boxes  # 动态归一化


    # 回傳模擬結果
    def backpropagate(self, node:MCTSNode, result):
        while node:
            node.visits += 1
            # 正確處理玩家視角
            if node.parent:  # 非根節點
                if node.player == self.symbol:
                    node.score += result
                    node.Wsa += result
                else:
                    node.score -= result
                    node.Wsa += result
            else:  # 根節點
                node.score += result
                node.Wsa += result

            node = node.parent


    def puct_select(self, node:MCTSNode):
        if node.parent is None:
            return float('inf')

         # Q(s,a): exploitation term
        if node.visits == 0:
            Q = 0  # 還沒被拜訪，Q值暫時設為0（或你也可以用其他初始化方式）
        else:
            Q = node.Wsa / node.visits

        # U(s,a): exploration term
        N = node.parent.visits  # 父節點的拜訪次數
        c_puct = self.exploration_weight
        U = c_puct * node.prior * math.sqrt(N) / (1 + node.visits)
        return Q+U