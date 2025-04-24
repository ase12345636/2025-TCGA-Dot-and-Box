import math
import random
from copy import deepcopy
from Dots_and_Box import *
from RandomBot import *
import time

# MCTS 節點類別，代表蒙地卡羅樹搜索中的每個節點
class MCTSNode:
    def __init__(self, game_state:STATE, parent=None, player=None, move=None):   # parent進行move達到self, player為誰進行了這項移動
        self.game_state = deepcopy(game_state)  # 節點的遊戲狀態
        self.parent = parent  # 父節點
        self.move = move  # 對應的移動
        self.children = []  # 子節點列表
        self.visits = 0  # 訪問次數
        self.score = 0  # 總得分
        self.player = player

        self.untried_moves = getValidMoves(game_state.board)
        random.shuffle(self.untried_moves)  # 隨機化未嘗試的移動順序

# MCTS 玩家類別，使用蒙地卡羅樹搜索進行決策
class MCTSPlayer:
    def __init__(self, num_simulations, state:STATE, symbol, exploration_weight=2, max_depth=1000):
        self.num_simulations = num_simulations  # 蒙地卡羅模擬次數
        self.exploration_weight = exploration_weight  # 探索權重
        self.max_depth = max_depth  # 模擬的最大深度
        self.root_state = state  # 遊戲狀態
        self.symbol = symbol
        self.select_meth_bot = Greedy_Bot_2(state.m, state.n)

    # 獲取下一步移動
    def get_move(self, board, player, verbose = True):
        current_time = time.time()

        total_moves = (self.root_state.m - 1) * self.root_state.n + self.root_state.m * (self.root_state.n - 1)
        remaining_moves = len(getValidMoves(self.root_state.board))
        progress = 1 - remaining_moves / total_moves

        # 更新當前遊戲狀態
        self.root_state.board = board
        self.root_state.current_player = player

        if progress < 0.45:  # 開局
            move = self.select_meth_bot.get_move(board, player)[0]
            return move, []
        elif progress < 0.6:
            self.num_simulations = 15000
        elif progress < 0.7:
            self.num_simulations = 16000
        elif progress < 0.8:
            self.num_simulations = 17000
        elif progress < 0.9:
            self.num_simulations = 18000
        else:  # 終盤
            self.num_simulations = 19000

        if verbose:
            print(f"Game progress: {progress}")
            print(f"Max num_simulations: {self.num_simulations}")

        if not self.root_state:
            raise ValueError("Game state not set")  # 若遊戲狀態未設置，則拋出錯誤

        root = MCTSNode(self.root_state)  # 根節點為當前遊戲狀態的複製
        # 進行多次模擬
        for _ in range(self.num_simulations):
            # print(_)
            node = self.select(root)  # 選擇節點

            # 從node往下擴展
            if not isGameOver(node.game_state.board) and node.untried_moves:
                node = self.expand(node)    # node變成擴展後的子節點

            # 對node進行simulate直到終止條件
            simulation_result = self.simulate(node.game_state)  # 模擬遊戲結果，得到node的評分

            self.backpropagate(node, simulation_result)  # 回傳模擬結果並一直加回到root為止

        # 如果根節點沒有子節點，則隨機選擇一個有效移動
        if not root.children:
            print("random choose")
            return self.select_meth_bot.get_move(board, player)

        # 選擇訪問次數最多的子節點
        best_child = max(root.children, key=lambda c: c.score / c.visits if c.visits > 0 else -float('inf'))
        # best_child = sorted(root.children, key = lambda c: float(c.score)/c.visits)[-1]
        # best_child = max(root.children, key=lambda c: c.score)

        end_time = time.time()

        if verbose:
            print(f"MCTS best move: {best_child.move}, score: {best_child.score}, visits: {best_child.visits}, took {end_time - current_time:.6f} seconds")
            for c in root.children:
                print(f"{c.score}/{c.visits}")
        return best_child.move, []

    # 節點選擇過程
    def select(self, node):
        while node.untried_moves == [] and node.children != []:
            node = self.uct_select(node)
        return node

    # 擴展節點
    def expand(self, node:MCTSNode):
        random.shuffle(node.untried_moves)
        # 從此節點狀態中可下且未被展開的move進行選擇
        move = node.untried_moves.pop()
        new_state = deepcopy(node.game_state)  # 複製當前遊戲狀態
        r, c = move

        # 執行該移動 (expand)
        new_node_player = new_state.current_player
        new_state.current_player, score = make_move(new_state.board,r,c,new_state.current_player)
        if new_state.current_player == -1:
            new_state.p1_p2_scores[0] += score
        elif new_state.current_player == 1:
            new_state.p1_p2_scores[1] += score
        new_state.history_8board.append(new_state.board)

        new_node = MCTSNode(new_state, parent=node, move=move, player=new_node_player)  # 創建新節點
        node.children.append(new_node)  # 將新節點添加為子節點
        return new_node # 回傳擴展後的子節點

    # 模擬遊戲進行
    def simulate(self, game_state:STATE):
        state = deepcopy(game_state)  # 複製遊戲狀態
        depth = 0
        # 以state為遊戲狀態往下模擬，以self.select_meth_bot進行雙方對弈模擬
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
            state.history_8board.append(state.board)
        return self.evaluate(state)  # 對遊戲結束或是達終止條件的state進行評分

    # 選擇模擬中的移動
    def choose_simulation_move(self, game_state:STATE):
        rand = random.random()
        if rand < 0.6:
            return self.select_meth_bot.get_move(game_state.board,game_state.current_player)[0]
        elif rand < 0.8:
            return Greedy_Bot(game_state.m, game_state.n).get_move(game_state.board,game_state.current_player)[0]
        else:
            return Random_Bot(game_state.m, game_state.n).get_move(game_state.board,game_state.current_player,verbose=False)[0]

    def evaluate(self, state:STATE):
        total_boxes = (state.m-1)*(state.n-1)
        my_score = state.p1_p2_scores[0] if self.symbol == -1 else state.p1_p2_scores[1]
        opp_score = state.p1_p2_scores[1] if self.symbol == -1 else state.p1_p2_scores[0]
        return (my_score - opp_score) / total_boxes        # if isGameOver(state.board):
        #     winner = GetWinner(state.board, state.p1_p2_scores)
        #     if winner == self.symbol:
        #         return 1
        #     elif winner == -self.symbol:
        #         return -1
        #     else:
        #         return 0  # 平手
        # return (my_score - opp_score) / total_boxes


    # 回傳模擬結果
    def backpropagate(self, node:MCTSNode, result):
        while node:
            node.visits += 1
            if node.parent:  # 非根節點
                if node.player == self.symbol:
                    node.score += result
                else:
                    node.score -= result
            else:  # 根節點
                node.score += result

            node = node.parent



    # 使用 UCT (Upper Confidence Bound) 選擇節點
    def uct_select(self, node:MCTSNode):
        if node.parent is None:
            log_parent_visits = math.log(node.visits + 1e-6)
        else:
            log_parent_visits = math.log(node.parent.visits + 1e-6)
        return max(node.children, key=lambda c: self.ucb1(c, log_parent_visits))

    # UCB1 函數計算
    def ucb1(self, node:MCTSNode, log_parent_visits):
        if node.visits == 0:
            return float('inf')
        exploitation = node.score / node.visits
        exploration = math.sqrt(2 * log_parent_visits / node.visits)
        return exploitation + self.exploration_weight * exploration