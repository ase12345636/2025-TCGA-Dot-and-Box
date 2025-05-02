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


class MCTSNode:
    def __init__(self, game_state: STATE, parent=None, move=None, prior=0.0, score=0):
        self.game_state = deepcopy(game_state)
        self.parent = parent
        self.move = move
        self.visited = 0
        self.prior = prior
        self.score = score
        self.children = []
        self.legal_move = getValidMoves(self.game_state.board)

        self.expand = False


class AlphaGoMCTSPlayer_Ver3:
    def __init__(self, num_simulations, game: DotsAndBox, symbol, exploration_weight=2):
        self.num_simulations = num_simulations  # 蒙地卡羅模擬次數
        self.exploration_weight = exploration_weight  # 探索權重
        self.root_state = game.state  # 遊戲狀態
        self.symbol = symbol
        self.select_meth_bot = ResnetBOT(
            input_size_m=m, input_size_n=n, game=game, args=args_Res)
        self.move_num = 0

    def get_move(self, board, player, verbose=True):
        def get_temperature(moves_num):
            if moves_num <= 10:
                return 1

            else:
                temperature_min = 1e-100
                temperature = 0.95 ** (moves_num - 10)
                return temperature if temperature > temperature_min else temperature_min

        current_time = time.time()

        # Update root state
        self.root_state.board = deepcopy(board)
        self.root_state.current_player = player
        self.root_state.remaining_moves = len(getValidMoves(self.root_state.board))
        self.root_state.progress = 1 - self.root_state.remaining_moves / self.root_state.total_moves

        root = MCTSNode(self.root_state)  # 根節點為當前遊戲狀態的複製

        simulations=self.num_simulations
        if root.game_state.progress < 0.45:  # 開局
            simulations *= 1
        elif root.game_state.progress < 0.6:
            simulations*= 0.9
        elif root.game_state.progress < 0.7:
            simulations *= 0.8
        elif root.game_state.progress < 0.8:
            simulations *= 0.7
        elif root.game_state.progress < 0.9:
            simulations *= 0.6
        else:  # 終盤
            simulations *= 0.5

        for _ in range(int(simulations)):
            node = self.select(root)
            self.expand_and_evaluate(node)
            self.backup(node)

        self.move_num += 1
        n_with_temperature = np.array([
            i.visited for i in root.children], dtype=float)**(1 / get_temperature(self.move_num))
        sum_n_with_temperature = np.sum(n_with_temperature)
        best_move = deepcopy(root.children[np.argmax(
            n_with_temperature/sum_n_with_temperature)].move)

        end_time = time.time()

        if verbose:
            print(f"AlphaGo takes {end_time - current_time:.6f}s")

        return best_move, []

    def select(self, node: MCTSNode):
        def get_puct_score(node: MCTSNode):
            N_parent = node.visited
            children_puct_score = []
            for child in node.children:
                Q = child.score/(child.visited+(child.visited == 0))
                N = node.visited
                U = self.exploration_weight*child.prior * \
                    math.sqrt(N_parent) / (1 + N)
                children_puct_score.append(
                    Q*node.game_state.current_player*(-1)+U)

            return np.array(children_puct_score, dtype=float)

        def puct_select(node: MCTSNode):
            return node.children[np.argmax(get_puct_score(node))]

        while node.expand and not isGameOver(node.game_state.board):
            node = puct_select(node)

        return node

    def expand_and_evaluate(self, node: MCTSNode):
        def evaluate(node: MCTSNode):
            # If node is root or node's player is different to node's parent's player, set false
            play_next = -1 if not node.parent or \
                node.game_state.current_player != node.parent.game_state.current_player else 1

            # Score Difference
            _, own_score = checkBox(
                node.game_state.board, node.game_state.current_player)
            _, oppo_score = checkBox(
                node.game_state.board, node.game_state.current_player*-1)
            difference_score = own_score-oppo_score

            # Box chain
            next_chain = play_next*checkBoxChain(node.game_state.board)

            # number of box
            total_boxes = (node.game_state.m-1)*(node.game_state.n-1)

            total_score = difference_score+next_chain*100/(total_boxes*100)

            return total_score if node.game_state.current_player == self.symbol else total_score*-1

        def expand_new_node(node: MCTSNode):
            policy = self.select_meth_bot.PredictPolicy(node.game_state.board)
            for move in node.legal_move:
                r, c = move
                move_prior = policy[r*node.game_state.board_cols + c]

                new_game_state = deepcopy(node.game_state)
                new_game_state.current_player, _ = make_move(
                    new_game_state.board, r, c, new_game_state.current_player)
                new_game_state.history_8board.append(new_game_state.board)

                new_node = MCTSNode(new_game_state, node, move, move_prior)

                node.children.append(new_node)

            return node

        node.expand = True

        node = expand_new_node(node)

        node.score = evaluate(node)

    def backup(self, node: MCTSNode):
        current_score = node.score
        node.visited += 1

        node = node.parent
        while node:
            node.visited += 1
            node.score += current_score

            node = node.parent
