import math
import numpy as np

import board as board
import config

from einops import rearrange


def normalize_with_mask(x, mask):
    x += 1e-10
    x_masked = np.multiply(x, mask)
    x_normalized = x_masked / np.sum(x_masked)
    return x_normalized


def detect_player(node_parent, player):
    if node_parent.fake:
        return player

    else:
        return player if not node_parent.board.next else -player


class FakeNode:
    def __init__(self):
        self.parent = None
        self.edge_N = np.zeros([config.all_moves_num], dtype=float)
        self.edge_W = np.zeros([config.all_moves_num], dtype=float)
        self.fake = True


class Node:
    def __init__(self, parent, move, player, board: board.Board = None):
        self.parent = parent
        self.expanded = False
        self.move = move
        self.board = board or parent.board.make_move(parent.player, move)
        self.legal_moves = self.board.get_legal_moves()
        self.child_nodes = {}
        self.is_game_root = False
        self.is_search_root = False
        self.is_terminal = False
        self.player = detect_player(parent, player)
        self.fake = False

        # policy of move
        self.pi = np.zeros([config.all_moves_num], dtype=float)
        # visit count
        self.edge_N = np.zeros([config.all_moves_num], dtype=float)
        # total action value
        self.edge_W = np.zeros([config.all_moves_num], dtype=float)
        # prior probability of selecting self
        self.edge_P = np.zeros([config.all_moves_num], dtype=float)

    @property
    # mean action value
    def edge_Q(self):
        return self.edge_W / (self.edge_N + (self.edge_N == 0))

    @property
    # variant of the PUCT algorithm
    def edge_U(self):
        return config.c_puct * self.edge_P * math.sqrt(max(1, self.self_N)) / (1 + self.edge_N)

    @property
    # variant of the PUCT algorithm with noise
    def edge_U_with_noise(self):
        noise = normalize_with_mask(np.random.dirichlet(
            [config.noise_alpha] * config.all_moves_num), self.legal_moves)
        p_with_noise = self.edge_P * \
            (1 - config.noise_weight) + noise * config.noise_weight
        return config.c_puct * p_with_noise * math.sqrt(max(1, self.self_N)) / (1 + self.edge_N)

    @property
    # used to select best next node
    def edge_Q_plus_U(self):
        if self.is_search_root:
            return self.edge_Q * self.player + self.edge_U_with_noise + self.legal_moves * 1000
        else:
            return self.edge_Q * self.player + self.edge_U + self.legal_moves * 1000

    @property
    def self_N(self):
        return self.parent.edge_N[self.move]

    @self_N.setter
    def self_N(self, n):
        self.parent.edge_N[self.move] = n

    @property
    def self_W(self):
        return self.parent.edge_W[self.move]

    @self_W.setter
    def self_W(self, w):
        self.parent.edge_W[self.move] = w

    # make input feature map
    # 1 st ~ (config.history_num) th feature map: self's history board
    # (config.history_num + 1) th ~ (config.history_num * 2) th feature map: enemy's history board
    # (config.history_num * 2 + 1) th feature map: player
    # feature map's shape: config.N_data, config.N_data, config.history_num * 2 + 1
    def to_features(self):
        features = np.zeros([config.history_num * 2 + 1,
                            config.N_data, config.N_data], dtype=float)
        player = self.player
        current = self

        for i in range(config.history_num):
            own, enemy = current.board.get_own_and_enemy(player)
            features[2 * i] = own
            features[2 * i + 1] = enemy

            if current.is_game_root:
                break

            current = current.parent

        features[config.history_num * 2] =\
            np.ones([config.N_data, config.N_data], dtype=float)*player

        return rearrange(features, 'c h w -> h w c')


class MCTS_Batch:
    def __init__(self, nn):
        self.nn = nn

    def select(self, nodes):
        best_nodes_batch = [None] * len(nodes)
        for i, node in enumerate(nodes):
            current = node

            # expend new subtree
            while current.expanded:
                # choose best next node
                best_edge = np.argmax(current.edge_Q_plus_U)

                if best_edge not in current.child_nodes:
                    current.child_nodes[best_edge] = Node(
                        current, best_edge, -current.player)

                # detect if the tree is end
                if current.is_terminal:
                    break

                if best_edge == config.pass_move and current.child_nodes[best_edge].legal_moves[config.pass_move] == 1:
                    current.is_terminal = True
                    break

                current = current.child_nodes[best_edge]

            # get last node
            best_nodes_batch[i] = current

        return best_nodes_batch

    def expand_and_evaluate(self, nodes_batch):
        # get feature map
        features_batch = np.zeros(
            [len(nodes_batch), config.N_data, config.N_data, config.history_num * 2 + 1], dtype=float)

        for i, node in enumerate(nodes_batch):
            node.expanded = True
            features_batch[i] = node.to_features()

        # use nn to predict policy, and current node's action value
        p_batch, v_batch = self.nn.f_batch(features_batch)

        # filter legal move, and norm their probability, prepared for PUCT algorithm
        for i, node in enumerate(nodes_batch):
            node.edge_P = normalize_with_mask(p_batch[i], node.legal_moves)

        return v_batch

    # backup each node's visited-count, action value
    def backup(self, nodes_batch, v_batch):
        for i, node in enumerate(nodes_batch):
            current = node

            while True:
                # accumulate visited-count
                current.self_N += 1
                # accumulate action value
                current.self_W += v_batch[i]

                if current.is_search_root:
                    break

                current = current.parent

    def search(self, nodes):
        best_nodes_batch = self.select(nodes)
        v_batch = self.expand_and_evaluate(best_nodes_batch)
        self.backup(best_nodes_batch, v_batch)

    def alpha(self, nodes, temperature):
        for i in range(config.simulations_num):
            self.search(nodes)

        # get next move
        pi_batch = np.zeros([len(nodes), config.all_moves_num], dtype=float)

        for i, node in enumerate(nodes):
            n_with_temperature = node.edge_N**(1 / temperature)
            sum_n_with_temperature = np.sum(n_with_temperature)

            # game end
            if sum_n_with_temperature == 0:
                node.pi = np.zeros([config.all_moves_num], dtype=float)
                node.pi[config.pass_move] = 1

            # final probability of each move
            else:
                node.pi = n_with_temperature / sum_n_with_temperature

            pi_batch[i] = node.pi

        return pi_batch
