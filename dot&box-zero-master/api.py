import numpy as np

import subprocess
from subprocess import PIPE, STDOUT, Popen

import config
from util import line_2_plane, log, plane_2_line

from random_bot.RandomBot import Random_Bot, Greedy_Bot


class HumanPlayer:
    def make_move(self, current_node):
        human_input = -1
        while True:
            human_input_str = input(">")
            if human_input_str == "pass":
                human_input = config.pass_move
            else:
                human_input = plane_2_line(human_input_str)

            if human_input is None or current_node.legal_moves[human_input] == 0:
                print("illegal.")
            else:
                return human_input


class RandomPlayer:
    def make_move(self, current_node):
        return Random_Bot(config.N, config.N).get_move(
            current_node.board.board, current_node.player)


class GreedyPlayer:
    def make_move(self, current_node):
        return Greedy_Bot(config.N, config.N).get_move(
            current_node.board.board, current_node.player)
