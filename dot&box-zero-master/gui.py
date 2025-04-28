import copy

import board
import config
from util import line_2_plane

string_board = [
    list("1 +-+-+-+-+"),
    list("2 | | | | |"),
    list("3 +-+-+-+-+"),
    list("4 | | | | |"),
    list("5 +-+-+-+-+"),
    list("6 | | | | |"),
    list("7 +-+-+-+-+"),
    list("8 | | | | |"),
    list("9 +-+-+-+-+"),
]
white_string = list('\033[37m')
blue_string = list('\033[34m')
red_string = list('\033[31m')
sign = list('●')


def print_board(board, cur_string_board):
    print("=================")
    print("  ABCDEFGHI")

    for i in range(config.N_data):
        line = cur_string_board[i][:2]
        for j in range(config.N_data):

            if (board[i][j] == config.vertex or board[i][j] == config.legal_edge or
                    board[i][j] == config.empty_box):
                line = line+white_string+list(cur_string_board[i][2+j])

            elif board[i][j] == config.blue_edge:
                line = line+blue_string+list(cur_string_board[i][2+j])

            elif board[i][j] == config.red_edge:
                line = line+red_string+list(cur_string_board[i][2+j])

            elif board[i][j] == config.blue_box:
                line = line+blue_string+sign

            elif board[i][j] == config.red_box:
                line = line+red_string+sign
        line = line+white_string

        print("".join(line))


def print_node(node, init=False):
    cur_string_board = copy.deepcopy(string_board)
    print_board(node.board.board, cur_string_board)

    if not init:
        last_move = ""
        if node.move == config.pass_move:
            last_move = "pass"
        else:
            last_move = line_2_plane(node.move)

        if node.parent.fake or node.parent.player == config.blue:
            print("blue plays " + last_move + ".")

        else:
            print("red plays " + last_move + ".")

        print("=================")
