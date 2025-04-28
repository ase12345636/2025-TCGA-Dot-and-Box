import numpy as np
import config


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


# initiallize board with shape (2*N-1) * (2*N-1)
def initialize_board():
    board = np.zeros((config.N_data, config.N_data), dtype=float)

    for i in range(config.N_data):
        for j in range(config.N_data):
            if i % 2 == 0 and j % 2 == 0:
                board[i][j] = config.vertex
            elif i % 2 == 1 and j % 2 == 1:
                board[i][j] = config.empty_box

    return board


if __name__ == '__main__':
    x = np.array([1, 2, 3, 4, 55])
    y = x
    y += np.array([1, 2, 3, 4, 55])

    print(x)
    print(y)
