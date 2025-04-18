from DaB_Utils import *

class Human():
    def __init__(self, m, n, verbose = False):
        self.verbose = verbose
        self.board_rows = 2*m - 1
        self.board_cols = 2*n - 1
    def get_move(self, board, player):
        while True:
            parts = input("Please input a coordinates of the index for the board (e.g. '1 2'):\n").split()
            if len(parts) != 2:
                print("Wrong input format! Please enter again.")
                continue
            r = int(parts[0])
            c = int(parts[1])
            if not isValid(board, r, c):
                print("invalid move!!!")
                continue
            return (r, c), []

# test = Human(3)
# print(test.get_move())