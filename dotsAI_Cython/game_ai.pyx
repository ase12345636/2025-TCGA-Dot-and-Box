# distutils: language = c++

from libcpp.vector cimport vector
from libcpp.utility cimport pair
from libcpp cimport bool

cdef extern from "state.cpp":
    struct STATE:
        vector[int] p1_p2_scores
        vector[vector[int]] board
        int m
        int n
        int board_rows
        int board_cols
        int current_player

    int evalute(STATE state, const int& symbol)
    int alphabeta(STATE state, int depth, int max_depth, int alpha, int beta, int symbol, bool maximizing)

cdef class PyState:
    cdef STATE cpp_state  # ✅ 用 struct，不用指標

    def __cinit__(self):
        self.cpp_state.p1_p2_scores.push_back(0)
        self.cpp_state.p1_p2_scores.push_back(0)

    def init_state(self, int m, int n, int board_rows, int board_cols, int current_player):
        self.cpp_state.m = m
        self.cpp_state.n = n
        self.cpp_state.board_rows = board_rows
        self.cpp_state.board_cols = board_cols
        self.cpp_state.current_player = current_player
        self.cpp_state.board.clear()
        for i in range(m):
            row = vector[int]()
            for j in range(n):
                row.push_back(0)
            self.cpp_state.board.push_back(row)

    def set_board(self, list py_board):
        cdef int i, j
        self.cpp_state.board.clear()
        for row in py_board:
            v_row = vector[int]()
            for val in row:
                v_row.push_back(val)
            self.cpp_state.board.push_back(v_row)

    def set_scores(self, list scores):
        if len(scores) != 2:
            raise ValueError("scores 必須是長度為 2 的 list")
        self.cpp_state.p1_p2_scores[0] = scores[0]
        self.cpp_state.p1_p2_scores[1] = scores[1]

    def set_current_player(self, int player):
        self.cpp_state.current_player = player

    def set_dimensions(self, int m, int n, int board_rows, int board_cols):
        self.cpp_state.m = m
        self.cpp_state.n = n
        self.cpp_state.board_rows = board_rows
        self.cpp_state.board_cols = board_cols

    cdef STATE get_cpp_state(self):
        return self.cpp_state

def py_evalute(PyState state, int symbol):
    return evalute(state.get_cpp_state(), symbol)

def py_alphabeta(PyState state, int depth, int max_depth, int alpha, int beta, int symbol, bint maximizing):
    return alphabeta(state.get_cpp_state(), depth, max_depth, alpha, beta, symbol, maximizing)
