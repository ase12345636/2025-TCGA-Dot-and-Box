from game_ai import PyState, py_evalute, py_alphabeta

s = PyState()

s.set_dimensions(3, 3, 5, 5)  # m, n, board_rows, board_cols
s.set_board([[5,0,5],[0,8,0],[5,0,5]])
s.set_scores([0, 0])
s.set_current_player(-1)

print(s.board)
print(py_evalute(s, 1))
print(py_alphabeta(s, 0, 1, -999999, 999999, 1, True))
