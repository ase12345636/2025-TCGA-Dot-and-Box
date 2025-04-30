// state.hpp
#ifndef STATE_HPP
#define STATE_HPP

#include <vector>

struct STATE {
    std::vector<int> p1_p2_scores;
    std::vector<std::vector<int>> board;
    int m;
    int n;
    int board_rows;
    int board_cols;
    int current_player;
};

int evalute(STATE state, const int& symbol);
int alphabeta(STATE state, int depth, int max_depth, int alpha, int beta, int symbol, bool maximizing);

#endif
