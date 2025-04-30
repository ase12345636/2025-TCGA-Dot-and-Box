// state.cpp
#include "state.hpp"
#include <iostream>
#include <vector>
#include <limits>
#include <algorithm>

using std::cout;
using std::endl;
using std::pair;
using std::vector;

bool isValid(const vector<vector<int>> &board, int r, int c)
{
    return board[r][c] == 0;
}

vector<pair<int, int>> getValidMoves(const vector<vector<int>> &board)
{
    vector<pair<int, int>> ValidMoves;
    for (int i = 0; i < (int)board.size(); i++)
    {
        for (int j = 0; j < (int)board[0].size(); j++)
        {
            if (board[i][j] == 0)
                ValidMoves.emplace_back(i, j);
        }
    }
    return ValidMoves;
}

pair<bool, int> checkBox(vector<vector<int>> &board, int player)
{
    bool box_filled = false;
    int score = 0;
    int m = ((int)board.size() + 1) / 2;
    int n = ((int)board[0].size() + 1) / 2;
    for (int i = 0; i < m - 1; i++)
    {
        for (int j = 0; j < n - 1; j++)
        {
            int box_i = 2 * i + 1;
            int box_j = 2 * j + 1;
            if (board[box_i][box_j] == 8 &&
                board[box_i - 1][box_j] != 0 &&
                board[box_i + 1][box_j] != 0 &&
                board[box_i][box_j - 1] != 0 &&
                board[box_i][box_j + 1] != 0)
            {
                board[box_i][box_j] += player;
                score += 1;
                box_filled = true;
            }
        }
    }
    return {box_filled, score};
}

pair<int, int> make_move(vector<vector<int>> &board, int r, int c, int &player)
{
    int score = 0;
    if (!isValid(board, r, c))
    {
        cout << "Invalid move!\n";
        return {player, score};
    }
    board[r][c] = player;
    auto [filled, box_score] = checkBox(board, player);
    if (!filled)
    {
        player *= -1;
    }
    return {player, box_score};
}

bool isGameOver(const vector<vector<int>> &board)
{
    return getValidMoves(board).empty();
}

int GetWinner(const vector<vector<int>> &board, const vector<int> &scores)
{
    if (!isGameOver(board))
        return 999;
    if (scores[0] > scores[1])
        return -1;
    if (scores[0] < scores[1])
        return 1;
    return 0;
}

int evalute(STATE state, const int &symbol)
{
    // int winner = GetWinner(state.board, state.p1_p2_scores);
    // if (winner == symbol) return 10000;
    // if (winner == 0) return 5000;
    // if (winner == -symbol) return -10000;
    int score_diff = state.p1_p2_scores[0] - state.p1_p2_scores[1];
    if (symbol == 1)
        score_diff *= -1;
    return score_diff * 100;
}

int alphabeta(STATE state, int depth, int max_depth, int alpha, int beta, int symbol, bool maximizing)
{
    if (depth >= max_depth || GetWinner(state.board, state.p1_p2_scores) != 999)
        return evalute(state, symbol);

    vector<pair<int, int>> validmoves = getValidMoves(state.board);

    if (maximizing)
    {
        int max_eval = -std::numeric_limits<int>::max();
        for (const auto &move : validmoves)
        {
            STATE new_state = state;
            int last_player = new_state.current_player;
            auto [_, score] = make_move(new_state.board, move.first, move.second, new_state.current_player);
            if (new_state.current_player == -1)
                new_state.p1_p2_scores[0] += score;
            if (new_state.current_player == 1)
                new_state.p1_p2_scores[1] += score;
            // 檢查有無換人，確認是否繼續最大化
            int eval = 0;
            if (last_player != new_state.current_player)
            {
                eval = alphabeta(new_state, depth + 1, max_depth, alpha, beta, symbol, false);
            }
            else
            {
                eval = alphabeta(new_state, depth + 1, max_depth, alpha, beta, symbol, true);
            }
            delete &new_state;
            max_eval = std::max(max_eval, eval);
            alpha = std::max(alpha, eval);
            if (beta <= alpha)
                break;
        }
        return max_eval;
    }
    else
    {
        int min_eval = std::numeric_limits<int>::max();
        for (const auto &move : validmoves)
        {
            STATE new_state = state;
            int last_player = new_state.current_player;
            auto [_, score] = make_move(new_state.board, move.first, move.second, new_state.current_player);
            if (new_state.current_player == -1)
                new_state.p1_p2_scores[0] += score;
            if (new_state.current_player == 1)
                new_state.p1_p2_scores[1] += score;
            int eval = 0;
            if (last_player != new_state.current_player)
            {
                eval = alphabeta(new_state, depth + 1, max_depth, alpha, beta, symbol, true);
            }
            else
            {
                eval = alphabeta(new_state, depth + 1, max_depth, alpha, beta, symbol, false);
            }
            delete &new_state;
            min_eval = std::min(min_eval, eval);
            beta = std::min(beta, eval);
            if (beta <= alpha)
                break;
        }
        return min_eval;
    }
}
