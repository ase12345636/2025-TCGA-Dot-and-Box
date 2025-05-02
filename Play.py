from Human import Human
from RandomBot import Random_Bot,Greedy_Bot
from Dots_and_Box import *
from DeepLearning import *
from Alpha.AlphaBeta import AlphaBetaPlayer
from Alpha.C_AB import C_AB_player
from Alpha.MCTS import MCTSPlayer
from AlphaGo.AlphaGoMCTS import AlphaGoMCTSPlayer
from AlphaGo.AlphaGoMCTS_Ver3 import AlphaGoMCTSPlayer_Ver3
from arg import *
import os

size_m = m
size_n = n
zero_board = np.zeros((2*size_m-1, 2*size_n-1))

game_state = STATE(
    p1_p2_scores=[0, 0],
    board=[[]],
    m=size_m,
    n=size_n,
    current_player=-1,
    history_8board=deque([zero_board.copy() for _ in range(
        history_move)], maxlen=history_move)  # 預填8個零矩陣
)

game = DotsAndBox(game_state)

def main():
    args_Res['train'] = False

    cab_1 = C_AB_player(-1, game_state, 4)
    cab_2 = C_AB_player(1, game_state, 4)

    AZ_MCTS_Ver3_1 = AlphaGoMCTSPlayer_Ver3(50, game, -1, 2)
    AZ_MCTS_Ver3_2 = AlphaGoMCTSPlayer_Ver3(50, game, 1, 2)
    
    # AZ_MCTS, Draw, C_AB
    winner = [0, 0, 0]
    for _ in range(1):
        game.NewGame()
        winner[game.play(AZ_MCTS_Ver3_1, cab_2, verbose=False,
                         train=True)+1] += 1
        game.NewGame()
        winner[game.play(cab_1, AZ_MCTS_Ver3_2, verbose=False, train=True)*-1+1] += 1

    print(f'AZ_MCTS_Ver3:   {winner[0]}')
    print(f'Draw:           {winner[1]}')
    print(f'C_AB:           {winner[2]}')


if __name__ == "__main__":
    main()