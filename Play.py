from Human import Human
from RandomBot import Random_Bot,Greedy_Bot
from Dots_and_Box import *
from DeepLearning import *
from Alpha.AlphaBeta import AlphaBetaPlayer
from Alpha.C_AB import C_AB_player
from Alpha.MCTS import MCTSPlayer
from AlphaGo.AlphaGoMCTS import AlphaGoMCTSPlayer
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
    history_8board=deque([zero_board.copy() for _ in range(history_move)], maxlen=history_move)  # 預填8個零矩陣
)


game = DotsAndBox(game_state)
p1 = [Human(m,n), 'Human']
p2 = [Random_Bot(m,n), 'random']
p3 = [Greedy_Bot(m,n), 'greedy']

def self_play(player1, player2):
    """
    讓兩個玩家對戰，回傳比賽結果
    """
    game.NewGame()
    return game.play(player1, player2, verbose=False, train=True)

def record_result(file, game_num, bot1_name, bot2_name, bot1_win, bot2_win):
    """
    記錄對戰結果到檔案
    """
    file.write(f"Game {game_num}\n")
    file.write(f"{bot1_name} win: {bot1_win}\n")
    file.write(f"{bot2_name} win: {bot2_win}\n")
    file.write("-" * 76 + "\n")

def dual(n_game, bot1, bot2, bot1_name, bot2_name):
    """
    讓兩個 Bot 進行多場對戰，記錄勝負結果
    """
    print(f"{bot1_name} VS {bot2_name}".center(76))
    os.makedirs("game_record", exist_ok=True)  # 確保資料夾存在
    file_path = f'game_record/{bot1_name} VS {bot2_name}.txt'

    bot1_win, bot2_win = 0, 0

    with open(file_path, "a") as f:
        f.write(f"{bot1_name} VS {bot2_name}".center(76) + "\n")

        for i in range(1, n_game + 1):
            print(f"Game {i}")

            # 第一局
            result = self_play(bot1, bot2)
            if result == -1:
                print('\033[92m' + 'player 1 won!' + '\033[0m')
                bot1_win += 1
            elif result == 1:
                print('\033[92m' + 'player 2 won!' + '\033[0m')
                bot2_win += 1
            else:
                print('Draw!')
            # 記錄結果
            record_result(f, i, bot1_name, bot2_name, bot1_win, bot2_win)
            print(f"{bot1_name} win: {bot1_win}")
            print(f"{bot2_name} win: {bot2_win}")
            print("-" * 76)

        first_winning_rate = f"first: {round((bot1_win / (bot1_win + bot2_win)) * 100, 2)}%\n"

        second_bot1_win = 0
        second_bot2_win = 0
        # 先後手交換
        for i in range(1, n_game + 1):
            print(f"Game {i}")
            result = self_play(bot2, bot1)
            if result == 1:
                print('\033[92m' + 'player 1 won!' + '\033[0m')
                bot1_win += 1
                second_bot1_win += 1
            elif result == -1:
                print('\033[92m' + 'player 2 won!' + '\033[0m')
                bot2_win += 1
                second_bot2_win += 1
            else:
                print('Draw!')

            # 記錄結果
            record_result(f, i, bot1_name, bot2_name, bot1_win, bot2_win)
            print(f"{bot1_name} win: {bot1_win}")
            print(f"{bot2_name} win: {bot2_win}")
            print("-" * 76)


        second_winning_rate = f"second: {round((second_bot1_win / (second_bot1_win + second_bot2_win)) * 100, 2)}%\n"

        print(bot1_win)
        print(bot2_win)
        if bot2_win == 0:
            winning_rate = "100%"
        else:
            print((bot1_win / (bot1_win + bot2_win)))
            winning_rate = f"{round((bot1_win / (bot1_win + bot2_win)) * 100, 2)}%\n"
        print(winning_rate)

        f.write(first_winning_rate)
        f.write(second_winning_rate)
        f.write(winning_rate)

def dualAB(n_game, bot1, bot1_name,depth=3):
    """
    讓兩個 Bot 進行多場對戰，記錄勝負結果
    """
    print(f"{bot1_name} VS AlphaBeta".center(76))
    os.makedirs("game_record", exist_ok=True)  # 確保資料夾存在
    file_path = f'game_record/{bot1_name} VS AlphaBeta.txt'

    bot1_win, bot2_win = 0, 0

    with open(file_path, "a") as f:
        f.write(f"{bot1_name} VS AlphaBeta".center(76) + "\n")

        bot2 = C_AB_player(1,game.state,depth)
        for i in range(1, n_game + 1):
            print(f"Game {i}")

            # 第一局
            result = self_play(bot1, bot2)
            if result == -1:
                print('\033[92m' + 'player 1 won!' + '\033[0m')
                bot1_win += 1
            elif result == 1:
                print('\033[92m' + 'player 2 won!' + '\033[0m')
                bot2_win += 1
            else:
                print('Draw!')
            # 記錄結果
            record_result(f, i, bot1_name, "AlphaBeta", bot1_win, bot2_win)
            print(f"{bot1_name} win: {bot1_win}")
            print(f"AlphaBeta win: {bot2_win}")
            print("-" * 76)
            
        first_winning_rate = f"first: {round((bot1_win / (bot1_win + bot2_win)) * 100, 2)}%\n"
        
        second_bot1_win = 0
        second_bot2_win = 0
        bot2 = C_AB_player(-1, game.state, depth)
        # 先後手交換
        for i in range(1, n_game + 1):
            print(f"Game {i}")
            result = self_play(bot2, bot1)
            if result == 1:
                print('\033[92m' + 'player 1 won!' + '\033[0m')
                bot1_win += 1
                second_bot1_win += 1
            elif result == -1:
                print('\033[92m' + 'player 2 won!' + '\033[0m')
                bot2_win += 1
                second_bot2_win += 1
            else:
                print('Draw!')

            # 記錄結果
            record_result(f, i, bot1_name, "AlphaBeta", bot1_win, bot2_win)
            print(f"{bot1_name} win: {bot1_win}")
            print(f"AlphaBeta win: {bot2_win}")
            print("-" * 76)
        
        
        second_winning_rate = f"second: {round((second_bot1_win / (second_bot1_win + second_bot2_win)) * 100, 2)}%\n"
        
        print(bot1_win)
        print(bot2_win)
        if bot2_win == 0:
            winning_rate = "100%"
        else:
            print((bot1_win / (bot1_win + bot2_win)))
            winning_rate = f"{round((bot1_win / (bot1_win + bot2_win)) * 100, 2)}%\n"
        print(winning_rate)
        
        f.write(first_winning_rate)
        f.write(second_winning_rate)
        f.write(winning_rate)

def dualMCTS(n_game, bot1, bot1_name, exp=1.3):
    """
    讓兩個 Bot 進行多場對戰，記錄勝負結果
    """
    print(f"{bot1_name} VS AlphaBeta".center(76))
    os.makedirs("game_record", exist_ok=True)  # 確保資料夾存在
    file_path = f'game_record/{bot1_name} VS AlphaBeta.txt'

    bot1_win, bot2_win = 0, 0

    with open(file_path, "a") as f:
        f.write(f"{bot1_name} VS AlphaBeta".center(76) + "\n")

        bot2 = MCTSPlayer(10000, game_state, 1, exp)
        for i in range(1, n_game + 1):
            print(f"Game {i}")

            # 第一局
            result = self_play(bot1, bot2)
            if result == -1:
                print('\033[92m' + 'player 1 won!' + '\033[0m')
                bot1_win += 1
            elif result == 1:
                print('\033[92m' + 'player 2 won!' + '\033[0m')
                bot2_win += 1
            else:
                print('Draw!')
            # 記錄結果
            record_result(f, i, bot1_name, "AlphaBeta", bot1_win, bot2_win)
            print(f"{bot1_name} win: {bot1_win}")
            print(f"AlphaBeta win: {bot2_win}")
            print("-" * 76)
            
        first_winning_rate = f"first: {round((bot1_win / (bot1_win + bot2_win)) * 100, 2)}%\n"
        
        second_bot1_win = 0
        second_bot2_win = 0
        bot2 = MCTSPlayer(10000, game_state, -1, exp)
        # 先後手交換
        for i in range(1, n_game + 1):
            print(f"Game {i}")
            result = self_play(bot2, bot1)
            if result == 1:
                print('\033[92m' + 'player 1 won!' + '\033[0m')
                bot1_win += 1
                second_bot1_win += 1
            elif result == -1:
                print('\033[92m' + 'player 2 won!' + '\033[0m')
                bot2_win += 1
                second_bot2_win += 1
            else:
                print('Draw!')

            # 記錄結果
            record_result(f, i, bot1_name, "AlphaBeta", bot1_win, bot2_win)
            print(f"{bot1_name} win: {bot1_win}")
            print(f"AlphaBeta win: {bot2_win}")
            print("-" * 76)
        second_winning_rate = f"second: {round((second_bot1_win / (second_bot1_win + second_bot2_win)) * 100, 2)}%\n"

        print(bot1_win)
        print(bot2_win)
        if bot2_win == 0:
            winning_rate = "100%"
        else:
            print((bot1_win / (bot1_win + bot2_win)))
            winning_rate = f"{round((bot1_win / (bot1_win + bot2_win)) * 100, 2)}%\n"
        print(winning_rate)
        f.write(first_winning_rate)
        f.write(second_winning_rate)
        f.write(winning_rate)

def main():
    args_Res['train'] = False
    # args_Res['load_model_name'] = f'Resnet_model_4x4_31.h5'
    # p5 = [ResnetBOT(input_size_m=m, input_size_n=n,input_size_c=history_move*2+1, game=game, args=args_Res), 'resnet']
    # game.play(p2[0], p5[0])
    MCTS_1 = MCTSPlayer(30000,game_state, -1, 1.3, 2)
    MCTS_2 = MCTSPlayer(30000,game_state, 1, 1.3, 2)
    cab_1 = C_AB_player(-1, game_state, 4)
    cab_2 = C_AB_player(1, game_state, 4)
    # AZ_MCTS_1 = AlphaGoMCTSPlayer(1000, game, -1, 1, 2)
    # AZ_MCTS_2 = AlphaGoMCTSPlayer(1000, game, 1, 1, 2)
    # dualMCTS(5, p3[0], p3[1])
    # game.play(MCTS_1, cab_2)
    # game.play(cab_1, cab_2)
    # game.play(cab_1, p1[0], verbose=True)
    # game.play(MCTS_1, p3[0])
    # game.play(p3[0], MCTS_2)
    # game.play(p3[0], AZ_MCTS_2)
    # game.play(AZ_MCTS_1, p3[0])
    # dualAB(n_game=25,
    #         bot1=p3[0],
    #         bot1_name=p3[1],
    #         depth=5)

    for ver in range(2,3):
        args_Res['train'] = False
        args_Res['load_model_name'] = f'Resnet_model_{size_m}x{size_n}_{ver}.h5'
        p5 = [ResnetBOT(input_size_m=m, input_size_n=n, game=game, args=args_Res), f'resnet_{size_m}x{size_n}']

        # dual(n_game=50,
        #     bot1=p5[0],
        #     bot1_name=p5[1]+f'_{ver}',
        #     bot2=p3[0],
        #     bot2_name=p3[1])

        dualAB(n_game=25,
            bot1=p5[0],
            bot1_name=p5[1]+f'_{ver}',
            depth=5)
    # print(game_state.board)
    # ab = AlphaBetaPlayer(-1, game_state, 6)
    # c_ab = C_AB_player(-1, game_state, 8)
    # game.play(cab_1, cab_2, train=False)
    # game.play(p3[0], p3[0])


if __name__ == "__main__":
    main()