from Dots_and_Box import DotsAndBox as DaB
from DeepLearning import ResnetBOT
from arg import m, n, history_move, args_Res
from RandomBot import *
# from Alpha.AlphaBeta import AlphaBetaPlayer

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
game = DaB(game_state, True)
# bot_CNN = CNNBOT(input_size_m=m, input_size_n=n, game=game, args=args_CNN)
bot_Res = ResnetBOT(input_size_m=m, input_size_n=n, game=game, args=args_Res)
# bot_LSTM = LSTM_BOT(input_size_m=m, input_size_n=n, game=game, args=args_LSTM)
# bot_ConvLSTM = ConvLSTM_BOT(
#     input_size_m=m, input_size_n=n, game=game, args=args_ConvLSTM)
# bot_Conv2Plus1D = Conv2Plus1D_BOT(
#    input_size_m=m, input_size_n=n, game=game, args=args_Conv2Plus1D)


args_Res['train'] = True  # True:開greedy, False:關
args_Oppo = {
    'verbose': True,
    'type': 0,
    'train': False,  # 對手關閉random
    'load_model_name': None
}

# oppo_bot = ResnetBOT(input_size_m=m, input_size_n=n, game=game, args=args_Oppo)
# oppo_bot = Greedy_Bot(game)


# bot_CNN.self_play_train()
print(args_Res)
bot_Res.self_play_train(oppo=None)
# bot_Res.train_from_json()
# bot_LSTM.self_play_train()
# bot_ConvLSTM.self_play_train()
# bot_Conv2Plus1D.self_play_train()
