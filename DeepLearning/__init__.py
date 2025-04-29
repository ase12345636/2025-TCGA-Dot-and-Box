import numpy as np
import copy
from Dots_and_Box import *
from RandomBot import *
from DeepLearning.DaB_Model import DaB_ResNet, DaB_ValueNet
from einops import rearrange

from DeepLearning import *
from arg import *

from Alpha.C_AB import C_AB_player
class BaseBot():
    # Initiallize
    def __init__(self, input_size_m, input_size_n, game:DotsAndBox, args):
        self.input_size_m = input_size_m * 2 - 1
        self.input_size_n = input_size_n * 2 - 1

        self.total_move = input_size_m * (input_size_n-1) + \
            (input_size_m-1) * input_size_n
        self.game = game
        self.args = args
        self.collect_gaming_data = True
        self.history = []

    def PredictValue(self, para_board, player):
        board = copy.deepcopy(para_board)
        playerlikeboard = np.ones_like(board) * player # 全1or-1的盤(current player)

        input_x = [board, playerlikeboard] #(9x9x2)
        predictValue = self.valueNet_model.predict(
                np.expand_dims(input_x, axis=0).astype(float))
        return predictValue

    def PredictPolicy(self, para_board):
        board = copy.deepcopy(para_board)
        predict = self.model.predict(
                np.expand_dims(board, axis=0).astype(float))

        valid_positions = getValidMoves(para_board)
        valids = np.zeros(
            (self.input_size_m * self.input_size_n,), dtype='int')
        for pos in valid_positions:
            idx = pos[0] * self.input_size_n + pos[1]
            valids[idx] = 1

        # Filtered invalid move and avoid invalid loop
        predict = (predict+1e-30) * valids

        return predict

    # Get move predicted by model
    def get_move(self, para_board, player):
        board = copy.deepcopy(para_board)

        # Type 0
        if (self.args['type'] == 0):

            # Predict move
            predict = self.model.predict(
                np.expand_dims(board, axis=0).astype(float))

        # Detect which move is valid
        valid_positions = getValidMoves(para_board)
        valids = np.zeros(
            (self.input_size_m * self.input_size_n,), dtype='int')
        for pos in valid_positions:
            idx = pos[0] * self.input_size_n + pos[1]
            valids[idx] = 1

        # Filtered invalid move and avoid invalid loop
        predict = (predict+1e-30) * valids

        # Get final prediction
        position = np.argmax(predict)
        # if (len(predict) - np.sum(predict == 0) > 2) and self.args['train'] == True:
        #     print("random")
        #     # 當 predict 中非零數>5 且為訓練模式下，取前2高機率的隨機一項增加隨機性
        #     position = np.random.choice(np.argsort(predict)[-2:])
        # else:
        #     print("max")
        #     # 剩不到2個非零的時候才選最高
        #     position = np.argmax(predict)

        # 把非零位置轉換成坐標系，當成validmoves由大到小排序進入greedyalg中
        non_zero_predict = np.argsort(predict)[np.sum(predict == 0)-len(predict):][::-1]
        predict_moves = []
        for n_z_p in non_zero_predict:
            nonzeropos = (n_z_p // self.input_size_n,
                          n_z_p % self.input_size_n)
            predict_moves.append(nonzeropos)

        greedy_board = copy.deepcopy(self.game.state.board)
        if self.args['train'] and (greedy_move := GreedAlg(board=greedy_board, ValidMoves=predict_moves, verbose=True)):
            r, c = greedy_move
            position = r*self.input_size_n+c

        # Append current board to history
        if self.collect_gaming_data:
            tmp = np.zeros_like(predict)
            tmp[position] = 1.0
            # print("current board")
            # for i in range(len(self.history)):
            #     print(self.history[i][0])

        position = (position // self.input_size_n,
                    position % self.input_size_n)

        return position, [board, tmp, player]

    # Training model based on history
    def self_play_train(self,oppo = None):
        # Generate history data
        def gen_data(type: 0, self_first=True): # self_first: 自己為先手

            # Data augmentation by getting symmetries
            def getSymmetries(board, pi):
                pi_board = np.reshape(
                    pi, (self.input_size_m, self.input_size_n))
                symmetries = []
                for i in range(4):
                    for flip in [True, False]:
                        newB = np.rot90(board, i)
                        newPi = np.rot90(pi_board, i)
                        if flip:
                            newB = np.fliplr(newB)
                            newPi = np.fliplr(newPi)
                        symmetries.append((newB, list(newPi.ravel())))
                return symmetries

            # Initiallize history
            self.history = []

            # Get history data
            self.game.NewGame()
            # AB VS AB對下
            AB1 = C_AB_player(-1, self.game.state, 6)
            AB2 = C_AB_player( 1, self.game.state, 6)
            self.game.play(AB1, AB2,train = True, verbose=False)

            for i in range(len(self.history)):
                print(self.history[i][0])

            # Process history data
            history = []
            self.history = copy.deepcopy(self.game.history)
            print(f"history len:{len(self.history)}")
            # Data augmentation
            for step, (board, probs, player) in enumerate(self.history):
                sym = getSymmetries(board, probs)
                for b, p in sym:
                    history.append([b, p, player])
            # history: [[b, p, player],[b, p, player],[b, p, player],....]

            self.history.clear()
            game_result = GetWinner(self.game.state.board, self.game.state.p1_p2_scores)

            # valueNet training data: x:board, y: good or bad for current player in this state(board)
            valueNet_his = []
            policyNet_his = []
            for game_data in history:   # game_data: [b,p,player]
                value = 0   # default: tie -> value is 0
                if game_data[2] == game_result: # current player win -> value is 1
                    value = 1
                    policyNet_x = copy.deepcopy(game_data[0])
                    policyNet_y = game_data[1]
                    policyNet_his.append((policyNet_x, policyNet_y))
                elif game_result == 0:
                    value = 0
                    policyNet_x = copy.deepcopy(game_data[0])
                    policyNet_y = game_data[1]
                    policyNet_his.append((policyNet_x, policyNet_y))

                elif game_data[2] == -game_result: # current player lose -> value is -1
                    value = -1
                playerlikeboard = np.ones_like(game_data[0]) * game_data[2] # 全1or-1的盤(current player)
                valueNet_x = [copy.deepcopy(game_data[0]), playerlikeboard]
                valueNet_y = value
                valueNet_his.append((valueNet_x, valueNet_y))    # [((board, player), value)]
            # Type 0
            if (type == 0):
                return policyNet_his, valueNet_his

        # Allow collecting history
        self.collect_gaming_data = True

        # Generate data
        data = []
        valueNet_data = []
        for i in range(self.args['num_of_generate_data_for_train']):
            if self.args['verbose']:
                print(f'Self playing {i + 1}')
            current_data = gen_data(self.args['type'],i%2)
            data += current_data[0]
            valueNet_data += current_data[1]

        self.collect_gaming_data = False
        # Training model
        print(f"Length of data: {len(data)}")
        history = self.model.fit(
            data, batch_size=self.args['batch_size'], epochs=self.args['epochs'])
        self.model.save_weights()
        self.model.plot_learning_curve(history)

        valueNet_history = self.valueNet_model.fit(
            valueNet_data, batch_size=self.args['batch_size'], epochs=self.args['epochs'])
        self.valueNet_model.save_weights()
        self.valueNet_model.plot_learning_curve(valueNet_history)

class ResnetBOT(BaseBot):
    def __init__(self, input_size_m, input_size_n, game, args):
        super().__init__(input_size_m, input_size_n, game, args)

        self.model = DaB_ResNet(input_shape=(
            self.input_size_m, self.input_size_n, self.total_move), args=args)

        self.valueNet_model = DaB_ValueNet(input_shape=(
            self.input_size_m, self.input_size_n, 2), args=args_ValueNet)

        try:
            self.model.load_weights(self.args['load_model_name'])
            self.valueNet_model.load_weights(load_model_name = None)
            # print(f'{self.model.model_name} loaded')
        except Exception as e:
            print(f'No model exists, \n{e}')
