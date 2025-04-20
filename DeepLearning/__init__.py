import numpy as np
import copy
from Dots_and_Box import DotsAndBox
from DeepLearning.DaB_Model import DaB_ResNet
from RandomBot import *
from einops import rearrange

from DeepLearning import *
from arg import *

from Alpha.C_AB import C_AB_player
class BaseBot():
    # Initiallize
    def __init__(self, input_size_m, input_size_n, game, args):
        self.input_size_m = input_size_m * 2 - 1
        self.input_size_n = input_size_n * 2 - 1

        self.total_move = input_size_m * (input_size_n-1) + \
            (input_size_m-1) * input_size_n
        self.game = game
        self.args = args
        self.collect_gaming_data = True
        self.history = []

    # Get move predicted by model
    def get_move(self, para_board, player):
        # board = self.preprocess_board(self.game.state.board)
        board = copy.deepcopy(para_board)

        # Type 0
        if (self.args['type'] == 0):

            # Predict move
            predict = self.model.predict(
                np.expand_dims(board, axis=0).astype(float))

        # Detect which move is valid
        valid_positions = getValidMoves(board)
        valids = np.zeros(
            (self.input_size_m * self.input_size_n,), dtype='int')
        for pos in valid_positions:
            idx = pos[0] * self.input_size_n + pos[1]
            valids[idx] = 1

        # Filtered invalid move and avoid invalid loop
        predict = (predict+1e-30) * valids

        # Get final prediction
        total_moves = (self.game.state.m - 1) * self.game.state.n + self.game.state.m * (self.game.state.n - 1)
        remaining_moves = len(getValidMoves(board))
        progress = 1 - remaining_moves / total_moves
        position = np.argmax(predict)
        # if  progress < 0.3 and self.args['train'] == True:
        #     print("random")
        #     position = np.random.choice(np.argsort(predict)[-2:])
        # else:
        #     print("max")
        #     # 剩不到2個非零的時候才選最高
        #     position = np.argmax(predict)

        # 把非零位置轉換成坐標系，當成validmoves由大到小排序進入greedyalg中
        if self.args['train'] and progress < 0.8:
            non_zero_predict = np.argsort(predict)[np.sum(predict == 0)-len(predict):][::-1]
            predict_moves = []
            for n_z_p in non_zero_predict:
                nonzeropos = (n_z_p // self.input_size_n,
                            n_z_p % self.input_size_n)
                predict_moves.append(nonzeropos)

            greedy_board = copy.deepcopy(board)
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

            # Split data and get 8 symmetries of history
            def splitSymmetries(sym):
                split_sym = []
                for i in range(8):
                    temp = []
                    j = i
                    while (j < len(sym)):
                        temp.append(sym[j])
                        j += 8

                    split_sym.append(temp)

                return split_sym

            # Initiallize history
            self.history = []

            # Get history data
            self.game.NewGame()
            # if oppo and self_first: # self先手
            #     print('self first')
            #     self.game.play(self, oppo,train = True)
            # elif oppo and not self_first:   # self後手
            #     print('oppo first')
            #     self.game.play(oppo, self,train = True)
            # else:   # 自行對下
            #     self.game.play(self, self,train = True)

            # 改成自身 VS AB輪流對下
            if self_first:
                print('Self first')
                AB = C_AB_player(symbol=1,      #AB做後手
                                 state= self.game.state,
                                 max_depth=6
                                )
                self.game.play(self, AB,train = True, verbose = False)
            elif not self_first:   # AB先手
                print('AB first')
                AB = C_AB_player(symbol=-1,      #AB做先手
                                 state= self.game.state,
                                 max_depth=6
                                )
                self.game.play(AB, self,train = True, verbose = False)

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
            self.history.clear()
            game_result = GetWinner(self.game.state.board, self.game.state.p1_p2_scores)
            # Type 0
            if (type == 0):
                return [(x[0], x[1]) for x in history if (game_result == 0 or x[2] == game_result)]

        # Allow collecting history
        self.collect_gaming_data = True

        # Generate data
        data = []
        for i in range(self.args['num_of_generate_data_for_train']):
            if self.args['verbose']:
                print(f'Self playing {i + 1}')
            current_data = gen_data(self.args['type'],i%2)
            data += current_data

        self.collect_gaming_data = False

        # Training model
        print(f"Length of data: {len(data)}")
        history = self.model.fit(
            data, batch_size=self.args['batch_size'], epochs=self.args['epochs'])
        self.model.save_weights()
        self.model.plot_learning_curve(history)

class ResnetBOT(BaseBot):
    def __init__(self, input_size_m, input_size_n, game, args):
        super().__init__(input_size_m, input_size_n, game, args)

        self.model = DaB_ResNet(input_shape=(
            self.input_size_m, self.input_size_n, self.total_move), args=args)
        try:
            self.model.load_weights(self.args['load_model_name'])
            # print(f'{self.model.model_name} loaded')
        except Exception as e:
            print(f'No model exists, \n{e}')