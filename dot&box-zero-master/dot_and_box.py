import argparse
import gc
import os
import random
import traceback
from multiprocessing import Pool, Process

import numpy as np
import tensorflow as tf

import api
import board
import config
import gui
import net
import tree
from util import log


class SelfPlayGame:
    def __init__(self, worker_id, batch_size=config.self_play_batch_size, echo_max=config.self_play_echo_max):
        self.version = 0
        self.echo = 0
        self.echo_max = echo_max
        self.worker_id = worker_id
        self.batch_size = batch_size
        self.fake_nodes = [None] * batch_size
        self.current_nodes = [None] * batch_size

    # collect training data
    def start(self):
        # set memory used with each process
        gpu_options = tf.compat.v1.GPUOptions(
            per_process_gpu_memory_fraction=config.self_play_woker_gpu_memory_fraction)

        with tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options)) as session:
            # restore model
            saver = tf.compat.v1.train.Saver()
            self.restore(session, saver)
            nn = net.NN(session)
            mcts_batch = tree.MCTS_Batch(nn)

            # collect training data with loop
            while self.echo < self.echo_max:
                log("selfplay worker", self.worker_id, "version:",
                    self.version, "echo:", self.echo, "session start.")
                self.play(mcts_batch)
                self.save()
                self.echo += 1

            log("selfplay worker", self.worker_id, "session end.")

    # play the game
    def play(self, mcts_batch):
        terminals_num = 0
        moves_num = 0

        # create batch_size tree roots
        for i in range(self.batch_size):
            self.fake_nodes[i] = tree.FakeNode()
            self.current_nodes[i] = tree.Node(
                self.fake_nodes[i], 0, config.blue, board.Board())
            self.current_nodes[i].is_game_root = True
            self.current_nodes[i].is_search_root = True

        # simulate game
        while terminals_num != self.batch_size:
            terminals_num = 0
            moves_num += 1

            gc.collect()
            pi_batch = mcts_batch.alpha(
                self.current_nodes, get_temperature(moves_num))

            for i in range(self.batch_size):
                # detect if MCTS of each tree is done
                if self.current_nodes[i].is_terminal is True:
                    terminals_num += 1

                # pick moves random
                else:
                    move = pick_move_probabilistically(pi_batch[i])
                    self.current_nodes[i] = make_move(
                        self.current_nodes[i], move)

    def save(self):
        data = []
        for node in self.current_nodes:
            winner = 0
            blue_box_num = node.board.count_score(config.blue)
            red_box_num = node.board.count_score(config.red)
            if blue_box_num > red_box_num:
                winner = -1
            elif blue_box_num < red_box_num:
                winner = 1

            current = node
            while True:
                # instance
                data.append(current.to_features())
                # ground truth
                data.append(current.pi)
                data.append(winner)

                if current.is_game_root:
                    break
                current = current.parent

        # zip training data
        np.savez_compressed(config.data_path + "{0:03d}_{1:03d}_{2:02d}{3:02d}".format(
            self.batch_size, self.version, self.worker_id, self.echo), data=data, dtype=object)

    def restore(self, session, saver):
        checkpoint_name = restore_from_last_checkpoint(session, saver)
        if checkpoint_name:
            self.version = int(checkpoint_name[1:].split('-')[0])

        last_echo = -1
        npz_file_names = get_npz_file_names()
        for file_name in npz_file_names:
            file_name_splited = file_name.split('_')
            if int(file_name_splited[-1][:2]) == self.worker_id:
                if int(file_name_splited[1]) < self.version:
                    os.rename(config.data_path + file_name,
                              config.archives_path + file_name)
                else:
                    this_echo = int(file_name_splited[-1][2:4])
                    if this_echo > last_echo:
                        last_echo = this_echo

        self.echo = last_echo + 1


class Train:
    def __init__(self, batch_size=config.train_batch_size, epoch_max=config.train_epoch_max):
        self.version = 0
        self.state_data = np.zeros(
            (0, config.N_data, config.N_data, config.history_num * 2 + 1), dtype=float)
        self.pi_data = np.zeros((0, config.all_moves_num), dtype=float)
        self.z_data = np.zeros((0, 1), dtype=float)
        self.batch_size = batch_size
        self.epoch_max = epoch_max
        self.data_len = self.load_data()
        self.batch_num = (self.data_len // self.batch_size) + 1
        self.global_step = 0

    def start(self):
        if self.data_len == 0:
            log("no data for training.")
            return

        with tf.compat.v1.Session() as session:
            saver = tf.compat.v1.train.Saver(
                max_to_keep=config.train_checkpoint_max_to_keep)
            self.restore(session, saver)
            nn = net.NN(session)
            log("training version:", self.version, "global step:",
                self.global_step, "session start.")

            with open(config.log_path + "loss_log.csv", "a+") as loss_log_file:
                # epoch
                for echo in range(self.epoch_max):
                    # batch
                    for batch_index in range(self.batch_num):
                        self.global_step += 1
                        state_batch, pi_batch, z_batch = self.get_next_batch(
                            batch_index, self.batch_size)
                        p_loss, v_loss = nn.train(
                            state_batch, pi_batch, z_batch)
                        loss_log_file.write("{0},{1},{2}\n".format(
                            self.global_step, p_loss, v_loss))
                    log("training echo:", echo, "global step:", self.global_step)
                    saver.save(session, config.checkpoint_path +
                               "v{0:03d}".format(self.version), global_step=self.global_step)
            self.clear()
            log("training session end.")

    def load_data(self):
        npz_file_names = get_npz_file_names()
        if len(npz_file_names) == 0:
            self.data_len = 0
            return self.data_len

        self.version = int(npz_file_names[0].split('_')[1]) + 1

        for npz_file_name in npz_file_names:
            data = np.load(config.data_path + npz_file_name,
                           allow_pickle=True)['data']
            data_len = int(len(data) / 3)
            _state_data = np.zeros(
                (data_len, config.N_data, config.N_data, config.history_num * 2 + 1), dtype=float)
            _pi_data = np.zeros(
                (data_len, config.all_moves_num), dtype=float)
            _z_data = np.zeros((data_len, 1), dtype=float)
            for i in range(data_len):
                _state_data[i] = data[3 * i]
                _pi_data[i] = data[3 * i + 1]
                _z_data[i] = data[3 * i + 2]
            self.state_data = np.concatenate((self.state_data, _state_data))
            self.pi_data = np.concatenate((self.pi_data, _pi_data))
            self.z_data = np.concatenate((self.z_data, _z_data))

        self.data_len = len(self.state_data)
        return self.data_len

    def get_next_batch(self, index, size):
        start = index * size
        end = (index + 1) * size
        if start >= self.data_len:
            start = self.data_len - size
        if end > self.data_len:
            end = self.data_len
        return self.state_data[start:end], self.pi_data[start:end], self.z_data[start:end]

    def clear(self):
        npz_file_names = get_npz_file_names()
        for file_name in npz_file_names:
            os.rename(config.data_path + file_name,
                      config.archives_path + file_name)
        log("all npz files archived.")

    def restore(self, session, saver):
        checkpoint_name = restore_from_last_checkpoint(session, saver)
        if checkpoint_name:
            self.global_step = int(checkpoint_name.split('-')[-1])


def pick_move_probabilistically(pi):
    r = random.random()
    s = 0
    for move in range(len(pi)):
        s += pi[move]
        if s >= r:
            return move
    return np.argmax(pi)


def pick_move_greedily(pi):
    return np.argmax(pi)


def get_temperature(moves_num):
    if moves_num <= 5:
        return 1

    # else:
    #     return 0.95 ** (moves_num - 50)

    else:
        temperature_min = 0.1
        temperature = 0.95 ** (moves_num - 5)
        return temperature if temperature > temperature_min else temperature_min


def validate(move):
    if not (isinstance(move, int) or isinstance(move, np.int64)) or not (0 <= move < config.N_data ** 2 or move == config.pass_move):
        raise ValueError("move must be integer from [0, {}] or {}, got {}".format(
            config.all_moves_num-1, config.pass_move, move))


def make_move(node, move):
    validate(move)
    if move not in node.child_nodes:
        node = tree.Node(node, move, -node.player)
    else:
        node = node.child_nodes[move]
    node.is_search_root = True
    node.parent.child_nodes.clear()
    node.parent.is_search_root = False
    return node


def get_winner(node):
    blue_box_num = node.board.count_score(config.blue)
    red_box_num = node.board.count_score(config.red)

    if blue_box_num > red_box_num:
        print("blue wins.")
        return -1

    elif blue_box_num < red_box_num:
        print("red wins.")
        return 1

    else:
        print("draw.")
        return 0


def restore_from_last_checkpoint(session, saver):
    checkpoint = tf.compat.v1.train.latest_checkpoint(config.checkpoint_path)
    if checkpoint:
        saver.restore(session, checkpoint)
        log("restored from last checkpoint.", checkpoint)
        return checkpoint.split('/')[-1]
    else:
        session.run(tf.compat.v1.global_variables_initializer())
        log("checkpoint not found.")
        return None


def get_npz_file_names():
    npz_file_names = []
    walk = os.walk(config.data_path)
    for dpath, _, fnames in walk:
        if dpath == config.data_path:
            for fname in fnames:
                if fname.split('.')[-1] == "npz":
                    npz_file_names.append(fname)
    return npz_file_names


def self_play_woker(worker_id):
    try:
        game = SelfPlayGame(worker_id)
        game.start()
    except Exception as ex:
        traceback.print_exc()


def train_woker():
    try:
        train = Train()
        train.start()
    except Exception as ex:
        traceback.print_exc()


def learning_loop(self_play_wokers_num=config.self_play_wokers_num, learning_loop_echo_max=config.learning_loop_echo_max):

    for i in range(learning_loop_echo_max):
        # multi threads
        # create training data
        pool = Pool(self_play_wokers_num)
        for i in range(self_play_wokers_num):
            pool.apply_async(self_play_woker, (i,))
        pool.close()
        pool.join()

        # train model
        process = Process(target=train_woker)
        process.start()
        process.join()


def play_game(player, color):
    moves_num = 0
    mcts_batch = None
    current_node = tree.Node(tree.FakeNode(), 0, config.blue, board.Board())
    current_node.is_game_root = True
    current_node.is_search_root = True

    def make_move_with_gui(current_node, move):
        current_node = make_move(current_node, move)
        gui.print_node(current_node)
        return current_node

    with tf.compat.v1.Session() as session:
        saver = tf.compat.v1.train.Saver()
        restore_from_last_checkpoint(session, saver)
        nn = net.NN(session)
        mcts_batch = tree.MCTS_Batch(nn)
        moves_num = 0

        gui.print_node(current_node, True)

        while True:
            gc.collect()
            moves_num += 1

            # zero is thinking
            if current_node.player == color:
                pi = mcts_batch.alpha(
                    [current_node], get_temperature(moves_num))[0]
                zero_move = pick_move_greedily(pi)
                current_node = make_move_with_gui(current_node, zero_move)

            # player is thinking
            else:
                mcts_batch.alpha([current_node], get_temperature(moves_num))[0]
                player_move = player.make_move(current_node)
                print("player move: {}".format(player_move))
                current_node = make_move_with_gui(current_node, player_move)

            if current_node.is_terminal:
                break

        # who is the winner
        return get_winner(current_node)


def play_with_human(color):
    _ = play_game(api.HumanPlayer(), color)


def play_with_random_bot(round):
    winner_log = [0, 0, 0]

    for i in range(round):
        win = play_game(api.RandomPlayer(), config.blue)
        if win == config.blue:
            winner_log[0] += 1

        elif win == config.red:
            winner_log[1] += 1

        else:
            winner_log[2] += 1

    for i in range(round):
        win = play_game(api.RandomPlayer(), config.red)
        if win == config.red:
            winner_log[0] += 1

        elif win == config.blue:
            winner_log[1] += 1

        else:
            winner_log[2] += 1

    print(f'Alpha Zero: {winner_log[0]}')
    print(f'Random Bot: {winner_log[1]}')
    print(f'Draw: {winner_log[2]}')

    winning_rate = float(winner_log[0]/(round*2)*100)
    print(f'Winning Rate: {winning_rate}%')


def play_with_greedy_bot(round):
    winner_log = [0, 0, 0]

    for i in range(round):
        win = play_game(api.GreedyPlayer(), config.blue)
        if win == config.blue:
            winner_log[0] += 1

        elif win == config.red:
            winner_log[1] += 1

        else:
            winner_log[2] += 1

    for i in range(round):
        win = play_game(api.GreedyPlayer(), config.red)
        if win == config.red:
            winner_log[0] += 1

        elif win == config.blue:
            winner_log[1] += 1

        else:
            winner_log[2] += 1

    print(f'Alpha Zero: {winner_log[0]}')
    print(f'Greedy Bot: {winner_log[1]}')
    print(f'Draw: {winner_log[2]}')

    winning_rate = float(winner_log[0]/(round*2)*100)
    print(f'Winning Rate: {winning_rate}%')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--learning-loop",
                        help='start a learning loop from the latest model, or a new random model if there is no any model', action="store_true")
    parser.add_argument("-m", "--play-with-human",
                        help='play with you on the command line', action="store_true")
    parser.add_argument("-r", "--play-with-random-bot",
                        help='play with random bot', action="store_true")
    parser.add_argument("-g", "--play-with-greedy-bot",
                        help='play with random bot', action="store_true")
    args = parser.parse_args()

    if args.learning_loop:
        learning_loop()
        # os.system('shutdown /s /t 1')

    elif args.play_with_human:
        color = 0
        while color != config.blue and color != config.red:
            color = int(input('Decide model is first move or second move: '))

        play_with_human(color)

    elif args.play_with_random_bot:
        # play_with_random_bot(int(input('Decide the round will play: ')))
        play_with_random_bot(10)

    elif args.play_with_greedy_bot:
        # play_with_greedy_bot(int(input('Decide the round will play: ')))
        play_with_greedy_bot(10)

    else:
        learning_loop()
        # os.system('shutdown /s /t 1')
