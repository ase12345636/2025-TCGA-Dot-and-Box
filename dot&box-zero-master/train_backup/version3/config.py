import os


# board config
N = 5
N_data = 2*N-1
board_length = N_data ** 2
pass_move = N_data ** 2
all_moves_num = N_data ** 2+1
blue = -1
red = 1
vertex = 5
legal_edge = 0
empty_box = 8
blue_edge = -1
red_edge = 1
blue_box = 7
red_box = 9


# mcts config
c_puct = 1
simulations_num = 400
noise_alpha = 0.5
noise_weight = 0.25


# nn config
history_num = 4
residual_blocks_num = 16
momentum = 0.9
l2_weight = 1e-4
learning_rate = 1e-2


# learning config
self_play_wokers_num = 4
self_play_woker_gpu_memory_fraction = 0.14
self_play_batch_size = 8
self_play_echo_max = 16
train_batch_size = 1024
train_epoch_max = 256
train_checkpoint_max_to_keep = 1
learning_loop_echo_max = 3000


# path config
checkpoint_path = "./checkpoint/"
data_path = "./data/"
archives_path = "./data/archives/"
log_path = "./log/"
if os.path.exists(data_path) is not True:
    os.mkdir(data_path)
if os.path.exists(archives_path) is not True:
    os.mkdir(archives_path)
if os.path.exists(log_path) is not True:
    os.mkdir(log_path)
file_log_path = ".\\log\\loss_log.csv"
file_image_path = ".\\log\\loss_log.png"


# image config
green_color = [29 / 255.0, 177 / 255.0, 0]
blue_color = [0, 118 / 255.0, 186 / 255.0]
yellow_color = [255 / 255.0, 106 / 255.0, 0]
moving_average_width = 4000
