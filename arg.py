m = 5
n = 5
history_move = 8

# batch_size = (m * (n - 1)) * 2
# batch_size = (2*m-1)*(2*n-1)
# batch_size = 1
batch_size = 32

'''
type 0: normal;             input shape: m * n
type 1: history_image;      input shape: m * n * total_move
type 2: history_sequence;   input shape: total_move * (m * n)
type 3: history_video;      input shape: total_move * m * n * 1
'''
args_Res = {
    'num_of_generate_data_for_train': 400,
    'epochs': 50,
    'batch_size': batch_size,
    'verbose': True,
    'type': 0,
    'train': True,
    'load_model_name': None
}

args_ValueNet = {
    'num_of_generate_data_for_train': 400,
    'epochs': 30,
    'batch_size': batch_size*2,
    'verbose': True,
    'type': 0,
    'train': True,
    'load_model_name': None
}
