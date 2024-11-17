import os

import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_path = 'D:/train_hr'
val_path = 'D:/Kodak'
batch_size = 32
num_epochs = 50
learning_rate = 0.0002
save_interval = 1000
example_save_dir = 'examples'
checkpoints_dir = 'checkpoints'
plot_save_dir = 'plots'
