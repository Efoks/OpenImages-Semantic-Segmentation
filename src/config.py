import os
import torch

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_DIR, 'data')
EXPERIMENT_DIR = os.path.join(PROJECT_DIR, 'experiments')
if not os.path.exists(EXPERIMENT_DIR):
    os.makedirs(EXPERIMENT_DIR)

CLASS_DICT = {'pizza': 1,
              'taxi': 2,
              'dog': 3}
DEVICE = ("cuda" if torch.cuda.is_available() else "cpu")

