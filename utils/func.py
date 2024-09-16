import torch
import os
import random
import numpy as np


class IOStream():
    def __init__(self, path):
        self.f = open(path, 'a')

    def cprint(self, text):
        # print(text)
        self.f.write(text+'\n')
        self.f.flush()

    def close(self):
        self.f.close()


def random_seed(num):
    np.random.seed(num)
    random.seed(num)
    os.environ['PYTHONHASHSEED'] = str(num)
    torch.manual_seed(num)
    torch.cuda.manual_seed(num)
    torch.cuda.manual_seed_all(num)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
