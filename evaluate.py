import yaml
import argparse
from utils.evaluator import Evaluator
from utils.func import random_seed


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str,
                        default='configs/DDD17.yaml',
                        help='Configuration file to use')
    args = parser.parse_args()

    with open(args.cfg) as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)

    # Random seed
    random_seed(cfg['SEED_NUM'])

    # Initialize the trainer
    evaluator = Evaluator(cfg)
    evaluator.eval()
