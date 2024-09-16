import yaml
import argparse
from utils.trainer import Trainer
from utils.func import random_seed


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str,
                        default='configs/DSEC_Semantic.yaml',
                        help='Configuration file to use')
    args = parser.parse_args()

    with open(args.cfg) as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)

    # Random seed
    random_seed(cfg['SEED_NUM'])

    # Initialize the trainer
    trainer = Trainer(cfg)
    trainer.train()
