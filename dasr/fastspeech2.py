import subprocess
import sys


def train_fastspeech2(cfg):

    data_root = cfg.data.root

    fs2_root = cfg.augment.fs2_root

    subprocess.run(['python', 
                    f'{fs2_root}train_internal_alignment.py', 
                    f'{fs2_root}hparams/train_internal_alignment.yaml',
                    f'--data_folder={data_root}',
                    ])