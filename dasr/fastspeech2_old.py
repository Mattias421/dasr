import subprocess
import sys
from dora import get_xp


def train_fastspeech2(cfg):

    xp = get_xp()

    data_root = cfg.data.root

    fs2_root = cfg.augment.fs2_root

    cfg.augment.preprocess.dataset = cfg.data.name
    cfg.augment.preprocess.path.corpus_path = data_root
    cfg.augment.preprocess.path.preprocessed_path = xp.folder / 'preprocessed'

    subprocess.run([f'{fs2_root}/montreal-forced-aligner/bin/mfa_align', 
	            f'{data_root}',
		    f'{cfg.augment.preprocess.path.lexicon_path}',
		    'english',
		    f'{cfg.augment.preprocess.path.preprocessed_path}'
                    ])
