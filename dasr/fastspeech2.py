from dora import get_xp

from .tts.FastSpeech2.preprocessor.preprocessor import Preprocessor

import logging
logger = logging.getLogger()

def train_fastspeech2(cfg):
    return 'In progress'


def prepare_fastspeech2(cfg):
    xp = get_xp()

    data_root = cfg.data.root

    cfg.augment = cfg.augment.fastspeech2

    cfg.augment.preprocess.dataset = cfg.data.name
    cfg.augment.preprocess.path.raw_path = xp.folder / 'raw_data'
    cfg.augment.preprocess.path.root = cfg.data.root
    cfg.augment.preprocess.path.manifest = data_manifest # TODO: put manifest in folde
    cfg.augment.preprocess.path.preprocessed_path = xp.folder / 'preprocessed_data'

    logger.info(f'Preprocessing {cfg.data.name} in xp {xp.sig}')

    preprocessor = Preprocessor(cfg.augment.preprocess)
    preprocessor.build_from_path()
