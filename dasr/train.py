from dora import get_xp, hydra_main
from .specaugment import spec_augment_data
from .fastspeech2 import train_fastspeech2, prepare_fastspeech2, debug
# from .dutavc import prepare_dutavc

import logging
logger = logging.getLogger()

    

def augment(cfg):
    if cfg.augment.name == 'spec-augment':
        spec_augment_data(cfg)

def train_augment(cfg):
    if cfg.augment.name == 'fastspeech2':
        train_fastspeech2(cfg)

def prepare_data(cfg):
    if cfg.augment.name == 'dutavc':
        # prepare_dutavc(cfg)
        logger.info('not implemented yet')
    elif cfg.augment.name == 'fastspeech2':
        prepare_fastspeech2(cfg)

@hydra_main(version_base=None, config_path='../conf', config_name='config')
def main(cfg):
    if cfg.task == 'augment':
        augment(cfg)
    elif cfg.task == 'train_augment':
        train_augment(cfg)
    elif cfg.task == 'prepare_data':
        prepare_data(cfg)
    elif cfg.task == 'debug':
        debug(cfg)
