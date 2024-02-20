from dora import get_xp, XP

from .tts.FastSpeech2.preprocessor.preprocessor import Preprocessor
from .tts.FastSpeech2.prepare_align import main as prepare_align 

import logging
logger = logging.getLogger()

def check_dir(path):
    if not path.exists():
        path.mkdir(parents=True)
        logger.info(f'Creating {path}')
        return True
    else:
        logger.info(f'{path} exists')
        return False

def make_preprocess_cfg(cfg, xp):
    raw_path = xp.folder / 'raw_data'
    preprocessed_path = xp.folder / 'preprocessed_data'

    cfg.augment = cfg.augment.fastspeech2

    cfg.augment.preprocess.dataset = cfg.data.name
    cfg.augment.preprocess.path.raw_path = raw_path
    cfg.augment.preprocess.path.preprocessed_path = preprocessed_path 

    return cfg.augment.preprocess

def train_fastspeech2(cfg):

    xp = get_xp()
    logger.info(f'Training fastspeech2 in {xp.sig}')
    print(xp.cfg)
    print(xp.argv)
    print(xp.delta)

    cfg_preproc = cfg
    cfg_preproc.task = 'prepare_data'
    args_preproc = [s.replace('task=train_augment', 'task=prepare_data') for s in xp.argv] 
    delta_preproc = [(d[0], 'prepare_data' if d[0] == 'task' else d[1]) for d in xp.delta]

    xp_pre_proc = XP(dora=xp.dora, cfg=cfg_preproc, argv=args_preproc, delta=delta_preproc, sig=None) 
    logger.info(f'Getting preprocessing data from {xp_pre_proc.sig}')
    preproc_cfg = make_preprocess_cfg(xp.cfg, xp)

    print(preproc_cfg)


    return 'In progress'


def prepare_fastspeech2(cfg):
    xp = get_xp()


    raw_path = xp.folder / 'raw_data'
    preprocessed_path = xp.folder / 'preprocessed_data'
    do_prepare = check_dir(raw_path)

    cfg.augment = cfg.augment.fastspeech2

    cfg.augment.preprocess.dataset = cfg.data.name
    cfg.augment.preprocess.path.raw_path = raw_path
    cfg.augment.preprocess.path.preprocessed_path = preprocessed_path 

    if do_prepare:
        logger.info(f'Prepare align to {cfg.augment.preprocess.path.raw_path}')
        logger.info('Warning, assume corpus contains text grids')
        prepare_align(cfg) 
        logger.info('Prepare complete')
    else:
        logger.info(f'Found prepared data at {raw_path}')

    do_preprocess = check_dir(preprocessed_path)

    if do_preprocess:
        logger.info(f'Preprocessing {cfg.data.name} in xp {xp.sig}')

        preprocessor = Preprocessor(cfg.augment.preprocess)
        preprocessor.build_from_path()
        logger.info('Preprocess complete')
    else:
        logger.info(f'Found preporcessed data at {preprocessed_path}')
        # DEBUG force preprocess
        logger.info(f'Preprocessing {cfg.data.name} in xp {xp.sig}')

        preprocessor = Preprocessor(cfg.augment.preprocess)
        preprocessor.build_from_path()
        logger.info('Preprocess complete')
