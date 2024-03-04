from dora import get_xp, XP

from .tts.FastSpeech2.preprocessor.preprocessor import Preprocessor
from .tts.FastSpeech2.prepare_align import main as prepare_align 
from .tts.FastSpeech2.train import main as train
from .tts.FastSpeech2.synthesize import synthesize
from .tts.FastSpeech2.utils.model import get_model, get_vocoder

from pathlib import Path
import torch

from types import SimpleNamespace

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


def debug(cfg):
    xp = get_xp()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_cfg = cfg.augment.fastspeech2.model
    train_cfg = cfg.augment.fastspeech2.train
    preprocess_cfg = cfg.augment.fastspeech2.preprocess

    xps_path = Path('/fastdata/acq22mc/exp/dasr/outputs/xps/')
    preprocess_path = xps_path / cfg.preprocess_xp
    preprocess_cfg.path.raw_path = preprocess_path / 'raw_data'
    preprocess_cfg.path.preprocessed_path = preprocess_path / 'preprocessed_data'

    train_xp = cfg.train_xp
    train_xp_path = xps_path / train_xp

    train_cfg.path.ckpt_path = train_xp_path / 'ckpt'
    train_cfg.path.log_path = train_xp_path / 'log'
    train_cfg.path.result_path = train_xp_path / 'result'

    configs = (preprocess_cfg, model_cfg, train_cfg)
    model = get_model(200000, configs, device, train=False)



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
    preproc_cfg = make_preprocess_cfg(xp.cfg, xp_pre_proc)

    print(preproc_cfg)

    model_cfg = cfg.augment.fastspeech2.model
    train_cfg = cfg.augment.fastspeech2.train
    train_cfg.path.ckpt_path = xp.folder / 'ckpt'
    train_cfg.path.log_path = xp.folder / 'log'
    train_cfg.path.result_path = xp.folder / 'result'

    for path in train_cfg.path:
        check_dir(train_cfg.path[path])

    args = SimpleNamespace(restore_step=0)
    configs = (preproc_cfg, model_cfg, train_cfg)

    train(args, configs)


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
        # logger.info(f'Prepare align to {cfg.augment.preprocess.path.raw_path}')
        # logger.info('Warning, assume corpus contains text grids')
        # prepare_align(cfg) 
        # logger.info('Prepare complete')

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


