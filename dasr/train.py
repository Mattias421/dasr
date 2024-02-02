from dora import get_xp, hydra_main
from .specaugment import spec_augment_data
from .fastspeech2 import train_fastspeech2

def augment(cfg):
    if cfg.augment.name == 'spec-augment':
        spec_augment_data(cfg)



def train_augment(cfg):
    if cfg.augment.name == 'fastspeech2':
        train_fastspeech2(cfg)


@hydra_main(version_base=None, config_path='../conf', config_name='config')
def main(cfg):
    if cfg.task == 'augment':
        augment(cfg)
    elif cfg.task == 'train_augment':
        train_augment(cfg)