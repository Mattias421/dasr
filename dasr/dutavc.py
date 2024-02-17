from dora import get_xp
from .tts.AtyTTS.AtyTTS.DuTaVC.prepare_data import get_mel_atypical, get_embed

def prepare_dutavc(cfg):

    xp = get_xp()
    root = xp.folder

    new_data_root = root / 'data'
    new_data_root.mkdir()
    mel_root = new_data_root / 'mels' 
    embed_root = new_data_root / 'embeds' 

    speakers = []


