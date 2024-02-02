import torch
import torchaudio
import torchaudio.transforms as T

from dora import get_xp
import os
from pathlib import Path
from typing import List

import pandas as pd

from .evaluate_tts import compute_metrics

import logging
logger = logging.getLogger()



class SpecAugment():
    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.stretch = T.TimeStretch()
        self.time_masking = T.TimeMasking(time_mask_param=cfg.time_mask_param)
        self.freq_masking = T.FrequencyMasking(freq_mask_param=cfg.freq_mask_param)
        self.ispec = T.InverseSpectrogram()
    


    def _get_sample(self, path, resample=None):
        effects = [["remix", "1"]]
        if resample:
            effects.extend(
                [
                    ["lowpass", f"{resample // 2}"],
                    ["rate", f"{resample}"],
                ]
            )
        return torchaudio.sox_effects.apply_effects_file(path, effects=effects)



    def get_speech_sample(self, path, resample=None):
        return self._get_sample(path, resample=resample)



    def get_spectrogram(
        self,
        path,
        n_fft=400,
        win_len=None,
        hop_len=None,
        power=2.0,
    ):
        waveform, sr = self.get_speech_sample(path)
        spectrogram = T.Spectrogram(
            n_fft=n_fft,
            win_length=win_len,
            hop_length=hop_len,
            center=True,
            pad_mode="reflect",
            power=power,
        )
        return spectrogram(waveform), sr



    def __call__(self, source_path, target_path):

        xp = get_xp()

        spec, sr = self.get_spectrogram(source_path,
                                    n_fft=self.cfg.n_fft,
                                    win_len=self.cfg.win_len,
                                    hop_len=self.cfg.hop_len,
                                    power=self.cfg.power)

        for i, rate in enumerate(self.cfg.time_stretch_rates):
            new_spec = self.stretch(spec, overriding_rate=rate)
            waveform = self.ispec(new_spec.to(torch.complex64))

            out_path = f'{target_path}_stretch_{i}.wav'
            torchaudio.save(out_path, waveform, sample_rate=sr)
            result = {'source':source_path,
                      'target':out_path,
                      }
            metrics = compute_metrics(source_path, out_path)
            result.update(metrics)
            xp.link.push_metrics(result)

        time_masked = self.time_masking(spec)
        freq_masked = self.freq_masking(spec)

        time_masked_path = f'{target_path}_time_mask.wav'
        time_masked_waveform = self.ispec(time_masked.to(torch.complex64))
        torchaudio.save(time_masked_path, time_masked_waveform, sample_rate=sr)
        time_masked_result = {'source': source_path, 'target': time_masked_path}
        time_masked_metrics = compute_metrics(source_path, time_masked_path)
        time_masked_result.update(time_masked_metrics)
        xp.link.push_metrics(time_masked_result)

        freq_masked_path = f'{target_path}_freq_mask.wav'
        freq_masked_waveform = self.ispec(freq_masked.to(torch.complex64))
        torchaudio.save(freq_masked_path, freq_masked_waveform, sample_rate=sr)
        freq_masked_result = {'source': source_path, 'target': freq_masked_path}
        freq_masked_metrics = compute_metrics(source_path, freq_masked_path)
        freq_masked_result.update(freq_masked_metrics)
        xp.link.push_metrics(freq_masked_result)



def spec_augment_data(cfg):

    xp = get_xp()
    synth_root = xp.folder / 'synth_data'
    p = Path(synth_root)
    p.mkdir(parents=True, exist_ok=True)

    model = SpecAugment(cfg.augment)

    manifest_df = pd.read_csv(cfg.data.manifest)

    manifest_list = manifest_df['wav'].to_list() 

    for wav in manifest_list:
        source_path = Path(wav)
        speaker = [parent for parent in source_path.parents[-3:]].join('/')
        file_name = source_path.stem

        spk_synth_root = synth_root / speaker

        spk_synth_root.mkdir(parents=True, exist_ok=True)

        target_path = spk_synth_root / file_name

        logger.info(f'Augmenting {target_path}')

        model(source_path, target_path)








