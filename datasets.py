import librosa
import librosa.display
import numpy as np
import torch
import os
from os import listdir
from os.path import isfile, join
import ntpath
from torch.utils.data import DataLoader
import warnings

def logmelfilterbank(audio,
                     sampling_rate,
                     fft_size=1024,
                     hop_size=256,
                     win_length=None,
                     window="hann",
                     num_mels=80,
                     fmin=None,
                     fmax=None,
                     eps=1e-10,
                     ):
    """Compute log-Mel filterbank feature.

    Args:
        audio (ndarray): Audio signal (T,).
        sampling_rate (int): Sampling rate.
        fft_size (int): FFT size.
        hop_size (int): Hop size.
        win_length (int): Window length. If set to None, it will be the same as fft_size.
        window (str): Window function type.
        num_mels (int): Number of mel basis.
        fmin (int): Minimum frequency in mel basis calculation.
        fmax (int): Maximum frequency in mel basis calculation.
        eps (float): Epsilon value to avoid inf in log calculation.

    Returns:
        ndarray: Log Mel filterbank feature (#frames, num_mels).

    """
    # get amplitude spectrogram
    x_stft = librosa.stft(audio, n_fft=fft_size, hop_length=hop_size,
                          win_length=win_length, window=window, pad_mode="reflect")
    spc = np.abs(x_stft).T  # (#frames, #bins)

    # get mel basis
    fmin = 0 if fmin is None else fmin
    fmax = sampling_rate / 2 if fmax is None else fmax
    mel_basis = librosa.filters.mel(sampling_rate, fft_size, num_mels, fmin, fmax)
    mel = np.dot(spc, mel_basis.T)
    return np.log10(np.maximum(1e-5, mel)).T

class LJDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir):
        self.data_files = self._get_data_files(root_dir)
        self.mel_matrix = librosa.filters.mel(sr=22050, n_fft=1024, n_mels=80)
        
    def _get_data_files(self, LJSpeech_dir):
        metadata = LJSpeech_dir + 'metadata.csv'

        data_files = []
        with open(metadata, 'r') as f:
            l = f.readline().strip()
            while l:
                l = l.split('|')
                wav_file = LJSpeech_dir + 'wavs/' + l[0] + '.wav'
                text = l[2]
                data_files.append((wav_file, text))
                l = f.readline().strip()

        return data_files    
    
    def _get_mel(self, data_file):
        wav, _ = librosa.core.load(data_file, sr=22050)
        wav, _ = librosa.effects.trim(wav, top_db=40)
        
        with warnings.catch_warnings():
            mel = logmelfilterbank(wav, sampling_rate=22050, fft_size=1024, hop_size=256, fmin=80, fmax=7600)

        return mel
    
    def _get_utf8_values(self, text):
        #text = g2p(text)
        text_utf = text.encode()
        ts = [0]
        for t in text_utf:
            ts.append(t)
        utf8_values = np.array(ts)
        ts.append(0)
        
        return utf8_values
        
        
    def __getitem__(self, index):
        mel = self._get_mel(self.data_files[index][0])
        text = self._get_utf8_values(self.data_files[index][1])
        
        return torch.LongTensor(text), torch.FloatTensor(mel)
        
    def __len__(self):
        return len(self.data_files)
    
class TextMelCollate():
    
    """ Zero-pads model inputs and targets based on number of frames per setep
    """
    def __init__(self):
        pass
        
    def __call__(self, batch):
        """Collate's training batch from normalized text and mel-spectrogram
        PARAMS
        ------
        batch: [text_normalized, mel_normalized]
        """
        
        outputs = {}
        # Right zero-pad all one-hot text sequences to max input length
        input_lengths, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([len(x[0]) for x in batch]),
            dim=0, descending=True)
        max_input_len = input_lengths[0]

        text_padded = torch.LongTensor(len(batch), max_input_len)
        text_padded.zero_()
        for i in range(len(ids_sorted_decreasing)):
            text = batch[ids_sorted_decreasing[i]][0]
            text_padded[i, :text.size(0)] = text
        outputs['text'] = text_padded
        outputs['text_lengths'] = input_lengths
            
        # include mel padded and gate padded
        num_mels = batch[0][1].size(0)    
        max_target_len = max([x[1].shape[1] for x in batch])
        #max_target_len = 1024
        mel_padded = torch.FloatTensor(len(batch), num_mels, max_target_len)
        mel_padded.fill_(-5)
        output_lengths = torch.LongTensor(len(batch))
        for i in range(len(ids_sorted_decreasing)):
            mel = batch[ids_sorted_decreasing[i]][1]
            mel_padded[i, :, :mel.size(1)] = mel
            output_lengths[i] = mel.size(1)
            
        outputs['mels'] = mel_padded
        outputs['mel_lengths'] = output_lengths

        return outputs