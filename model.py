import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable

from STT import create_hparams, STT
from TTS import TTS

class Model(nn.Module):
    def __init__(self, in_channels=80, 
                 enc_channels=128, enc_hidden_channels=256, 
                 n_blocks=3, dec_hidden_channels=256, z_channels=32, std=False):
        super().__init__()
        
        hparams = create_hparams()
        self.stt = STT(hparams)
        self.tts = TTS(in_channels, enc_channels, enc_hidden_channels, n_blocks, dec_hidden_channels, z_channels, std)
        
    def forward(self, batch, beta=1.0):
        stt_outputs = self.stt(batch)
        tts_outputs = self.tts(batch, stt_outputs, beta)
        
        return stt_outputs, tts_outputs
    
    def inference(self, c, length, alignments, temperature=1.0):
        y = self.tts.inference(c, length, alignments, temperature)
        
        return y
    
def to_cuda(batch):
    batch['text'] = batch['text'].cuda()
    batch['text_lengths'] = batch['text_lengths'].cuda()
    batch['mels'] = batch['mels'].cuda()
    batch['mel_lengths'] = batch['mel_lengths'].cuda()
    
    return batch