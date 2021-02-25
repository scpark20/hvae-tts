import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable

class Conv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=True, zero_weight=False):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        if zero_weight:
            self.conv.weight.data.zero_()
        else:
            self.conv.weight.data.normal_(0, 0.02)

    def forward(self, x):
        # x : (b, c, t)
        y = self.conv(x)
        
        return y
    
class ConvTranspose1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=True):
        super().__init__()
        self.conv = nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        self.conv.weight.data.normal_(0, 0.02)
        
    def forward(self, x):
        y = self.conv(x)
        
        return y
        
class TTSTextEncoder(nn.Module):
    def __init__(self, hidden_channels=128):
        super().__init__()

        convolutions = []
        for _ in range(3):
            conv_layer = nn.Sequential(
                Conv1d(hidden_channels, hidden_channels, kernel_size=5, padding=2),
                nn.BatchNorm1d(hidden_channels))
            convolutions.append(conv_layer)
        self.convolutions = nn.ModuleList(convolutions)
        self.lstm = nn.LSTM(hidden_channels, hidden_channels//2, num_layers=1, batch_first=True, bidirectional=True)
        self.gmm = nn.Linear(hidden_channels, 2)
        self.gmm.weight.data.zero_()

    def forward(self, x):
        for conv in self.convolutions:
            x = F.dropout(F.relu(conv(x)), 0.5, self.training)
            
        x = x.transpose(1, 2)
        outputs, _ = self.lstm(x)
        gmm_params = self.gmm(outputs)

        return gmm_params
    
class ConvBlock(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, last_zero=False):
        super().__init__()
        
        self.conv = nn.Sequential(Conv1d(in_channels, hidden_channels, kernel_size=1),
                                  nn.BatchNorm1d(hidden_channels),
                                  nn.GELU(),
                                  Conv1d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
                                  nn.BatchNorm1d(hidden_channels),
                                  nn.GELU(),
                                  Conv1d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
                                  nn.BatchNorm1d(hidden_channels),
                                  nn.GELU(),
                                  Conv1d(hidden_channels, out_channels, kernel_size=1, zero_weight=last_zero))
        
    def forward(self, x):
        y = self.conv(x)
        
        return y
    
class TTSMelEncoderBlocks(nn.Module):
    def __init__(self, n_blocks, hidden_channels, out_channels):
        super().__init__()
        self.convs = nn.ModuleList([ConvBlock(hidden_channels, hidden_channels, hidden_channels) for _ in range(n_blocks)])
        self.outs = nn.ModuleList([Conv1d(hidden_channels, out_channels) for _ in range(n_blocks)])
        
    def forward(self, x):
        xs = []
        for conv, out in zip(self.convs, self.outs):
            x = x + conv(x)
            xs.append(out(x))
        
        return x, list(reversed(xs))
        
class TTSMelEncoder(nn.Module):
    def __init__(self, n_blocks, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.in_layer = Conv1d(in_channels, hidden_channels)
        self.encoder_blocks = nn.ModuleList([TTSMelEncoderBlocks(n_blocks, hidden_channels, out_channels),
                                             TTSMelEncoderBlocks(n_blocks, hidden_channels, out_channels),
                                             TTSMelEncoderBlocks(n_blocks, hidden_channels, out_channels),
                                             TTSMelEncoderBlocks(n_blocks, hidden_channels, out_channels),
                                             TTSMelEncoderBlocks(n_blocks, hidden_channels, out_channels)])
        self.downs = nn.ModuleList([Conv1d(hidden_channels, hidden_channels, kernel_size=2, stride=2),
                                    Conv1d(hidden_channels, hidden_channels, kernel_size=2, stride=2),
                                    Conv1d(hidden_channels, hidden_channels, kernel_size=2, stride=2),
                                    Conv1d(hidden_channels, hidden_channels, kernel_size=2, stride=2),
                                    nn.Identity()])
                
    def forward(self, x):
        # x : (b, 80, t)
        
        x = self.in_layer(x)
        
        xs_list = []
        for block, down in zip(self.encoder_blocks, self.downs):
            x, xs = block(x)
            xs_list.append(xs)
            x = down(x)
            
        return list(reversed(xs_list))
            
class TTSMelDecoderBlock(nn.Module):
    def __init__(self, in_channels, enc_channels, z_channels, std):
        super().__init__()
        self.in_channels = in_channels
        self.z_channels = z_channels
        
        self.std = std
        if not std:
            self.p = ConvBlock(in_channels, in_channels, 2*z_channels, last_zero=True)
            
        self.q = ConvBlock(in_channels+enc_channels, in_channels, 2*z_channels, last_zero=True)
        self.latent = Conv1d(z_channels, in_channels)
        self.out = ConvBlock(in_channels, in_channels, in_channels)
        
    def _get_kl_div(self, q_params, p_params):
        if self.std:
            p_mean = 0
            p_logstd = 0
            q_mean = q_params[0]
            q_logstd = q_params[1]
            
            return (p_logstd - q_logstd) + 0.5 * (q_logstd.exp() ** 2 + (q_mean - p_mean) ** 2) / (1 ** 2) - 0.5
        
        else:
            p_mean = p_params[0]
            p_logstd = p_params[1]
            q_mean = q_params[0]
            q_logstd = q_params[1]

            return (p_logstd - q_logstd) + 0.5 * (q_logstd.exp() ** 2 + (q_mean - p_mean) ** 2) / (p_logstd.exp() ** 2) - 0.5
    
    def _sample(self, params, temperature=1.0):
        mean = params[0]
        logstd = params[1]
        sample = mean + mean.new(mean.shape).normal_() * logstd.exp() * temperature
        
        return sample
    
    def _sample_from_p(self, tensor, shape, temperature=1.0):
        sample = tensor.new(*shape).normal_() * temperature
        
        return sample
    
    def forward(self, x, src, c):
        x2 = x + c
        
        p_params = None
        if not self.std:
            p_params = self.p(x2).split(self.z_channels, dim=1)
         
        q_params = self.q(torch.cat([x2, src], dim=1)).split(self.z_channels, dim=1)
        kl_div = self._get_kl_div(q_params, p_params)
        z = self._sample(q_params)
        l = self.latent(z)
        y = x + self.out(x2 + l)
        
        return y, kl_div
    
    def inference(self, x, c, temperature):
        x2 = x + c
        
        if not self.std:
            p_params = self.p(x2).split(self.z_channels, dim=1)
            z = self._sample(p_params, temperature)
        else:
            z = self._sample_from_p(x, (x.size(0), self.z_channels, x.size(2)), temperature)
            
        l = self.latent(z)
        y = x + self.out(x2 + l)
        
        return y
    
class TTSMelDecoderBlocks(nn.Module):
    def __init__(self, n_blocks, hidden_channels, enc_channels, z_channels, std):
        super().__init__()
        self.decoders = nn.ModuleList([TTSMelDecoderBlock(hidden_channels, enc_channels, z_channels, std) \
                                       for _ in range(n_blocks)])
        
    def forward(self, x, srcs, c):
        
        kl_divs = []
        for decoder, src in zip(self.decoders, srcs):
            x, kl_div = decoder(x, src, c)
            kl_divs.append(kl_div)
            
        return x, kl_divs
    
    def inference(self, x, c, temperature):
        
        for decoder in self.decoders:
            x = decoder.inference(x, c, temperature)
            
        return x
        
class TTSMelDecoder(nn.Module):
    def __init__(self, n_blocks, out_channels, hidden_channels, enc_channels, z_channels, std):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.decoders = nn.ModuleList([TTSMelDecoderBlocks(n_blocks, hidden_channels, enc_channels, z_channels, std),
                                       TTSMelDecoderBlocks(n_blocks, hidden_channels, enc_channels, z_channels, std),
                                       TTSMelDecoderBlocks(n_blocks, hidden_channels, enc_channels, z_channels, std),
                                       TTSMelDecoderBlocks(n_blocks, hidden_channels, enc_channels, z_channels, std),
                                       TTSMelDecoderBlocks(n_blocks, hidden_channels, enc_channels, z_channels, std),
                                      ])
        self.ups = nn.ModuleList([nn.Identity(),
                                  ConvTranspose1d(hidden_channels, hidden_channels, kernel_size=2, stride=2),
                                  ConvTranspose1d(hidden_channels, hidden_channels, kernel_size=2, stride=2),
                                  ConvTranspose1d(hidden_channels, hidden_channels, kernel_size=2, stride=2),
                                  ConvTranspose1d(hidden_channels, hidden_channels, kernel_size=2, stride=2),
                                 ])
        self.out = Conv1d(hidden_channels, out_channels, kernel_size=3, padding=1)
        
    def forward(self, srcs, cs):
        b, _, t = srcs[0][0].size()
        x = srcs[0][0].data.new(b, self.hidden_channels, t).zero_()
        kl_divs = []
        for decoder, up, src, c in zip(self.decoders, self.ups, srcs, cs):
            x = up(x)
            x, kl_div = decoder(x, src, c)
            kl_divs.extend(kl_div)
        y = self.out(x)
        
        return y, kl_divs
    
    def inference(self, cs, length, temperature):
        
        x = torch.zeros(cs[0].size(0), self.hidden_channels, length).zero_().cuda()
        for decoder, up, c in zip(self.decoders, self.ups, cs):
            x = up(x)
            x = decoder.inference(x, c, temperature)
        y = self.out(x)
        
        return y
    
class PoolingLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.pooling = nn.Conv1d(in_channels, out_channels, kernel_size=2, stride=2)
        
    def forward(self, x):
        y = self.pooling(x)
        
        return y
        
class Poolings(nn.Module):
    def __init__(self, in_c, n_layers):
        super().__init__()
        self.poolings = nn.ModuleList([PoolingLayer(in_c, in_c),
                                       PoolingLayer(in_c, in_c),
                                       PoolingLayer(in_c, in_c),
                                       PoolingLayer(in_c, in_c)])
        
    def forward(self, x):
        xs = [x]
        for pooling in self.poolings:
            x = pooling(x)
            xs.append(x)
            
        return list(reversed(xs))
       
def get_attention_matrix(gmm_params, mel_length):
    batch, text_length, _ = gmm_params.size()
    
    mean = (gmm_params[:, :, 0:1].exp() * 8).cumsum(dim=1)
    scale = gmm_params[:, :, 1:2].exp() * 8
    Z = torch.sqrt(2 * np.pi * scale ** 2)
    matrix = torch.linspace(0, mel_length-1, mel_length).cuda().repeat(batch, text_length, 1)
    matrix = 1 / Z * torch.exp(-0.5 * (matrix - mean) ** 2 / (scale ** 2))
        
    return matrix

import time

class TTS(nn.Module):
    def __init__(self, in_channels=80, 
                 enc_channels=128, enc_hidden_channels=256, 
                 n_blocks=3, dec_hidden_channels=256, z_channels=32, std=False):
        super().__init__()
        self.unit = 16
        self.embedding = nn.Embedding(256, dec_hidden_channels)
        self.text_encoder = TTSTextEncoder(dec_hidden_channels)
        self.poolings = Poolings(dec_hidden_channels, 4)
        
        self.mel_encoder = TTSMelEncoder(n_blocks=n_blocks,
                                  in_channels=in_channels, 
                                  hidden_channels=enc_hidden_channels, 
                                  out_channels=enc_channels)
        
        self.mel_decoder = TTSMelDecoder(n_blocks=n_blocks, 
                                  out_channels=in_channels, 
                                  hidden_channels=dec_hidden_channels, 
                                  enc_channels=enc_channels, 
                                  z_channels=z_channels, std=std)
        
    def _get_loss(self, src, pred, kl_divs, stt_params, tts_params, beta):
        recon_loss = ((pred - src) ** 2).sum(dim=[1, 2])
        kl_loss = None
        for kl in kl_divs:
            if kl is None:
                continue    
            kl_loss = kl.sum(dim=[1, 2]) if kl_loss is None else kl_loss + kl.sum(dim=[1, 2])  
        if kl_loss is None:
            kl_loss = torch.zeros([1]).cuda()
        dim = src.size(1) * src.size(2)
        loss = (recon_loss + beta * kl_loss).mean() / dim
        loss = loss + nn.MSELoss()(stt_params, tts_params)
        
        recon_loss = recon_loss.mean() / dim
        kl_loss = kl_loss.mean() / dim
        
        return loss, recon_loss, kl_loss     
    
    def _normalize(self, alignments):
        return (alignments + 1e-8).log().softmax(dim=1)
        
    def forward(self, batch, stt_outputs, beta=1.0):
#         torch.Size([4, 80, 635])
#         torch.Size([4, 108])
#         torch.Size([4, 108, 635])

        # (b, 80, t)
        x = batch['mels']
        # (b, l)
        c = batch['text']
        # (b, l, t)
        stt_alignments = self._normalize(stt_outputs['alignments'].detach())
        
        pad_length = ((x.size(2) - 1) // self.unit + 1) * self.unit - x.size(2)
        x = F.pad(x, (0, pad_length))
        stt_alignments = F.pad(stt_alignments, (0, pad_length))
        
        # (b, c, l)
        c = self.embedding(c).transpose(1, 2)
        # (b, c, l), (b, l, 3)
        params = self.text_encoder(c)
        # (b, c, t)
        alignments = self._normalize(get_attention_matrix(params, x.size(2)))
        # (b, c, t)
        c = torch.bmm(c, stt_alignments)

        # [(b, c, t)...]
        cs = self.poolings(c)
        xs = self.mel_encoder(x)
        
        y, kl_divs = self.mel_decoder(xs, cs)
        loss, recon_loss, kl_loss = self._get_loss(x, y, kl_divs, stt_outputs['params'].detach(), params, beta=beta)
        
        outputs = {'mels': x,
                   'pred': y,
                   'loss': loss,
                   'recon_loss': recon_loss, 
                   'kl_loss': kl_loss,
                   'kl_divs': kl_divs,
                   'alignments': alignments}
        
        return outputs
        
    def inference(self, c, length, alignments, temperature=1.0):
        # (b, c, l)
        t0 = time.time()
        c = self.embedding(c).transpose(1, 2)
        params = self.text_encoder(c)
        t1 = time.time()
        print('encoding :', t1 - t0)
        
        t0 = time.time()
        if alignments is None:
            alignments = get_attention_matrix(params, length)
        alignments = self._normalize(alignments)
        pad_length = ((alignments.size(2) - 1) // self.unit + 1) * self.unit - alignments.size(2)
        alignments = F.pad(alignments, (0, pad_length))
        t1 = time.time()
        print('alignment :', t1 - t0)    
        
        t0 = time.time()
        c = torch.bmm(c, alignments)
        cs = self.poolings(c)
        t1 = time.time()
        print('pooling :', t1 - t0)    
        
        t0 = time.time()
        y = self.mel_decoder.inference(cs, alignments.size(2)//16, temperature)
        t1 = time.time()
        print('decoding :', t1 - t0)    
        
        return y
    