import os, random
import numpy as np
import torch
import torch.nn as nn
import torchaudio


def set_random_seed(seed) -> None:
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    
    
def reset_all_weights(model: nn.Module) -> None:
    """
    refs:
        - https://discuss.pytorch.org/t/how-to-re-set-alll-parameters-in-a-network/20819/6
        - https://stackoverflow.com/questions/63627997/reset-parameters-of-a-neural-network-in-pytorch
        - https://pytorch.org/docs/stable/generated/torch.nn.Module.html
    """

    @torch.no_grad()
    def weight_reset(m: nn.Module):
        # - check if the current module has reset_parameters & if it's callabed called it on m
        reset_parameters = getattr(m, "reset_parameters", None)
        if callable(reset_parameters):
            m.reset_parameters()

    # Applies fn recursively to every submodule see: https://pytorch.org/docs/stable/generated/torch.nn.Module.html
    model.apply(fn=weight_reset)
    
    
def find_sorted_position(arr):
    """
    return positions in sorted arr
    """
    sorted_idx = np.argsort(arr)[::-1]
    ret = np.empty_like(arr, dtype=np.int32)
    for i, v in enumerate(sorted_idx):
        ret[v] = i
    return ret


def convert_to_mel(x, sr=16000, nfft=400, win_length=400, hop_length=320, n_mels=64, power=2, normalized=False):
    transformer = torchaudio.transforms.MelSpectrogram(sample_rate=sr, 
                                                       n_fft=nfft, 
                                                       win_length=win_length, 
                                                       hop_length=hop_length, 
                                                       n_mels=n_mels, 
                                                       normalized=normalized)
    x = torch.tensor(x).unsqueeze(0)
    melspec = transformer(x)
    
    return melspec

def convert_to_spectrogram(x, sr=16000, nfft=400, win_length=400, hop_length=320):
    transformer = torchaudio.transforms.MelSpectrogram(sample_rate=sr, 
                                                       n_fft=nfft, 
                                                       win_length=win_length, 
                                                       hop_length=hop_length)
    x = torch.tensor(x).unsqueeze(0)
    melspec = transformer(x)
    
    return melspec