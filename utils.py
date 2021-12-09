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
    sorted_idx = np.argsort(arr)
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

def convert_to_spectrogram(x, sr=16000, nfft=1024, win_length=720, hop_length=320):
    transformer = torchaudio.transforms.MelSpectrogram(sample_rate=sr, 
                                                       n_fft=nfft, 
                                                       win_length=win_length, 
                                                       hop_length=hop_length)
    x = torch.tensor(x).unsqueeze(0)
    melspec = transformer(x)
    
    return melspec


def load_wav2vec2(mode='base', device='cuda', eval=True):
    checkpoint_base = torch.load('./pretrained_checkpoints/wav2vec_small.pt')
    wav2vec2 = Wav2Vec2Model.build_model(convert_namespace_to_omegaconf(checkpoint_base['args']).model, task='audio_pretraining')

    if mode == 'random':
        utils.reset_all_weights(wav2vec2)
    else:
        if mode == 'finetune':
            checkpoint_finetune = torch.load('./pretrained_checkpoints/wav2vec_small_960h.pt')
            for key in checkpoint_finetune['model']:
                if 'w2v_encoder.w2v_model.' == key[:len('w2v_encoder.w2v_model.')]:
                    checkpoint_base['model'][key[len('w2v_encoder.w2v_model.'):]] = checkpoint_finetune['model'][key]
        wav2vec2.load_state_dict(checkpoint_base['model'])
        
    wav2vec2.to(device)
    if eval:
        wav2vec2.eval()
    return wav2vec2


def wav2vec2forward(model, source, aggregation=True):
    """
    Inference function of pretrained wav2vec2 to extract intermediate representations
    Ref: https://github.com/pytorch/fairseq/blob/89ec6e7efff867d258947acafc57189b257212d0/fairseq/models/wav2vec/wav2vec2.py
    """
    
    with torch.no_grad():
        cnn_features = model.feature_extractor(source)
    
        cnn_features = cnn_features.transpose(1, 2)
        features = model.layer_norm(cnn_features)

        if model.quantizer: # this is not None in pretrained w2v
            q = model.quantizer(features, produce_targets=False)
            quantized_features = q["x"]
            projected_quantized_features = model.project_q(quantized_features)

        if model.post_extract_proj is not None: # this is not None in pretrained w2v
            features = model.post_extract_proj(features)

        if model.input_quantizer is not None: # this is None in pretrained w2v
            q = model.input_quantizer(features, produce_targets=False)
            features = q['x']
            features = model.project_inp(features)
            
        encoder_outputs, encoder_layers_features = model.encoder(features, padding_mask=None, layer=hidden_layer)
            
        context_vectors = model.final_proj(encoder_outputs)
        
        ret = dict()
        ret['cnn_output'] = torch.squeeze(cnn_features)
        ret['vq'] = torch.squeeze(quantized_features)
        ret['projected_vq'] = torch.squeeze(projected_quantized_features)
        ret['encoder_output'] = torch.squeeze(encoder_outputs)
        ret['context_vector'] = torch.squeeze(context_vectors)
        
        if aggregation:
            ret['cnn_output'] = torch.mean(ret['cnn_output'], dim=0)
            ret['vq'] = torch.mean(ret['vq'], dim=0)
            ret['projected_vq'] = torch.mean(ret['projected_vq'], dim=0)
            ret['encoder_output'] = torch.mean(ret['encoder_output'], dim=0)
            ret['context_vector'] = torch.mean(ret['context_vector'], dim=0)
        
    return ret