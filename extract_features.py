import os, glob, argparse
import soundfile as sf
import numpy as np
import pandas as pd
import h5py
import tqdm
import torch, fairseq

import utils
from fairseq.models.wav2vec.wav2vec2 import Wav2Vec2Model
from fairseq.dataclass.utils import convert_namespace_to_omegaconf

def wav2vec2forward(model, source, aggregation=True, hidden_layer=None):
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
        ret['cnn_output'] = cnn_features[0]
        ret['vq'] = quantized_features[0]
        ret['projected_vq'] = projected_quantized_features[0]
        ret['encoder_output'] = encoder_outputs[0]
        ret['context_vector'] = context_vectors[0]
        if len(encoder_layers_features) > 0:
            ret['encoder_hiddens'] = [h[0][0] for h in encoder_layers_features]
        
        if aggregation:
            ret['cnn_output'] = torch.mean(ret['cnn_output'], dim=0)
            ret['vq'] = torch.mean(ret['vq'], dim=0)
            ret['projected_vq'] = torch.mean(ret['projected_vq'], dim=0)
            ret['encoder_output'] = torch.mean(ret['encoder_output'], dim=0)
            ret['context_vector'] = torch.mean(ret['context_vector'], dim=0)
            if len(encoder_layers_features) > 0:
                ret['encoder_hiddens'] = [torch.mean(h, dim=0) for h in ret['encoder_hiddens']]
        
        return ret
    
    
def load_wav2vec2(ckpt_path, load_weights):
    checkpoint = torch.load('./pretrained_checkpoints/wav2vec_small.pt')

#     model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([ckpt_path], arg_overrides={"data": "./data"})
    model = Wav2Vec2Model.build_model(convert_namespace_to_omegaconf(checkpoint['args']).model, task=None)
    wav2vec2 = model
    wav2vec2.cuda()
    if load_weights:
        if os.path.basename(ckpt_path) != 'wav2vec_small.pt':
            checkpoint_tmp = torch.load(ckpt_path)
            for key in checkpoint_tmp['model']:
                if 'w2v_encoder.w2v_model.' == key[:len('w2v_encoder.w2v_model.')]:
                    checkpoint['model'][key[len('w2v_encoder.w2v_model.'):]] = checkpoint_tmp['model'][key]
        wav2vec2.load_state_dict(checkpoint['model'])
    else:
        utils.reset_all_weights(wav2vec2)
    wav2vec2.eval()
    
    return wav2vec2


def extract_wav2vec2_features(model, md_file_path, output_h5_path, aggragation=True):
    # Load metadata file containing audio id and their paths
    md = pd.read_csv(md_file_path)
    
    hf = h5py.File(output_h5_path, 'w')
    
    for i, row in tqdm.tqdm(md.iterrows()):
        wav_id = row['wav_id']
        audio, sr = sf.read(row['wav_path'], dtype='float32')
        audio = torch.from_numpy(audio).unsqueeze(0).cuda()
        output = wav2vec2forward(model, audio, aggregation=aggragation)

        hf.create_dataset(f"{wav_id}-cnn_output", data=output['cnn_output'].cpu())
        hf.create_dataset(f"{wav_id}-vq", data=output['vq'].cpu())
        hf.create_dataset(f"{wav_id}-projected_vq", data=output['projected_vq'].cpu())
        hf.create_dataset(f"{wav_id}-encoder_output", data=output['encoder_output'].cpu())
        hf.create_dataset(f"{wav_id}-context_vector", data=output['context_vector'].cpu())
        if 'encoder_hiddens' in output:
            hf.create_dataset(f"{wav_id}-encoder_hiddens", data=torch.stack(output['encoder_hiddens'].cpu()))
    
    hf.close()
    
    
if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Extract features from pretrained speech models')
    parser.add_argument('--seed', type=int, default=42,
                        help='random seed')
    parser.add_argument('--md_file', type=str,
                        help='metadata file containing audio id and their paths')
    parser.add_argument('--ckpt', type=str, default='./pretrained_checkpoints/wav2vec_small.pt',
                        help='ckpt path to pretrained model')
    parser.add_argument('--model', type=str, default='wav2vec2', choices=['wav2vec2'],
                        help='pretrain model')
    parser.add_argument('--random_init', action='store_true',
                        help='use a randomly initialized model')
    parser.add_argument('--aggregation', action='store_true',
                        help='average features over time dimension')
    parser.add_argument('--out_path', type=str,
                        help='pretrain model')
    
    args = parser.parse_args()
    
    utils.set_random_seed(args.seed)
    
    load_weights = False if args.random_init else True
    
    if args.model == 'wav2vec2':
        model = load_wav2vec2(args.ckpt, load_weights)
        extract_wav2vec2_features(model, args.md_file, args.out_path, args.aggregation)