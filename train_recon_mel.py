import os, glob, argparse
import numpy as np
import pandas as pd
import h5py
import soundfile as sf
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import auraloss
from fairseq.modules import Fp32LayerNorm, TransposeLast
import utils


class Vec2Mel(pl.LightningModule):
    def __init__(self, in_channel=256, out_feature=128):
        super().__init__()
        self.linear_layers = torch.nn.ModuleList()
        hidden_channel = in_channel
        self.linear_layers.append(nn.Linear(in_features=in_channel, out_features=hidden_channel))
        for i in range(2):
            self.linear_layers.append(nn.Linear(in_features=hidden_channel, out_features=hidden_channel))
        self.final_linear = nn.Linear(in_features=hidden_channel, out_features=out_feature)
        
        self.activation_func = torch.nn.ReLU()
        
        
    def forward(self, x):
        for i, layer in enumerate(self.linear_layers):
            x = layer(x) + x
            x = self.activation_func(x)
        
        x = self.final_linear(x)
        
        return x
    
    
    def log_loss(self, recon, spec):
        spec = spec[0, :recon.shape[1], :]
        spec = spec[0]
        bz = recon.shape[0]
        loss = torch.sum( torch.log(recon) + spec/(recon) )
        return loss / bz
    
    def training_step(self, batch, batch_idx):
        feat, spec = batch
        recon = self(feat)
        
#         print(spec.shape, recon.shape)
#         print(recon.shape, wav.shape)
        spec = spec[:, :recon.shape[1], :]
        #loss = auraloss.time.SISDRLoss()(recon, wav)
        #loss = auraloss.freq.MelSTFTLoss(16000)()
        loss = torch.nn.L1Loss()(recon, spec)
#         loss = self.recon_loss(recon, spec)
        
        self.log('train_loss', loss, on_step=False, on_epoch=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        feat, spec = batch
        recon = self(feat)
#         print(spec.shape, recon.shape)
        spec = spec[:, :recon.shape[1], :]
        loss = torch.nn.L1Loss()(recon, spec)
        
        result = {'val_loss': loss}
        
        return result
    
    
    def validation_epoch_end(self, output):
        val_loss = torch.stack([x['val_loss'] for x in output]).mean()
        self.log('val_loss', val_loss, logger=True) 
        
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=5e-4)
        
#         lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
#             optimizer, 'min', factor=0.5, patience=10, threshold=0.02,
#             threshold_mode='rel', min_lr=1e-6, verbose=True
#         )
        return {
            'optimizer': optimizer,
#             'lr_scheduler': lr_scheduler,
#             'monitor': 'train_sisdr'
        }
    
    
class WavFeatDataset(Dataset):
    def __init__(self, h5_path, h5_target, df_path, feat_name):
        self.h5 = h5py.File(h5_path, 'r')
        self.h5spec = h5py.File(h5_target, 'r')
        self.keys = []
        df = pd.read_csv(df_path)
        if 'test' in h5_path:
            num_sa1 = len(df[df['sentence']=='SA1'])
            num_sa2 = len(df[df['sentence']=='SA2'])
            df = df.drop(list(df[df['sentence']=='SA1'].sample(num_sa1-10).index))
            df = df.drop(list(df[df['sentence']=='SA2'].sample(num_sa2-10).index))
        else:
            num_sa1 = len(df[df['sentence']=='SA1'])
            num_sa2 = len(df[df['sentence']=='SA2'])
            df = df.drop(list(df[df['sentence']=='SA1'].sample(num_sa1-50).index))
            df = df.drop(list(df[df['sentence']=='SA2'].sample(num_sa2-50).index))
        for i, row in df.iterrows():
            self.keys.append((row['wav_path'], row['wav_id'] + '-' + feat_name))
                
    def __len__(self):
        return len(self.keys)
    
    def __getitem__(self, idx):
#         idx = (idx % 64)**2 % len(self)
        s, sr = sf.read(self.keys[idx][0], dtype='float32')
        return self.h5[self.keys[idx][1]][:], self.h5spec[self.keys[idx][1].split('-')[0]][:]
    
    
def train(args):
    train_set = WavFeatDataset(args.h5_train, args.h5_target_train, args.df_train, args.feature_name)
    val_set = WavFeatDataset(args.h5_test, args.h5_target_test, args.df_test, args.feature_name)
    
    train_loader = DataLoader(
        train_set,
        shuffle=True,
        batch_size=1,
        num_workers=8
    )
    
    val_loader = DataLoader(
        val_set,
        shuffle=False,
        batch_size=1,
        num_workers=8
    )
    
    logger = TensorBoardLogger(os.path.join(args.exp_dir, "logs"), name=args.feature_name)
    
    checkpoint_saver = ModelCheckpoint(
            dirpath=os.path.join(args.exp_dir, 'checkpoints'),
            save_top_k=1,
            monitor="val_loss", mode="min", verbose=True, save_last=False
        )
    
    trainer = pl.Trainer(
        accumulate_grad_batches=32,
        gradient_clip_val=0.5,
        max_epochs=500,
        callbacks=[checkpoint_saver],
        default_root_dir=args.exp_dir,
        gpus=0,
        logger=logger
    )

    in_channel = {'encoder_output': 768, 
                  'cnn_output': 512, 
                  'context_vector': 256, 
                  'vq': 256, 
                  'projected_vq': 256}[args.feature_name]
    
    vec2wav = Vec2Mel(in_channel, out_feature=64)
    
    # Training
    trainer.fit(vec2wav, train_dataloaders=train_loader, val_dataloaders=val_loader)
    
    
if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Reconstruct input wav from hidden representation')
    parser.add_argument('--exp_dir', type=str, default='./exp/tmp',
                        help='directory to store logs and checkpoint')
    parser.add_argument('--seed', type=int, default=42,
                        help='random seed')
    parser.add_argument('--df_train', type=str,
                        help='metadata file containing audio id and their properties')
    parser.add_argument('--df_test', type=str,
                        help='metadata file containing audio id and their properties')
    parser.add_argument('--feature_name', type=str, choices=['cnn_output', 'vq', 'projected_vq', 'encoder_output', 'context_vector'],
                        help='feature name in h5 file to be used for classification')
    parser.add_argument('--h5_train', type=str,
                        help='h5 file containing extracted features')
    parser.add_argument('--h5_target_train', type=str,
                        help='h5 file containing extracted features')
    parser.add_argument('--h5_test', type=str,
                        help='h5 file containing extracted features')
    parser.add_argument('--h5_target_test', type=str,
                        help='h5 file containing extracted features')
    
    args = parser.parse_args()
    
    utils.set_random_seed(args.seed)
    
    train(args)