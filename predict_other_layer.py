import os, glob, argparse
import numpy as np
import pandas as pd
import h5py, pickle
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


class Feat2Feat(pl.LightningModule):
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
#         loss = torch.nn.L1Loss()(recon, spec)
        loss = torch.nn.MSELoss()(recon, spec)
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
    
    
class FeatFeatDataset(Dataset):
    def __init__(self, h5_path, feat_input, feat_target):
        self.h5 = h5py.File(h5_path, 'r')
        
        self.key_input = []
        self.key_target = []
        
        for key in self.h5:
            feat = key.split('-')[-1]
            if feat == feat_input:
                key_target = key.split('-')[0] + '-' + feat_target
                self.key_input.append(key)
                self.key_target.append(key_target)
            
                
    def __len__(self):
        return len(self.key_input)
    
    def __getitem__(self, idx):
        return self.h5[self.key_input[idx]][:], self.h5[self.key_target[idx]][:]
    
    
def train(args):
    train_set = FeatFeatDataset(args.h5_train, args.feat_input, args.feat_target)
    val_set = FeatFeatDataset(args.h5_test, args.feat_input, args.feat_target)
    
    train_loader = DataLoader(
        train_set,
        shuffle=True,
        batch_size=args.bz,
        num_workers=32
    )
    
    val_loader = DataLoader(
        val_set,
        shuffle=False,
        batch_size=args.bz,
        num_workers=32
    )
    
    logger = TensorBoardLogger(os.path.join(args.exp_dir, "logs"), name=f"{args.feat_input}-{args.feat_target}")
    
    checkpoint_saver = ModelCheckpoint(
            dirpath=os.path.join(args.exp_dir, 'checkpoints'),
            save_top_k=1,
            monitor="val_loss", mode="min", verbose=True, save_last=False
        )
    
    accumulate_grad_batches = 32 if args.bz == 1 else 1
    
    trainer = pl.Trainer(
        accumulate_grad_batches=accumulate_grad_batches,
        gradient_clip_val=0.5,
        max_epochs=args.epoch,
        callbacks=[checkpoint_saver],
        default_root_dir=args.exp_dir,
        gpus=1,
        logger=logger,
    )

    channel_map = {'encoder_output': 768, 
                  'cnn_output': 512, 
                  'context_vector': 256, 
                  'vq': 256, 
                  'projected_vq': 256}
    
    vec2wav = Feat2Feat(channel_map[args.feat_input], channel_map[args.feat_target])
    
    # Training
    trainer.fit(vec2wav, train_dataloaders=train_loader, val_dataloaders=val_loader)
    
    
if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Reconstruct input wav from hidden representation')
    parser.add_argument('--exp_dir', type=str, default='./exp/tmp',
                        help='directory to store logs and checkpoint')
    parser.add_argument('--seed', type=int, default=42,
                        help='random seed')
    parser.add_argument('--bz', type=int, default=1,
                        help='batch size')
    parser.add_argument('--epoch', type=int, default=200,
                        help='max epoch')
    parser.add_argument('--feat_input', type=str, choices=['cnn_output', 'vq', 'projected_vq', 'encoder_output', 'context_vector'],
                        help='feature input')
    parser.add_argument('--feat_target', type=str, choices=['cnn_output', 'vq', 'projected_vq', 'encoder_output', 'context_vector'],
                        help='feature output')
    parser.add_argument('--h5_train', type=str,
                        help='h5 file containing extracted features')
    parser.add_argument('--h5_test', type=str,
                        help='h5 file containing extracted features')
    
    args = parser.parse_args()
    
    utils.set_random_seed(args.seed)
    
    train(args)