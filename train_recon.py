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


class Vec2Wav(pl.LightningModule):
    def __init__(self, in_channel=256):
        super().__init__()
        
#         self.deconv_layers = torch.nn.ModuleList()
#         in_channels = [in_channel, 512, 256, 128, 64, 32, 16]
#         out_channels = [512, 256, 128, 64, 32, 16, 1]
#         kernel_sizes = [2, 2, 3, 3, 3, 3, 10]
#         strides = [2, 2, 2, 2, 2, 2, 5]
#         for i in range(6):
#             self.deconv_layers.append(
#                 nn.Sequential(nn.ConvTranspose1d(in_channels[i], out_channels[i], 
#                                                  kernel_size=kernel_sizes[i], stride=strides[i], bias=False),
#                               TransposeLast(),
#                               Fp32LayerNorm(out_channels[i], elementwise_affine=True),
#                               TransposeLast(),
#                               nn.GELU())
#             )
        
#         self.final_deconv = nn.ConvTranspose1d(16, 1, kernel_size=10, stride=5, bias=False)
        
        self.deconv0 = torch.nn.ConvTranspose1d(in_channel, 512, kernel_size=2, stride=2, bias=False)
        self.layer_norm1 = Fp32LayerNorm(512, elementwise_affine=True)
        self.deconv1 = torch.nn.ConvTranspose1d(512, 256, kernel_size=2, stride=2, bias=False)
        self.layer_norm2 = Fp32LayerNorm(256, elementwise_affine=True)
        self.deconv2 = torch.nn.ConvTranspose1d(256, 128, kernel_size=3, stride=2, bias=False)
        self.deconv3 = torch.nn.ConvTranspose1d(128, 64, kernel_size=3, stride=2, bias=False)
        self.deconv4 = torch.nn.ConvTranspose1d(64, 32, kernel_size=3, stride=2, bias=False)
        self.deconv5 = torch.nn.ConvTranspose1d(32, 16, kernel_size=3, stride=2, bias=False)
        self.deconv6 = torch.nn.ConvTranspose1d(16, 1, kernel_size=10, stride=5, bias=False)
        
        self.activation_func = torch.nn.GELU()
        
        
    def forward(self, x):
#         for layer in self.deconv_layers:
#             x = layer(x)
        
#         x = self.final_deconv(x)
        
        x = self.deconv0(x)
        x = TransposeLast()(x)
        x = self.layer_norm1(x)
        x = TransposeLast()(x)
        x = self.activation_func(x)
        x = self.deconv1(x)
        x = TransposeLast()(x)
        x = self.layer_norm2(x)
        x = TransposeLast()(x)
        x = self.activation_func(x)
        x = self.deconv2(x)
        x = self.activation_func(x)
        x = self.deconv3(x)
        x = self.activation_func(x)
        x = self.deconv4(x)
        x = self.activation_func(x)
        x = self.deconv5(x)
        x = self.activation_func(x)
        x = self.deconv6(x)
        x = self.activation_func(x)
        
        return x
    
    
    def training_step(self, batch, batch_idx):
        feat, wav = batch
        recon = self(feat).squeeze(0)
#         print(recon.shape, wav.shape)
        wav = wav[:, :recon.shape[1]]
        #loss = auraloss.time.SISDRLoss()(recon, wav)
        loss = torch.nn.L1Loss()(recon, wav)
        
        self.log('train_sisdr', loss, on_step=False, on_epoch=True)
        self.log('sisdr', loss, prog_bar=True)
        
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=2e-3)
        
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
    def __init__(self, h5_path, df_path, feat_name):
        self.h5 = h5py.File(h5_path, 'r')
        self.keys = []
        df = pd.read_csv(df_path)
        for i, row in df.iterrows():
            self.keys.append((row['wav_path'], row['wav_id'] + '-' + feat_name))
                
    def __len__(self):
        return len(self.keys)
    
    def __getitem__(self, idx):
#         idx = (idx % 64)**2 % len(self)
        s, sr = sf.read(self.keys[idx][0], dtype='float32')
        return self.h5[self.keys[idx][1]][:].T, s
    
    
def train(args):
    train_set = WavFeatDataset(args.h5, args.md_file, args.feature_name)
    
    train_loader = DataLoader(
        train_set,
        shuffle=True,
        batch_size=1,
        num_workers=8
    )
    
    logger = TensorBoardLogger(os.path.join(args.exp_dir, "logs"), name=args.feature_name)
    
    checkpoint_saver = ModelCheckpoint(
            dirpath=os.path.join(args.exp_dir, 'checkpoints'),
            save_top_k=1,
            monitor="train_sisdr", mode="min", verbose=True, save_last=False
        )
    
    trainer = pl.Trainer(
        accumulate_grad_batches=32,
        gradient_clip_val=0.5,
        max_epochs=1000,
        callbacks=[checkpoint_saver],
        default_root_dir=args.exp_dir,
        gpus=0,
        logger=logger
    )

    in_channel = 768
    
#     cp = torch.load("/mnt/scratch09/vnguyen/SpeakerRecognition/exp/tmp/checkpoints/epoch=22-step=3334.ckpt")
    vec2wav = Vec2Wav(in_channel)
#     vec2wav.load_state_dict(cp['state_dict'])
    
    # Training
    trainer.fit(vec2wav, train_dataloader=train_loader)
    
    
if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Reconstruct input wav from hidden representation')
    parser.add_argument('--exp_dir', type=str, default='./exp/tmp',
                        help='directory to store logs and checkpoint')
    parser.add_argument('--seed', type=int, default=42,
                        help='random seed')
    parser.add_argument('--md_file', type=str,
                        help='metadata file containing audio id and their properties')
    parser.add_argument('--feature_name', type=str, choices=['cnn_output', 'vq', 'projected_vq', 'encoder_output', 'context_vector'],
                        help='feature name in h5 file to be used for classification')
    parser.add_argument('--h5', type=str,
                        help='h5 file containing extracted features')
    
    args = parser.parse_args()
    
    utils.set_random_seed(args.seed)
    
    train(args)