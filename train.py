from __future__ import division
from __future__ import print_function
import os, time, scipy.io, shutil, sys
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import glob
import argparse
import logging

from utils import AverageMeter, save_checkpoint, adjust_learning_rate
from datasets import loadedDataset
from losses import RMSE_Loss
from models.model import Network


parser = argparse.ArgumentParser(description = 'Train')
# training parameters
parser.add_argument('--batch_size', default=4, type=int, help='mini-batch size')
parser.add_argument('--patch_size', default=256, type=int, help='image patch size')
parser.add_argument('-lr', default=1e-4, type=float, help='G learning rate')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('-frame_num', default=6*2, type=int, help='sum of frames')
parser.add_argument('-time_freq', default=6, type=int, help='predict freq')
parser.add_argument('-epochs', default=40, type=int, help='sum of epochs')
# visualization setting
parser.add_argument('-save_freq', default=10, type=int, help='save freq of visualization')
args = parser.parse_args()


train_dir = './img_data/train/'
save_dir = './save_model/'
result_dir = './result/'
HEADS = ['A', 'B', 'C']

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')
if not os.path.isdir(result_dir):
    os.makedirs(result_dir)
fh = logging.FileHandler(os.path.join(result_dir, 'train.log'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

train_datasets = [None] * len(HEADS)
train_loaders = [None] * len(HEADS)

for i in range(len(HEADS)):
    train_datasets[i] = loadedDataset(train_dir, HEADS[i], args.frame_num, args.time_freq, args.patch_size)
    train_loaders[i] = torch.utils.data.DataLoader(
        train_datasets[i], batch_size=args.batch_size, shuffle=True, pin_memory=True)

model = Network(in_channels=3, out_channels=3).cuda()
rmse_loss = RMSE_Loss()
criterion = nn.MSELoss()

if os.path.exists(os.path.join(save_dir, 'checkpoint.pth.tar')):
    # load existing model
    print('==> loading existing model')
    model_info = torch.load(os.path.join(save_dir, 'checkpoint.pth.tar'))
    model.load_state_dict(model_info['state_dict'])
    optimizer = torch.optim.Adam(model.parameters())
    optimizer.load_state_dict(model_info['optimizer'])
    cur_epoch = model_info['epoch']
else:
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    # create model
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    cur_epoch = 0


for epoch in range(cur_epoch, args.epochs + 1):
    rmse = AverageMeter()
    losses = AverageMeter()

    if epoch > args.epochs // 2:
        optimizer = adjust_learning_rate(optimizer, args.lr * 0.1)

    for headid in range(len(HEADS)):
        for ind, (ids, frames) in enumerate(train_loaders[headid]):
            input_list = []
            for input_tensor in frames[:-(args.frame_num // 2)]:
                input_list.append(input_tensor.unsqueeze(2))
            input = torch.cat(input_list, 2).cuda()
            
            target_list = []
            for target_tensor in frames[-(args.frame_num // 2):]:
                target_list.append(target_tensor.unsqueeze(2))
            target = torch.cat(target_list, 2).cuda()

            output = model(input)

            loss = criterion(output, target)
            optimizer.zero_grad()
            loss.backward()
            losses.update(loss.item())
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            t_rmse = rmse_loss(output, target)
            rmse.update(t_rmse.item())     # show RMSE loss
            
            logging.info('[{0}][{1}][{2}]\t'
                'lr: {lr:.5f}\t'
                'loss: {loss.val:.4f} ({loss.avg:.4f})\t'
                'RMSE: {rmse.val:.4f} ({rmse.avg:.4f})'.format(
                epoch, headid, ind, 
                lr=optimizer.param_groups[-1]['lr'],
                loss=losses,
                rmse=rmse))

            if epoch % args.save_freq == 0:
                if not os.path.isdir(os.path.join(result_dir, '%04d'%epoch)):
                    os.makedirs(os.path.join(result_dir, '%04d'%epoch))
                    
                output_np = np.clip(output.detach().cpu().numpy(), 0, 1)
                target_np = np.clip(target.detach().cpu().numpy(), 0, 1)

                for indp in range(output_np.shape[0]): 
                    temp = np.concatenate((
                        np.transpose(target_np[indp, :, 0, :, :], axes=[1, 2, 0]), 
                        np.transpose(output_np[indp, :, 0, :, :], axes=[1, 2, 0])
                        ), axis=1)  # only show first output
                    scipy.misc.toimage(temp*255, high=255, low=0, cmin=0, cmax=255).save(os.path.join(result_dir, '%04d/train_%d_%d_%d.jpg'%(epoch, headid, ind, indp)))

    save_checkpoint({
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'optimizer' : optimizer.state_dict()}, 
        save_dir=save_dir,
        filename='checkpoint.pth.tar')

