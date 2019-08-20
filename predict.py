from __future__ import division
from __future__ import print_function
import os, time, scipy.io, shutil
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import PIL.Image as Image
import cv2 as cv
import glob
import argparse
import re

from models.model import Network

parser = argparse.ArgumentParser(description = 'Train')
# training parameters
parser.add_argument('--origin_size', default=1999, type=int, help='origin image size')
parser.add_argument('--patch_size', default=512, type=int, help='image patch size')
parser.add_argument('--clip_size', default=184, type=int, help='clipped image size')
args = parser.parse_args()


test_dir = './img_data/test/'
save_dir = './save_model/'
result_dir = './result/final/'
HEADS = ['U', 'V', 'W', 'X', 'Y', 'Z']

model = Network(in_channels=3, out_channels=3).cuda()
model.eval()

if os.path.exists(os.path.join(save_dir, 'checkpoint.pth.tar')):
    # load existing model
    print('==> loading existing model')
    g_info = torch.load(os.path.join(save_dir, 'checkpoint.pth.tar'))
    model.load_state_dict(g_info['state_dict'])
else:
    print('==> No trained model detected!')
    exit(1)

if not os.path.isdir(result_dir):
    os.makedirs(result_dir)

for head in HEADS:
    print('Predicting:', head)
    filename = head + '_Hour_*.png'
    files = glob.glob(os.path.join(test_dir, filename))
    tids = []
    for i in range(len(files)):
        tids.append(int(re.findall(r'Hour_(\d+)', files[i])[0]))
    last_tid = max(tids)

    frames = []
    for past_tid in range(last_tid - 5, last_tid + 1):
        frame = np.array(Image.open(os.path.join(test_dir, head + '_Hour_' + str(past_tid) + '.png'))) / 255.0
        frame = np.transpose(frame.astype(np.float32), axes=[2, 0, 1])
        frames.append(torch.from_numpy(frame).unsqueeze(0).unsqueeze(2).cuda())
    
    input = torch.cat(frames, 2)

    with torch.no_grad():
        output = model(input)

    output_np = np.clip(output.squeeze().detach().cpu().numpy(), 0, 1)

    for i in range(6):
        pre_tid = last_tid + (i + 1) * 6

        output_img = np.transpose(output_np[:, i, :, :], axes=[1, 2, 0])
        output_img = cv.resize(output_img, (1999, 1999), interpolation=cv.INTER_CUBIC)
        # scipy.misc.toimage(output_img*255, high=255, low=0, cmin=0, cmax=255).save(os.path.join(result_dir, head + '_Hour_' + str(pre_tid) + '.jpg'))

        img_np = np.array(output_img * 4095.0, dtype=np.uint16)
        output_ban08 = img_np[:, :, 0]
        output_ban09 = img_np[:, :, 1]
        output_ban10 = img_np[:, :, 2]

        np.save(os.path.join(result_dir, head + '_Hour_' + str(pre_tid) + '_Band_08.npy'), output_ban08)
        np.save(os.path.join(result_dir, head + '_Hour_' + str(pre_tid) + '_Band_09.npy'), output_ban09)
        np.save(os.path.join(result_dir, head + '_Hour_' + str(pre_tid) + '_Band_10.npy'), output_ban10)
