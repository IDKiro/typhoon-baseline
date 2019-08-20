import shutil, os
import numpy as np
import cv2
import torch


class AverageMeter(object):
	def __init__(self):
		self.reset()

	def reset(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.count = 0

	def update(self, val, n=1):
		self.val = val
		self.sum += val * n
		self.count += n
		self.avg = self.sum / self.count


class ListAverageMeter(object):
    """Computes and stores the average and current values of a list"""
    def __init__(self):
        self.len = 10000  # set up the maximum length
        self.reset()

    def reset(self):
        self.val = [0] * self.len
        self.avg = [0] * self.len
        self.sum = [0] * self.len
        self.count = 0

    def set_len(self, n):
        self.len = n
        self.reset()

    def update(self, vals, n=1):
        assert len(vals) == self.len, 'length of vals not equal to self.len'
        self.val = vals
        for i in range(self.len):
            self.sum[i] += self.val[i] * n
        self.count += n
        for i in range(self.len):
            self.avg[i] = self.sum[i] / self.count
            

def save_checkpoint(state, save_dir, is_best=False, filename='checkpoint.pth.tar'):
    torch.save(state, os.path.join(save_dir, filename))
    if is_best:
        shutil.copyfile(os.path.join(save_dir, filename), os.path.join(save_dir, 'model_best.pth.tar'))


def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer


#for [B,C,W,H]
def bgr_gray(input_tensor):
    B=input_tensor[:,0].view(input_tensor.size()[0],1,input_tensor.size()[2],input_tensor.size()[3])
    G=input_tensor[:,1].view(input_tensor.size()[0],1,input_tensor.size()[2],input_tensor.size()[3])
    R=input_tensor[:,2].view(input_tensor.size()[0],1,input_tensor.size()[2],input_tensor.size()[3])
    gray_tensor=B*0.114+G*0.587+R*0.299
    return gray_tensor

def diff_mask(gen_frames, gt_frames, min_value=-1, max_value=1):
    # normalize to [0, 1]
    delta = max_value - min_value
    gen_frames = (gen_frames - min_value) / delta
    gt_frames = (gt_frames - min_value) / delta

    gen_gray_frames = bgr_gray(gen_frames)
    gt_gray_frames = bgr_gray(gt_frames)

    diff = torch.abs(gen_gray_frames - gt_gray_frames)
    return diff

