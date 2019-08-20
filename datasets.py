import os
import random
import numpy as np
import PIL.Image as Image
import glob
import re
import torch
from torch.utils.data import Dataset


class loadedDataset(Dataset):
	def __init__(self, root_dir, sid, frame_num, time_freq, ps):
		self.root_dir = root_dir
		self.sid = str(sid)
		self.fnum = frame_num
		self.freq = time_freq
		self.ps = ps
		
		filename = self.sid + '_Hour_*.png'
		self.files = glob.glob(os.path.join(self.root_dir, filename))

		self.allframes = [None] * len(self.files)
		tids = []
		for i in range(len(self.allframes)):
			self.allframes[i] = []
			tids.append(int(re.findall(r'Hour_(\d+)', self.files[i])[0]))
		
		self.first_tid = min(tids)
		self.last_tid = max(tids)
		self.tids = sorted(tids)

	def __len__(self):
		return len(self.files) - (self.fnum // 2) * (self.freq + 1) + 1

	def __getitem__(self, idx):
		tid = self.tids[idx]
		ids1 = list(range(tid, tid + (self.fnum // 2)))
		ids2 = list(range(tid + (self.fnum // 2) + self.freq - 1, tid + (self.fnum // 2) + self.freq + (self.fnum // 2) * self.freq - 1, self.freq))
		ids = ids1 + ids2

		frames = []
		for ctid in ids:
			if not len(self.allframes[ctid - self.first_tid]):
				frame = np.array(Image.open(os.path.join(self.root_dir, self.sid + '_Hour_' + str(ctid) + '.png'))) / 255.0
				frame = np.transpose(frame.astype(np.float32), axes=[2, 0, 1])
				self.allframes[ctid - self.first_tid] = frame
			else:
				frame = self.allframes[ctid - self.first_tid]
			frames.append(frame)

		frames_crop = self._get_patch(frames)

		return ids, frames_crop

	def _get_patch(self, imgs):
		H = imgs[0].shape[1]
		W = imgs[0].shape[2]

		if self.ps < W and self.ps < H:
			xx = np.random.randint(0, W-self.ps)
			yy = np.random.randint(0, H-self.ps)
		
			imgs_crop = []
			for img in imgs:
				img_crop = img[:, yy:yy+self.ps, xx:xx+self.ps]
				imgs_crop.append(img_crop)
		else:
			imgs_crop = imgs

		if np.random.randint(2, size=1)[0] == 1:
			for i in range(len(imgs_crop)):
				imgs_crop[i] = np.flip(imgs_crop[i], axis=2).copy()
		if np.random.randint(2, size=1)[0] == 1: 
			for i in range(len(imgs_crop)):
				imgs_crop[i] = np.flip(imgs_crop[i], axis=1).copy()
		if np.random.randint(2, size=1)[0] == 1:
			for i in range(len(imgs_crop)):
				imgs_crop[i] = np.transpose(imgs_crop[i], (0, 2, 1)).copy()
		
		return imgs_crop
