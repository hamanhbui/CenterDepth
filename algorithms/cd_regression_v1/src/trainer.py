import os
import time
import logging
import shutil
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from progress.bar import Bar
from utils.utils import AverageMeter

from models.losses import FastFocalLoss, RegWeightedL1Loss
from models.losses import BinRotLoss, WeightedBCELoss, DepthLoss
from models.decode import generic_decode
from models.utils import _sigmoid, flip_tensor, flip_lr_off, flip_lr
from utils.post_process import generic_post_process

from models.model import create_model, load_model, save_model
from dataloaders.dataset_factory import get_dataset

class GenericLoss(torch.nn.Module):
	def __init__(self, opt):
		super(GenericLoss, self).__init__()
		self.crit = FastFocalLoss(opt=opt)
		self.crit_reg = RegWeightedL1Loss()
		self.depth_loss = DepthLoss()
		self.opt = opt

	def _sigmoid_output(self, output):
		if 'hm' in output:
			output['hm'] = _sigmoid(output['hm'])
		if 'dep' in output:
			output['dep'] = 1. / (output['dep'].sigmoid() + 1e-6) - 1.
		return output

	def forward(self, outputs, batch):
		opt = self.opt
		losses = {head: 0 for head in opt.heads}

		for s in range(opt.num_stacks):
			output = outputs[s]
			output = self._sigmoid_output(output)

			if 'hm' in output:
				losses['hm'] += self.crit(
					output['hm'], batch['hm'], batch['ind'], 
					batch['mask'], batch['cat']) / opt.num_stacks
		
			regression_heads = ['reg', 'wh', 'tracking', 'dep']

			for head in regression_heads:
				if head == 'dep':
					losses[head] += self.depth_loss(
						output[head], batch[head + '_mask'],
						batch['ind'], batch) / opt.num_stacks
				elif head in output:
					losses[head] += self.crit_reg(
						output[head], batch[head + '_mask'],
						batch['ind'], batch[head]) / opt.num_stacks
		
		losses['tot'] = 0
		for head in opt.heads:
			losses['tot'] += opt.weights[head] * losses[head]
		
		return losses['tot'], losses

class ModleWithLoss(torch.nn.Module):
	def __init__(self, model, loss):
		super(ModleWithLoss, self).__init__()
		self.model = model
		self.loss = loss
  
	def forward(self, batch):
		pre_img = batch['pre_img'] if 'pre_img' in batch else None
		pre_hm = batch['pre_hm'] if 'pre_hm' in batch else None
		outputs = self.model(batch['image'], pre_img, pre_hm)
		loss, loss_stats = self.loss(outputs, batch)
		return outputs[-1], loss, loss_stats

def get_optimizer(opt, model):
	if opt.optim == 'adam':
		optimizer = torch.optim.Adam(model.parameters(), opt.learning_rate)
	elif opt.optim == 'sgd':
		optimizer = torch.optim.SGD(
		model.parameters(), opt.learning_rate, momentum=0.9, weight_decay=0.0001)
	else:
		assert 0, opt.optim
	return optimizer

class Trainer_cd_regression_v1:
	def __init__(self, opt, device, exp_id):
		self.device = device
		self.exp_id = exp_id
		self.opt = opt

		print('Creating model...')
		self.model = create_model(opt.model, opt.heads, opt.head_conv, opt=opt)
		self.optimizer = get_optimizer(opt, self.model)

		self.start_epoch = 0
		if opt.resume:
			self.model, self.optimizer, self.start_epoch = load_model(self.model, opt.checkpoint_name, opt, self.optimizer)

		self.loss_stats, self.loss = self._get_losses(opt)
		self.model_with_loss = ModleWithLoss(self.model, self.loss)

		self.set_device(self.device)

		Dataset = get_dataset('custom')
		print('Setting up train data...')
		self.train_loader = DataLoader(Dataset(opt.train_meta_filenames, opt, 'train'), batch_size=opt.batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)

		print('Setting up validation data...')
		self.val_loader = DataLoader(Dataset(opt.val_meta_filenames, opt, 'val'), batch_size=1, shuffle=False, num_workers=1, pin_memory=True)

		print('Setting up test data...')
		self.test_loader = DataLoader(Dataset(opt.test_meta_filenames, opt, 'test'), batch_size=1, shuffle=False, num_workers=1, pin_memory=True)

	def set_writer(self, log_dir):
		writers = {}
		if not os.path.exists(log_dir):
			os.mkdir(log_dir)
		shutil.rmtree(log_dir)
		for mode in ["train", "val"]:
			writers[mode] = SummaryWriter(os.path.join(log_dir, mode))
		return writers

	def set_device(self, device):
		self.model_with_loss = self.model_with_loss.to(device)
		
		for state in self.optimizer.state.values():
			for k, v in state.items():
				if isinstance(v, torch.Tensor):
					state[k] = v.to(device=device, non_blocking=True)

	def run_epoch(self, phase, epoch, data_loader):
		model_with_loss = self.model_with_loss
		if phase == 'train':
			model_with_loss.train()
		else:
			model_with_loss.eval()
			torch.cuda.empty_cache()

		opt = self.opt
		data_time, batch_time = AverageMeter(), AverageMeter()
		avg_loss_stats = {l: AverageMeter() for l in self.loss_stats \
						if l == 'tot' or opt.weights[l] > 0}
		num_iters = len(data_loader)
		bar = Bar('{}'.format(self.exp_id), max=num_iters)
		end = time.time()
		for iter_id, batch in enumerate(data_loader):
			if iter_id >= num_iters:
				break
			data_time.update(time.time() - end)

			for k in batch:
				if k != 'meta':
					batch[k] = batch[k].to(device=self.device, non_blocking=True)  

			output, loss, loss_stats = model_with_loss(batch)
			loss = loss.mean()
			if phase == 'train':
				self.optimizer.zero_grad()
				loss.backward()
				self.optimizer.step()
			batch_time.update(time.time() - end)
			end = time.time()

			Bar.suffix = '{phase}: [{0}][{1}/{2}]|Tot: {total:} |ETA: {eta:} '.format(
				epoch, iter_id, num_iters, phase=phase,
				total=bar.elapsed_td, eta=bar.eta_td)
			for l in avg_loss_stats:
				avg_loss_stats[l].update(
					loss_stats[l].mean().item(), batch['image'].size(0))
				Bar.suffix = Bar.suffix + '|{} {:.4f} '.format(l, avg_loss_stats[l].avg)
			Bar.suffix = Bar.suffix + '|Data {dt.val:.3f}s({dt.avg:.3f}s) ' \
				'|Net {bt.avg:.3f}s'.format(dt=data_time, bt=batch_time)
		
			bar.next()
			
			del output, loss, loss_stats
		
		bar.finish()
		ret = {k: v.avg for k, v in avg_loss_stats.items()}
		ret['time'] = bar.elapsed_td.total_seconds() / 60.
		return ret

	def _get_losses(self, opt):
		loss_order = ['hm', 'wh', 'reg', 'dep', 'tracking']
		loss_states = ['tot'] + [k for k in loss_order if k in opt.heads]
		loss = GenericLoss(opt)
		return loss_states, loss

	def train(self):
		opt = self.opt
		logging.basicConfig(filename = "algorithms/" + opt.algorithm + "/results/logs/" + opt.exp_name + "_" + opt.exp_id + '.log', filemode = 'w', level = logging.INFO)
		self.writer = self.set_writer(log_dir = "algorithms/" + self.opt.algorithm + "/results/tensorboards/" + self.opt.exp_name + "_" + opt.exp_id + "/")
		for epoch in range(self.start_epoch + 1, opt.epochs + 1):
			log_dict_train = self.run_epoch('train', epoch, self.train_loader)

			str_out = ""            
			for k, v in log_dict_train.items():
				if k != "time":
					self.writer["train"].add_scalar('Loss/' + k, v, epoch)
					str_out += k + ":" + str(round(v, 8)) + "\t"

			logging.info('Train set: Epoch: [{}/{}]\tLosses: {}'.format(epoch, opt.epochs, str_out))
			save_model(opt.checkpoint_name, epoch, self.model, self.optimizer)
			
			if opt.val_intervals > 0 and epoch % opt.val_intervals == 0:
				with torch.no_grad():
					log_dict_val = self.run_epoch('val', epoch, self.val_loader)

				str_out = ""            
				for k, v in log_dict_val.items():
					if k != "time":
						self.writer["val"].add_scalar('Loss/' + k, v, epoch)
						str_out += k + ":" + str(round(v, 8)) + "\t"
						
				logging.info('Val set: Epoch: [{}/{}]\tLosses: {}'.format(epoch, opt.epochs, str_out))
		
			if epoch in opt.lr_step:
				lr = opt.learning_rate * (0.1 ** (opt.lr_step.index(epoch) + 1))
				print('Drop LR to', lr)
				for param_group in self.optimizer.param_groups:
					param_group['lr'] = lr

	def test(self):
		with torch.no_grad():
			log_dict_test = self.run_epoch('test', self.opt.epochs, self.test_loader)
		
		str_out = ""            
		for k, v in log_dict_test.items():
			str_out += k + ":" + str(round(v, 8)) + "\t"
		print('Test set: Epoch: [{}/{}]\tLosses: {}'.format(self.opt.epochs, self.opt.epochs, str_out))