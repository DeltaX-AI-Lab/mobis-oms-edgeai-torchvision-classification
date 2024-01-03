import os
import sys
import onnx
import copy
import math
import time
import shutil
import random
import datetime
import progiter
import warnings
import numpy as np
import torchmetrics
import pandas as pd
import seaborn as sn
import edgeai_torchmodelopt
import matplotlib.pyplot as plt

import torch
import torchvision
import torch.optim
import torch.utils.data
import torch.nn.parallel
import torch.distributed as dist
import torch.utils.data.distributed

from PIL import Image
from colorama import Fore
from onnx import helper
from onnx import TensorProto, shape_inference
from edgeai_xvision import xvision
from edgeai_torchmodelopt import xnn


def get_config():
	""" Main config function to select hyper-parameters """
	args = xnn.utils.ConfigNode()
	args.model_config = xnn.utils.ConfigNode()
	args.dataset_config = xnn.utils.ConfigNode()
	args.model_config.input_channels = 3				# num input channels
	args.model_config.output_type = 'classification'
	args.model_config.output_channels = None
	args.model_config.strides = None					# (2,2,2,2,2)
	args.model_config.num_tiles_x = int(1)
	args.model_config.num_tiles_y = int(1)
	args.model_config.en_make_divisible_by8 = True
	args.model_config.enable_fp16 = False				# FP16 half precision mode
	args.model_config.num_classes = None				# number of classes
	args.model_config.cls_names = None					# class names

	args.input_channel_reverse = False					# rgb to bgr
	args.data_path = './data/datasets/ilsvrc'			# path to dataset
	args.model_name = 'mobilenetv2_tv_x1'				# model architecture'
	args.model = None									# if mdoel is crated externaly 
	args.dataset_name = 'imagenet_classification'		# image folder classification
	args.transforms = None								# the transforms itself can be given from outside
	args.save_path = None								# checkpoints save path
	args.phase = 'training'								# training/calibration/validation
	args.date = None									# date to add to save path. if this is None, current date will be added.
	args.exp_name = "EXP_v0"							# add experiment name in the end of folder name

	args.workers = 12									# number of data loading workers (default: 8)
	args.logger = None									# logger stream to output into

	args.epochs = 150									# number of total epochs to run: recommended 100 or 150
	args.warmup_epochs = 5								# number of epochs to warm up by linearly increasing lr
	args.warmup_factor = 1e-3							# max lr allowed for the first epoch during warmup (as a factor of initial lr)
	args.max_skip_epochs_qat = 5						# while doing QTA we might want to skipp a few epochs,
														# since it might not give so much effect on fine-tuning for int-8

	args.epoch_size = 0									# fraction of training epoch to use each time. 0 indicates full
	args.epoch_size_val = 0								# manual epoch size (will match dataset size if not specified)
	args.start_epoch = 0								# manual epoch number to start
	args.stop_epoch = None								# manual epoch number to stop
	args.batch_size = 512								# mini_batch size (default: 256)
	args.total_batch_size = None						# accumulated batch size. total_batch_size = batch_size*iter_size
	args.iter_size = 1									# iteration size. total_batch_size = batch_size*iter_size

	args.lr = 0.1										# initial learning rate
	args.lr_clips = None								# use args.lr itself if it is None
	args.lr_calib = 0.05								# lr for bias calibration
	args.momentum = 0.9									# momentum
	args.weight_decay = 4e-5							# weight decay (default: 1e-4)
	args.bias_decay = None								# bias decay (default: 0.0)

	args.shuffle = True									# shuffle or not
	args.shuffle_val = True								# shuffle val dataset or not

	args.rand_seed = 42									# random seed
	args.print_freq = 10_000							# print frequency (default: 100)
	args.resume = None									# path to latest checkpoint (default: none)
	args.evaluate_start = True							# evaluate right at the begining of training or not
	args.world_size = 1									# number of distributed processes
	args.dist_url = 'tcp://224.66.41.62:23456'			# url used to set up distributed training
	args.dist_backend = 'gloo'							# distributed backend

	args.optimizer = 'sgd'								# optimizer algorithms, choices=['adam','sgd','sgd_nesterov','rmsprop']
	args.scheduler = 'cosine'							# help='scheduler algorithms, choices=['step','poly','exponential', 'cosine']
	args.milestones = 20								# epochs at which learning rate is divided ('step=20')
	args.multistep_gamma = 0.5							# multi step gamma (default: 0.1) ('step=0.5', 'exponential=0.9')
	args.polystep_power = 12.0							# poly step gamma (default: 1.0) ('poly=12')
	args.step_size = 3									# step size for exp lr decay ('exponential=3')
														# NOTE ['step','poly','exponential', 'cosine']
														# 'step'		(args.epochs = 500, args.multistep_gamma = 0.5, args.milestones = 20)
														# 'exponential' (args.epochs = 500, args.multistep_gamma = 0.9, args.step_size = 3)
														# 'poly'		(args.epochs = 500, args.polystep_power = 12)

	args.beta = 0.999									# beta parameter for adam
	args.pretrained = None								# path to pre_trained model
	args.img_resize = 256								# image resize
	args.img_crop = 224									# image crop
	args.rand_scale = (0.2, 1.0)						# random scale range for training
	args.data_augument = 'inception'					# data augumentation method, choices=['inception','resize','adaptive_resize']
	args.auto_augument = None							# one of the auto augument modes defined in torchvision - used in training only
														# currently supported options are: "imagenet", "cifar10", "svhn"
														# according to the paper, auto augument policy from one dataset generalizes to other datasets as well.
	args.random_erasing = 0								# random erasing probability - used in training only

	args.count_flops = True								# count flops and report
	args.save_onnx = True								# apply quantized inference or not
	args.save_pt = True									# save .pt model
	args.print_model = False							# print the model to text
	args.run_soon = True								# Set to false if only cfs files/onnx modelsneeded but no training

	args.multi_color_modes = None						# input modes created with multi color transform
	args.image_mean = (123.675, 116.28, 103.53)			# image mean for input image normalization')
	args.image_scale = (0.017125, 0.017507, 0.017429)	# image scaling/mult for input iamge normalization')

	args.parallel_model = True							# Usedata parallel for model

	args.quantize = False								# apply quantized inference or not
	#args.model_surgery = None							# replace activations with PAct2 activation module. Helpful in quantized training.
	args.bitwidth_weights = 8							# bitwidth for weights
	args.bitwidth_activations = 8						# bitwidth for activations
	args.histogram_range = True							# histogram range for calibration
	args.bias_calibration = True						# apply bias correction during quantized inference calibration
	args.per_channel_q = False							# apply separate quantizion factor for each channel in depthwise or not
	args.constrain_bias = None							# constrain bias according to the constraints of convolution engine

	args.freeze_bn = False								# freeze the statistics of bn
	args.save_mod_files = False							# saves modified files after last commit. Also stores commit id.

	args.opset_version = 11								# onnx opset_version
	return args


def main(args):
	# Warning steps
	assert (not hasattr(args, 'evaluate')), 'args.evaluate is deprecated. use args.phase=training or calibration or validation'
	assert is_valid_phase(args.phase), f'invalid phase {args.phase}'
	assert not hasattr(args, 'model_surgery'), 'the argument model_surgery is deprecated, it is not needed now - remove it'

	if (args.phase == 'validation' and args.bias_calibration):
		args.bias_calibration = False
		warnings.warn('switching off bias calibration in validation')

	#################################################
	# Global settings for repeatability
	random.seed(args.rand_seed)
	np.random.seed(args.rand_seed)
	torch.manual_seed(args.rand_seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False
	# torch.autograd.set_detect_anomaly(True)
	#################################################

	# Define initial variables
	args.best_prec1 = -1
	performance_results = []

	cls_names = sorted(os.listdir(os.path.join(args.data_path, "train")))
	args.model_config.num_classes = len(cls_names)
	args.model_config.cls_names = cls_names
	
	# Get save path for an experiment
	if args.save_path is None:
		args.save_path = get_save_path(args)
	else:
		args.save_path = args.save_path

	# Plot LR scheduler and set new milestones for stepLR and 
	if isinstance(args.milestones, int):
		args.milestones = list(range(args.milestones, args.epochs+1, args.milestones))
	plot_LR_scheduler(args)

	# resume has higher priority
	args.pretrained = None if (args.resume is not None) else args.pretrained

	# Create a folder for the experiment
	if not os.path.exists(args.save_path):
		os.makedirs(args.save_path)

	if args.save_mod_files:
		#store all the files after the last commit.
		mod_files_path = os.path.join(args.save_path, "mod_files")
		os.makedirs(mod_files_path)
		
		cmd = "git ls-files --modified | xargs -i cp {} {}".format("{}", mod_files_path)
		print("cmd:", cmd)
		os.system(cmd)

		#stoe last commit id. 
		cmd = "git log -n 1  >> {}".format(mod_files_path + '/commit_id.txt')
		print("cmd:", cmd)
		os.system(cmd)

	# Define a loger object
	if args.logger is None:
		log_path = os.path.join(args.save_path, os.path.splitext(os.path.basename(__file__))[0] + '.log')
		args.logger = xnn.utils.TeeLogger(filename=log_path)

	# Redefine extra initial variables
	args.pretrained = None if (args.pretrained == 'None') else args.pretrained
	args.num_inputs = len(args.multi_color_modes) if (args.multi_color_modes is not None) else 1

	if args.iter_size != 1 and args.total_batch_size is not None:
		warnings.warn("only one of --iter_size or --total_batch_size must be set")

	if args.total_batch_size is not None:
		args.iter_size = args.total_batch_size // args.batch_size
	else:
		args.total_batch_size = args.batch_size * args.iter_size

	args.stop_epoch = args.stop_epoch if (args.stop_epoch and args.stop_epoch <= args.epochs) else args.epochs
	args.distributed = args.world_size > 1

	if args.distributed:
		dist.init_process_group(backend=args.dist_backend,
								init_method=args.dist_url,
								world_size=args.world_size)

	# Print everything for log, and reset character color, in case it is different
	print('{}'.format(Fore.RESET))
	print("=> args: ", args)
	print('\n'.join("%s: %s" % item for item in sorted(vars(args).items())))
	print("=> resize resolution: {}".format(args.img_resize))
	print("=> crop resolution  : {}".format(args.img_crop))
	sys.stdout.flush()

	# Get pre-trained weights
	#################################################
	pretrained_data = None
	model_surgery_quantize = False
	if args.pretrained and args.pretrained != "None":
		if args.pretrained.startswith('http://') or args.pretrained.startswith('https://'):
			pretrained_file = xnn.utils.download_url(args.pretrained, './data/downloads')
		else:
			pretrained_file = args.pretrained

		print(f'=> using pre-trained weights from: {args.pretrained}')
		pretrained_data = torch.load(pretrained_file)
		model_surgery_quantize = pretrained_data['quantize'] if 'quantize' in pretrained_data else False

	#################################################

	# Create model
	print("=> creating model '{}'".format(args.model_name))

	is_onnx_model = False
	if isinstance(args.model, torch.nn.Module):
		model = args.model
	elif isinstance(args.model, str) and args.model.endswith('.onnx'):
		model = xnn.onnx.import_onnx(args.model)
		is_onnx_model = True
	else:
		model = xvision.models.__dict__[args.model_name](args.model_config)

	# check if we got the model as well as parameters to change the names in pretrained
	model, change_names_dict = model if isinstance(model, (list, tuple)) else (model, None)

	# Apply quantization
	#################################################
	if args.quantize:
		# dummy input is used by quantized models to analyze graph
		is_cuda = next(model.parameters()).is_cuda
		dummy_input = create_rand_inputs(args, is_cuda=is_cuda)

		if 'training' in args.phase:
			model = edgeai_torchmodelopt.xmodelopt.quantization.v1.QuantTrainModule(model, per_channel_q=args.per_channel_q,
						bitwidth_weights=args.bitwidth_weights, bitwidth_activations=args.bitwidth_activations,
						histogram_range=args.histogram_range, constrain_bias=args.constrain_bias,
						dummy_input=dummy_input, total_epochs=args.epochs)
		elif 'calibration' in args.phase:
			model = edgeai_torchmodelopt.xmodelopt.quantization.v1.QuantCalibrateModule(model, per_channel_q=args.per_channel_q,
						bitwidth_weights=args.bitwidth_weights, bitwidth_activations=args.bitwidth_activations,
						histogram_range=args.histogram_range, constrain_bias=args.constrain_bias,
						bias_calibration=args.bias_calibration, dummy_input=dummy_input,
						lr_calib=args.lr_calib)
		elif 'validation' in args.phase:
			# Note: bias_calibration is not used in test
			model = edgeai_torchmodelopt.xmodelopt.quantization.v1.QuantTestModule(model, per_channel_q=args.per_channel_q,
						bitwidth_weights=args.bitwidth_weights, bitwidth_activations=args.bitwidth_activations,
						histogram_range=args.histogram_range, dummy_input=dummy_input,
						model_surgery_quantize=model_surgery_quantize)
		else:
			assert False, f'invalid phase {args.phase}'

	# Load pre-trained weights
	if pretrained_data is not None and not is_onnx_model:
		model_orig = get_model_orig(model)
		if hasattr(model_orig, 'load_weights'):
			model_orig.load_weights(pretrained=pretrained_data, change_names_dict=change_names_dict)
		else:
			xnn.utils.load_weights(model_orig, pretrained=pretrained_data, change_names_dict=change_names_dict)
	
	#################################################
	if args.count_flops:
		count_flops(args, model)

	#################################################
	if args.save_onnx:
		write_onnx_model(args, get_model_orig(model))

	#################################################		# TODO MAX
	if args.save_pt:
		write_pt_model(args, get_model_orig(model))
		
	#################################################
	if args.print_model:
		print(model)
	else:
		args.logger.debug(str(model))

	#################################################
	if (not args.run_soon):
		print("Training not needed for now")
		close(args)
		exit()

	#################################################
	# DataParallel does not work for QuantCalibrateModule or QuantTestModule
	if args.parallel_model and (not isinstance(model, (edgeai_torchmodelopt.xmodelopt.quantization.v1.QuantCalibrateModule, edgeai_torchmodelopt.xmodelopt.quantization.v1.QuantTestModule))):
		if args.distributed:
			model = torch.nn.parallel.DistributedDataParallel(model)
		else:
			model = torch.nn.DataParallel(model)

	#################################################
	model = model.cuda()

	#################################################
	# define loss function (criterion) and optimizer
	criterion = torch.nn.CrossEntropyLoss().cuda()

	model_module = model.module if hasattr(model, 'module') else model
	if args.lr_clips is not None:
		learning_rate_clips = args.lr_clips if 'training' in args.phase else 0.0
		clips_decay = args.bias_decay if (args.bias_decay is not None and args.bias_decay != 0.0) else args.weight_decay
		clips_params = [p for n,p in model_module.named_parameters() if 'clips' in n]
		other_params = [p for n,p in model_module.named_parameters() if 'clips' not in n]
		param_groups = [{'params': clips_params, 'weight_decay': clips_decay, 'lr': learning_rate_clips},
						{'params': other_params, 'weight_decay': args.weight_decay}]
	else:
		param_groups = [{'params': filter(lambda p: p.requires_grad, model.parameters()), 'weight_decay': args.weight_decay}]

	print("=> args: ", args)
	print("=> optimizer type   : {}".format(args.optimizer))
	print("=> learning rate    : {}".format(args.lr))
	print("=> resize resolution: {}".format(args.img_resize))
	print("=> crop resolution  : {}".format(args.img_crop))
	print("=> batch size       : {}".format(args.batch_size))
	print("=> total batch size : {}".format(args.total_batch_size))
	print("=> epoch size       : {}".format(args.epoch_size))
	print("=> data augument    : {}".format(args.data_augument))
	print("=> epochs           : {}".format(args.epochs))
	if args.scheduler == 'step':
		print("=> milestones   : {}".format(args.milestones))

	learning_rate = args.lr if ('training'in args.phase) else 0.0
	if args.optimizer == 'adam':
		optimizer = torch.optim.Adam(param_groups, learning_rate, betas=(args.momentum, args.beta))
	elif args.optimizer == 'sgd':
		optimizer = torch.optim.SGD(param_groups, learning_rate, momentum=args.momentum)
	elif args.optimizer == 'sgd_nesterov':
		optimizer = torch.optim.SGD(param_groups, learning_rate, momentum=args.momentum, nesterov=True)
	elif args.optimizer == 'rmsprop':
		optimizer = torch.optim.RMSprop(param_groups, learning_rate, momentum=args.momentum)
	else:
		raise ValueError('Unknown optimizer type{}'.format(args.optimizer))
	
	# optionally resume from a checkpoint
	if args.resume:
		if os.path.isfile(args.resume):
			print("=> resuming from checkpoint '{}'".format(args.resume))
			checkpoint = torch.load(args.resume)
			if args.start_epoch == 0:
				args.start_epoch = checkpoint['epoch'] + 1
				
			args.best_prec1 = checkpoint['best_prec1']
			model = xnn.utils.load_weights(model, checkpoint)
			optimizer.load_state_dict(checkpoint['optimizer'])
			print("=> resuming from checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
		else:
			print("=> no checkpoint found at '{}'".format(args.resume))

	train_loader, val_loader = get_data_loaders(args)

	args.cur_lr = adjust_learning_rate(args, optimizer, args.start_epoch)

	if args.evaluate_start or args.phase=='validation':
		validate(args, val_loader, model, criterion, args.start_epoch)

	if args.phase == 'validation':
		close(args)
		return

	grad_scaler = torch.cuda.amp.GradScaler() if args.model_config.enable_fp16 else None
	
	training_time_start = time.time()
	
	# ============ Train stage --> epoch ============
	for epoch in range(args.start_epoch, args.stop_epoch):
		if args.distributed:
			train_loader.sampler.set_epoch(epoch)

		# One epoch training
		train_acc = train(args, train_loader, model, criterion, optimizer, epoch, grad_scaler)

		# Evaluate model on validation set
		val_acc = validate(args, val_loader, model, criterion, epoch)

		# Save results
		performance_results.append((epoch+1, float(train_acc), float(val_acc)))

		# remember best prec@1 and save checkpoint
		if args.quantize and max(args.warmup_epochs, args.max_skip_epochs_qat) >= epoch:	# TODO
			is_best = False
			args.best_prec1 = max(val_acc, args.best_prec1)
		else:
			is_best = val_acc > args.best_prec1
			args.best_prec1 = max(val_acc, args.best_prec1)

		model_orig = get_model_orig(model)

		save_dict = {
			'epoch': epoch,
			'arch': args.model_name,
			'state_dict': model_orig.state_dict(),
			'optimizer': optimizer.state_dict(),
			'best_prec1': args.best_prec1,
			'quantize': args.quantize,
			}
		
		save_checkpoint(args, model_orig, save_dict, is_best)
		try:
			plot_performance_results(args, performance_results)
		except Exception as e:
			print(f"An error occurred in plot_performance_results: {e}")

	training_time_end = time.time()
	print(f"\n\nTraining time: {int((training_time_end - training_time_start) / 3600.0)} ours!\n\n")

	# Close and clean-up
	close(args)


############################################################################ START: custom functions TODO MAX

def get_adjust_learning_rate(args, epoch):
	"""
	choices=['step','poly','exponential', 'cosine']
	cosine:			good, but decay lr to slow
	step:			good, you have to manually specify milestones
	exponential:	good, looks like step with auto milestones (better then step?)
	poly:			good, looks like smoothed exponential (better then exponential?)

	step 		(args.epochs = 500, args.multistep_gamma = 0.5, args.milestones_step = 20)
	exponential (args.epochs = 500, args.multistep_gamma = 0.9, args.step_size = 3)
	poly 		(args.epochs = 500, args.polystep_power = 12)
	"""
	cur_lr = args.lr

	if args.warmup_epochs and (epoch <= args.warmup_epochs):
		cur_lr = epoch * args.lr / args.warmup_epochs
		if epoch == 0 and args.warmup_factor is not None:
			cur_lr = max(cur_lr, args.lr * args.warmup_factor)

	elif args.scheduler == 'poly':
		epoch_frac = (args.epochs - epoch) / args.epochs
		epoch_frac = max(epoch_frac, 0)
		cur_lr = args.lr * (epoch_frac ** args.polystep_power)

	elif args.scheduler == 'step':											# step
		num_milestones = 0
		for m in args.milestones:
			num_milestones += (1 if epoch >= m else 0)
		cur_lr = args.lr * (args.multistep_gamma ** num_milestones)

	elif args.scheduler == 'exponential':									# exponential
		cur_lr = args.lr * (args.multistep_gamma ** (epoch//args.step_size))

	elif args.scheduler == 'cosine':										# cosine
		if epoch == 0:
			cur_lr = args.lr
		else:
			lr_min = 0
			cur_lr = (args.lr - lr_min) * (1 + math.cos(math.pi * epoch / args.epochs)) / 2.0 + lr_min

	return cur_lr


def plot_LR_scheduler(args, show=False):
	""" Visualize LR-scheduler """
	svg_path = os.path.join(args.save_path, f"lr_scheduler_{args.scheduler}.svg")

	if not os.path.exists(args.save_path):
		os.makedirs(args.save_path)

	lr_steps = list()
	epochs = list(range(args.epochs))

	for i, epoch in enumerate(epochs):
		cur_lr = get_adjust_learning_rate(args, epoch)
		lr_steps.append(cur_lr)
		if show: print(f"{i}: {cur_lr}")

	# Initialize the plot
	plt.figure(figsize=(20, 12))
	plt.title(f'LR scheduler: {args.scheduler}', fontsize=20)
	plt.xlabel('Epoch', fontsize=16)
	plt.ylabel('LR', fontsize=16)

	# Plot the training and validation accuracies
	plt.plot(epochs, lr_steps, label='LR scheduler', marker='o')

	# Show grid, legend, and labels
	plt.grid(True)
	plt.legend()
	plt.savefig(svg_path)
	plt.close()


def plot_performance_results(args, performance_results):
	""" Plot performance results for train and validation accuracy """
	if 'training' in args.phase and args.quantize:
		exp_phase = "training_quantizing"
	elif 'training' in args.phase and not args.quantize:
		exp_phase = "training"
	elif args.quantize:
		exp_phase = "PTQ_quantizing"

	save_path_split = args.save_path.split("/")
	image_folder_classification_path = '/'.join(save_path_split[:-2])

	exp_name = save_path_split[-2]
	svg_folder_path = os.path.join(image_folder_classification_path, "svg_backup", exp_phase)
	svg_extra_path = os.path.join(svg_folder_path, f"{exp_name}.svg")
	svg_path = os.path.join(args.save_path, "performance_results.svg")
	

	if not os.path.exists(svg_folder_path):
		os.makedirs(svg_folder_path)

	# Extract epoch numbers and accuracies for training and validation
	epochs, train_accuracies, val_accuracies = zip(*performance_results)

	best_train_accuracy = max(train_accuracies)
	best_val_accuracy = max(val_accuracies)
	stop_epoch = val_accuracies.index(best_val_accuracy)
	train_stop_accuracy = train_accuracies[stop_epoch]
	stop_epoch += 1

	results = "Epoch={0}, TM-Acc={1:.3f}, T-Acc={2:.3f}, V-Acc={3:.3f}".format(
		stop_epoch, best_train_accuracy, train_stop_accuracy, best_val_accuracy)
	
	# Initialize the plot
	plt.figure(figsize=(20, 12))
	plt.title(f'Training and Validation Accuracy Over Epochs: {results}', fontsize=20)
	plt.xlabel('Epoch', fontsize=16)
	plt.ylabel('Accuracy (%)', fontsize=16)

	# Plot the training and validation accuracies
	plt.plot(epochs, train_accuracies, label='Training Accuracy', marker='o')
	plt.plot(epochs, val_accuracies, label='Validation Accuracy', marker='o')

	plt.scatter(x=stop_epoch, y=best_val_accuracy, s=50, color='k', zorder=5.5)
	plt.axhline(y=best_val_accuracy, color='k', linestyle='--', linewidth=2.0)
	plt.axvline(x=stop_epoch, color='k', linestyle='--', linewidth=2.0)

	# Show grid, legend, and labels
	plt.grid(True)
	plt.legend()
	plt.savefig(svg_path)
	plt.savefig(svg_extra_path)
	plt.close()


def plot_confusion_matrix(args, conf_matrix, cls_names):
	svg_path = os.path.join(args.save_path, "confusion_matrix_results.svg")

	# Transform to data frame
	df_conf_matrix = pd.DataFrame(data=conf_matrix, index=cls_names, columns=cls_names)

	# Show the confusion matrix
	print(f"\nConfusionMatrix:\n{df_conf_matrix}")

	# Set visualization
	plt.figure(figsize=(35, 25))
	sn.set(font_scale=1.6)
	sn.heatmap(df_conf_matrix, annot=True, annot_kws={"size": 18}, cmap="Greens", fmt='g')

	plt.title("Confusion matrix", fontsize=35)
	plt.xlabel("Predicted label", fontsize=30)
	plt.ylabel("True label", fontsize=30)
	plt.savefig(svg_path)
	plt.close()
	
	# Reset to default parameters
	sn.reset_defaults()


def tidlOnnxModelOptimize(args, input_name, output_name):
	# Get extra info from args adn paths
	meanList = args.image_mean
	scaleList = args.image_scale
	opset_version = args.opset_version
	in_model_path = os.path.join(args.save_path, input_name)
	out_model_path = os.path.join(args.save_path, output_name)

	#Read Model
	meanList = [x * -1 for x in meanList]
	model = onnx.load_model(in_model_path)
	op = onnx.OperatorSetIdProto()
	#Track orginal opset:
	op.version = model.opset_import[0].version
	#Get Graph:
	originalGraph = model.graph
	#Get Nodes:
	originalNodes = originalGraph.node
	#Get Initializers:
	originalInitializers = originalGraph.initializer
	#Create Lists
	nodeList = [node for node in originalNodes]
	initList = [init for init in originalInitializers]

	nInCh = int(originalGraph.input[0].type.tensor_type.shape.dim[1].dim_value)

	#Input & Output Dimensions:
	inDims = tuple([x.dim_value for x in originalGraph.input[0].type.tensor_type.shape.dim])
	outDims = tuple([x.dim_value for x in originalGraph.output[0].type.tensor_type.shape.dim])

	#Construct bias & scale tensors
	biasTensor = helper.make_tensor("TIDL_preProc_Bias", TensorProto.FLOAT, [1,nInCh, 1, 1], np.array(meanList,dtype=np.float32))
	scaleTensor = helper.make_tensor("TIDL_preProc_Scale", TensorProto.FLOAT, [1, nInCh, 1, 1], np.array(scaleList,dtype=np.float32))

	#Add these tensors to initList:
	initList.append(biasTensor)
	initList.append(scaleTensor)

	#Cast Node:
	attrib_dict = {"to":TensorProto.FLOAT}
	cast = onnx.helper.make_node('Cast', inputs=[originalGraph.input[0].name+"Net_IN"], outputs=['TIDL_cast_in'], **attrib_dict)

	#Add Node:
	addNode = onnx.helper.make_node('Add', inputs=["TIDL_cast_in","TIDL_preProc_Bias"], outputs=["TIDL_Scale_In"])

	#Scale Node:
	scaleNode = onnx.helper.make_node('Mul', inputs=["TIDL_Scale_In","TIDL_preProc_Scale"], outputs=[originalGraph.input[0].name]) #Assumption that input[0].name is the input node

	nodeList = [cast, addNode, scaleNode] + nodeList #Toplogically Sorted

	outSequence = originalGraph.output
	#Check for Argmax:
	for node in nodeList:
		if node.op_type == "ArgMax":
			#Check if it is final output:
			if node.output[0] == originalGraph.output[0].name:
				#Argmax Output is final output:
				attrib_dict_1 = {"to":TensorProto.UINT8}
				cast_out = onnx.helper.make_node('Cast', inputs=[originalGraph.output[0].name], outputs=[originalGraph.output[0].name+'TIDL_cast_out'], **attrib_dict_1)
				nodeList = nodeList + [cast_out] #Toplogically Sorted
				outSequence = [helper.make_tensor_value_info(originalGraph.output[0].name+'TIDL_cast_out', TensorProto.UINT8, outDims)]

	#Construct Graph:
	newGraph = helper.make_graph(
		nodeList,
		'Rev_Model',
		[helper.make_tensor_value_info(originalGraph.input[0].name+"Net_IN", TensorProto.UINT8, inDims)],
		outSequence,
		initList
		)
	#Construct Model:
	op.version = opset_version
	model_def_noShape = helper.make_model(newGraph, producer_name='onnx-TIDL', opset_imports=[op])
	model_def = shape_inference.infer_shapes(model_def_noShape)

	try:
		onnx.checker.check_model(model_def)
	except onnx.checker.ValidationError as e:
		print('Converted model is invalid: %s' % e)
	else:
		print('Converted model is valid!')
		onnx.save_model(model_def, out_model_path)
	print("*** Quantisation Succeeded ***")
	

############################################################################ END: custom functions TODO MAX
def is_valid_phase(phase):
	phases = ('training', 'calibration', 'validation')
	return any(p in phase for p in phases)


def close(args):
	if args.logger is not None:
		args.logger.close()
		del args.logger
		args.logger = None
	#
	args.best_prec1 = -1


def get_save_path(args, phase=None):
	date = args.date if args.date else datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
	save_path_base = os.path.join('./data/checkpoints/edgeailite', args.dataset_name, date + '_' + args.dataset_name + '_' + args.model_name)
	save_path = save_path_base + '_resize{}_crop{}_{}'.format(args.img_resize, args.img_crop, args.exp_name)
	phase = phase if (phase is not None) else args.phase
	save_path = os.path.join(save_path, phase)
	return save_path


def get_model_orig(model):
	is_parallel_model = isinstance(model, (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel))
	model_orig = (model.module if is_parallel_model else model)
	model_orig = (model_orig.module if isinstance(model_orig, (edgeai_torchmodelopt.xmodelopt.quantization.v1.QuantBaseModule)) else model_orig)
	return model_orig


def create_rand_inputs(args, is_cuda):
	dummy_input = torch.rand((1, args.model_config.input_channels, args.img_crop*args.model_config.num_tiles_y,
						args.img_crop*args.model_config.num_tiles_x))
	dummy_input = dummy_input.cuda() if is_cuda else dummy_input
	return dummy_input


def count_flops(args, model):
	is_cuda = next(model.parameters()).is_cuda
	dummy_input = create_rand_inputs(args, is_cuda)
	model.eval()
	total_mult_adds, total_params = xnn.utils.get_model_complexity(model, dummy_input)
	total_mult_adds_giga = total_mult_adds/1e9
	total_flops = total_mult_adds_giga*2
	total_params_mega = total_params/1e6
	print('=> Resize = {}, Crop = {}, GFLOPs = {}, GMACs = {}, MegaParams = {}'.format(args.img_resize, args.img_crop, total_flops, total_mult_adds_giga, total_params_mega))


def write_onnx_model(args, model, name='checkpoint.onnx'):
	filepath = os.path.join(args.save_path, name)
	is_cuda = next(model.parameters()).is_cuda
	dummy_input = create_rand_inputs(args, is_cuda)

	model.eval()
	torch.onnx.export(model, dummy_input, filepath, export_params=True, verbose=False,
					  do_constant_folding=True, opset_version=args.opset_version)
	# infer shapes
	onnx.shape_inference.infer_shapes_path(filepath, filepath)
	# export torchscript model
	traced_model = torch.jit.trace(model, dummy_input)
	torch.jit.save(traced_model, os.path.splitext(filepath)[0]+'_jit_traced.pth')
	print('torchscript_jit_traced.pth export done.')


def write_pt_model(args, model, name='checkpoint.pt'):
	filepath = os.path.join(args.save_path, name)
	state = {'model': copy.deepcopy(model.eval())}
	torch.save(state, filepath)
	print(f'{name} export done.')


def train(args, train_loader, model, criterion, optimizer, epoch, grad_scaler):
	# actual training code
	batch_time = AverageMeter()
	data_time = AverageMeter()
	losses = AverageMeter()
	top1 = AverageMeter()
	topK = AverageMeter()

	# switch to train mode
	model.train()

	num_iters = len(train_loader)
	progress_bar = progiter.ProgIter(np.arange(num_iters), chunksize=1)
	args.cur_lr = adjust_learning_rate(args, optimizer, epoch)

	end = time.time()
	last_update_iter = -1

	progressbar_color = (Fore.YELLOW if (('calibration' in args.phase) or ('training' in args.phase and args.quantize)) else Fore.WHITE)
	print('{}'.format(progressbar_color), end='')

	for iteration, (input, target) in enumerate(train_loader):
		input = [inp.cuda() for inp in input] if xnn.utils.is_list(input) else input.cuda()
		input_size = input[0].size() if xnn.utils.is_list(input) else input.size()
		target = target.cuda(non_blocking=True)

		data_time.update(time.time() - end)

		# preprocess to make tiles
		if args.model_config.num_tiles_y > 1 or args.model_config.num_tiles_x > 1:
			input = xnn.utils.reshape_input_4d(input, args.model_config.num_tiles_y, args.model_config.num_tiles_x)

		# compute output
		output = model(input)

		if args.model_config.num_tiles_y > 1 or args.model_config.num_tiles_x > 1:
			# [1, n_class, n_tiles_y, n_tiles_x] to [1, n_tiles_y, n_tiles_x, n_class]
			# e.g. [1,10,4,5] to [1,4,5,10]
			output = output.permute(0, 2, 3, 1)
			# change shape from [1, n_tiles_y, n_tiles_x, n_class] to [1 * n_tiles_y * n_tiles_x, n_class]
			output = torch.reshape(output, (-1, output.shape[-1]))

		# compute loss
		loss = criterion(output, target) / args.iter_size

		# measure accuracy and record loss
		topk_cat = min(2, args.model_config.num_classes)
		prec1, precK = accuracy(output, target, topk=(1, topk_cat))
		losses.update(loss.item(), input_size[0])
		top1.update(prec1[0], input_size[0])
		topK.update(precK[0], input_size[0])

		if 'training' in args.phase:
			if args.model_config.enable_fp16:
				grad_scaler.scale(loss).backward()
			else:
				loss.backward()

			if ((iteration+1) % args.iter_size) == 0:
				if args.model_config.enable_fp16:
					grad_scaler.step(optimizer)
					grad_scaler.update()
				else:
					optimizer.step()

				# setting grad=None is a faster alternative instead of optimizer.zero_grad()
				xnn.utils.clear_grad(model)

		# measure elapsed time
		batch_time.update(time.time() - end)
		end = time.time()
		final_iter = (iteration >= (num_iters-1))

		if ((iteration % args.print_freq) == 0) or final_iter:
			epoch_str = "{}/{}".format(epoch + 1, args.epochs)
			status_str = "{epoch} LR={cur_lr:.8f} Time={batch_time.avg:0.3f} Loss={loss.avg:0.3f} Prec@1={top1.avg:0.3f} Prec@{topk_cat}={topK.avg:0.3f}" \
						 .format(epoch=epoch_str, cur_lr=args.cur_lr, batch_time=batch_time, loss=losses, top1=top1, topK=topK, topk_cat=topk_cat)

			progress_bar.set_description(f"=> {args.phase}")
			progress_bar.set_postfix(dict(Epoch='{}'.format(status_str)))
			progress_bar.update(iteration-last_update_iter)
			last_update_iter = iteration

	progress_bar.close()

	# to print a new line - do not provide end=''
	print('{}'.format(Fore.RESET), end='')

	##########################
	if args.quantize:
		def debug_format(v):
			return ('{:.3f}'.format(v) if v is not None else 'None')

		clips_act = [m.get_clips_act()[1] for n, m in model.named_modules() if isinstance(m, xnn.layers.PAct2)]
		if len(clips_act) > 0:
			args.logger.debug('\nclips_act : ' + ' '.join(map(debug_format, clips_act)))
			args.logger.debug('')

	return top1.avg


def validate(args, val_loader, model, criterion, epoch):
	batch_time = AverageMeter()
	losses = AverageMeter()
	top1 = AverageMeter()
	topK = AverageMeter()

	TorchMetricsObjectMicro = TorchMetrics(	# TODO
		task="binary" if args.model_config.num_classes==1 else "multiclass",
		num_classes=args.model_config.num_classes,
		cls_names=args.model_config.cls_names,
		average="micro",
		normalize="none",
		threshold=0.5,
		top_k=1,
		device="cuda",
		show_parameters=True if epoch==0 else False,
	)
	TorchMetricsObjectMacro = TorchMetrics(	# TODO
		task="binary" if args.model_config.num_classes==1 else "multiclass",
		num_classes=args.model_config.num_classes,
		cls_names=args.model_config.cls_names,
		average="macro",
		normalize="none",
		threshold=0.5,
		top_k=1,
		device="cuda",
		show_parameters=True if epoch==0 else False,
	)
	TorchMetricsObjectWeighted = TorchMetrics(	# TODO
		task="binary" if args.model_config.num_classes==1 else "multiclass",
		num_classes=args.model_config.num_classes,
		cls_names=args.model_config.cls_names,
		average="weighted",
		normalize="none",
		threshold=0.5,
		top_k=1,
		device="cuda",
		show_parameters=True if epoch==0 else False,
	)

	# switch to evaluate mode
	model.eval()

	num_iters = len(val_loader)
	progress_bar = progiter.ProgIter(np.arange(num_iters), chunksize=1)
	last_update_iter = -1

	# change color to green
	print('{}'.format(Fore.GREEN), end='')

	with torch.no_grad():
		end = time.time()
		for iteration, (input, target) in enumerate(val_loader):
			input = [inp.cuda() for inp in input] if xnn.utils.is_list(input) else input.cuda()
			input_size = input[0].size() if xnn.utils.is_list(input) else input.size()
			target = target.cuda(non_blocking=True)

			# preprocess to make tiles
			if args.model_config.num_tiles_y > 1 or args.model_config.num_tiles_x > 1:
				input = xnn.utils.reshape_input_4d(input, args.model_config.num_tiles_y, args.model_config.num_tiles_x)

			# compute output
			output = model(input)

			if args.model_config.num_tiles_y > 1 or args.model_config.num_tiles_x > 1:
				# [1, n_class, n_tiles_y, n_tiles_x] to [1, n_tiles_y, n_tiles_x, n_class] 
				# e.g. [1,10,4,5] to [1,4,5,10]
				output = output.permute(0, 2, 3, 1)
				# change shape from [1, n_tiles_y, n_tiles_x, n_class] to [1 * n_tiles_y * n_tiles_x, n_class]
				output = torch.reshape(output, (-1, output.shape[-1]))

			loss = criterion(output, target)

			# measure accuracy and record loss
			topk_cat = min(2, args.model_config.num_classes)
			prec1, precK = accuracy(output, target, topk=(1, topk_cat))
			losses.update(loss.item(), input_size[0])
			top1.update(prec1[0], input_size[0])
			topK.update(precK[0], input_size[0])
			TorchMetricsObjectMicro.update(output, target)
			TorchMetricsObjectMacro.update(output, target)
			TorchMetricsObjectWeighted.update(output, target)

			# measure elapsed time
			batch_time.update(time.time() - end)
			end = time.time()
			final_iter = (iteration >= (num_iters-1))

			if ((iteration % args.print_freq) == 0) or final_iter:
				epoch_str = "{}/{}".format(epoch+1, args.epochs)
				status_str = "{epoch} LR={cur_lr:.8f} Time={batch_time.avg:0.3f} Loss={loss.avg:0.3f} Prec@1={top1.avg:0.3f} Prec@{topk_cat}={topK.avg:0.3f}" \
							 .format(epoch=epoch_str, cur_lr=args.cur_lr, batch_time=batch_time, loss=losses, top1=top1, topK=topK, topk_cat=topk_cat)
			
				prefix = '**' if final_iter else '=>'
				progress_bar.set_description("{} {}".format(prefix, 'validation'))
				progress_bar.set_postfix(dict(Epoch='{}'.format(status_str)))
				progress_bar.update(iteration - last_update_iter)
				last_update_iter = iteration

		progress_bar.close()

		# to print a new line - do not provide end=''
		print('{}'.format(Fore.RESET), end='')

	# Metric on all batches using custom accumulation
	TorchMetricsObjectMicro.compute(args, show=True)
	TorchMetricsObjectMacro.compute(args, show=True)
	TorchMetricsObjectWeighted.compute(args, show=True)
	# Resetting internal state such that metric ready for new data
	TorchMetricsObjectMicro.reset()
	TorchMetricsObjectMacro.reset()
	TorchMetricsObjectWeighted.reset()

	return top1.avg


def save_checkpoint(args, model, state, is_best, filename='checkpoint.pth'):
	filename = os.path.join(args.save_path, filename)
	torch.save(state, filename)

	if is_best:
		bestname = os.path.join(args.save_path, 'model_best.pth')
		shutil.copyfile(filename, bestname)

	if args.save_onnx:
		write_pt_model(args, model, name='checkpoint.pt')
		write_onnx_model(args, model, name='checkpoint.onnx')
		if is_best:
			write_pt_model(args, model, name='model_best.pt')
			write_onnx_model(args, model, name='model_best.onnx')
			tidlOnnxModelOptimize(args, input_name="model_best.onnx", output_name="model_best_quantized.onnx")


class AverageMeter(object):
	""" Computes and stores the average and current value """
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


class TorchMetrics:
	""" TorchMetrics class combined metrics wrappers.
	Check: https://torchmetrics.readthedocs.io/en/stable/classification/accuracy.html

	Args:
		task: Literal["binary", "multiclass", "multilabel"],
		num_classes: Optional[int] = None,
		average: Optional[Literal["micro", "macro", "weighted", "none"]] = "micro",
		normalize: Optional[Literal["true", "pred", "all", "none"]] = None,
		multidim_average: Literal["global", "samplewise"] = "global",
		threshold: float = 0.5,
		top_k: Optional[int] = 1,
		device: Optional[Literal["cpu", "cuda"]] = "cuda",

	Note:
		Macro Averaging: Calculate the metric independently for each class and then take the average.
			This method treats all classes equally, giving equal weight to each class, regardless of how many instances of that class are in your dataset.
		Micro Averaging: Calculate the metric globally by counting the total true positives, false negatives, and false positives.
			This method can be preferable when you have class imbalance because it weights each instance equally.
		Weighted Averaging: Calculate metrics for each class, and find their average weighted by the number of true instances for each class.
			This approach accounts for class imbalance by giving more weight to the more frequent classes.

	Returns:
		result (float or 2D-list): accuracy, or ConfusionMatrix
	"""

	def __init__(self,
			task="binary",
			average="micro",
			normalize="none",
			num_classes=None,
			cls_names=["0_off", "1_on"],
			threshold=0.5,
			top_k=1,
			device="cuda",
			show_parameters=True):
		self.device = device
		self.cls_names = cls_names
		self.show_parameters = show_parameters
		
		self.parameters = {
			"task": task,
			"num_classes": num_classes,
			"threshold": threshold,
			"average": average,
			"top_k": top_k,
		}

		self.CMparameters = {
			"task": task,
			"num_classes": num_classes,
			"threshold": threshold,
			"normalize": normalize
		}

		if self.show_parameters:
			print(self.cls_names)
			print({**self.parameters, **self.CMparameters})

		self.accuracy = torchmetrics.Accuracy(**self.parameters).to(self.device)
		self.precision = torchmetrics.Precision(**self.parameters).to(self.device)
		self.recall = torchmetrics.Recall(**self.parameters).to(self.device)
		self.F1Score = torchmetrics.F1Score(**self.parameters).to(self.device)
		self.confusionMatrix = torchmetrics.ConfusionMatrix(**self.CMparameters).to(self.device)

	def reset(self):
		""" Resetting internal state such that metric ready for new data """
		self.accuracy.reset()
		self.precision.reset()
		self.recall.reset()
		self.F1Score.reset()
		self.confusionMatrix.reset()

	def update(self, output, target):
		""" Update metrics """
		self.accuracy(output, target)
		self.precision(output, target)
		self.recall(output, target)
		self.F1Score(output, target)
		self.confusionMatrix(output, target)
		
	def compute(self, args, show=False):
		""" Metric on all batches using custom accumulation """
		acc = self.accuracy.compute().item()
		prec = self.precision.compute().item()
		rec = self.recall.compute().item()
		f1 = self.F1Score.compute().item()
		cm = self.confusionMatrix.compute().cpu().numpy()

		# Print metrics
		if self.show_parameters or show:
			print(f"Accuracy:\t{acc * 100:0.3f}")
			print(f"Precision:\t{prec * 100:0.3f}")
			print(f"Recall:\t\t{rec * 100:0.3f}")
			print(f"F1 Score:\t{f1 * 100:0.3f}")
			plot_confusion_matrix(args, conf_matrix=cm, cls_names=self.cls_names)


def adjust_learning_rate(args, optimizer, epoch):
	""" Sets the learning rate to the initial LR decayed by 10 every 30 epochs """
	cur_lr = args.lr

	if args.warmup_epochs and (epoch <= args.warmup_epochs):
		cur_lr = epoch * args.lr / args.warmup_epochs
		if epoch == 0 and args.warmup_factor is not None:
			cur_lr = max(cur_lr, args.lr * args.warmup_factor)

	elif args.scheduler == 'poly':
		epoch_frac = (args.epochs - epoch) / args.epochs
		epoch_frac = max(epoch_frac, 0)
		cur_lr = args.lr * (epoch_frac ** args.polystep_power)

	elif args.scheduler == 'step':											# step
		num_milestones = 0
		for m in args.milestones:
			num_milestones += (1 if epoch >= m else 0)
		cur_lr = args.lr * (args.multistep_gamma ** num_milestones)

	elif args.scheduler == 'exponential':									# exponential
		cur_lr = args.lr * (args.multistep_gamma ** (epoch//args.step_size))

	elif args.scheduler == 'cosine':										# cosine
		if epoch == 0:
			cur_lr = args.lr
		else:
			lr_min = 0
			cur_lr = (args.lr - lr_min) * (1 + math.cos(math.pi * epoch / args.epochs)) / 2.0 + lr_min
	else:
		ValueError('Unknown scheduler {}'.format(args.scheduler))

	for param_group in optimizer.param_groups:
		param_group['lr'] = cur_lr

	return cur_lr


def accuracy(output, target, topk=(1,)):
	""" Computes the precision@k for the specified values of k """
	with torch.no_grad():
		maxk = max(topk)
		batch_size = target.size(0)

		_, pred = output.topk(maxk, 1, True, True)
		pred = pred.t()
		correct = pred.eq(target.view(1, -1).expand_as(pred))

		res = []
		for k in topk:
			correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
			res.append(correct_k.mul_(100.0 / batch_size))
		return res


def get_dataset_sampler(dataset_object, epoch_size, balanced_sampler=False):
	num_samples = len(dataset_object)
	epoch_size = int(epoch_size * num_samples) if epoch_size < 1 else int(epoch_size)
	print('=> creating a random sampler as epoch_size is specified')

	if balanced_sampler:
		# going through the dataset this way may take too much time
		progress_bar = progiter.ProgIter(np.arange(num_samples), chunksize=1, \
			desc='=> reading data to create a balanced data sampler : ')
		sample_classes = [target for _, target in progress_bar(dataset_object)]
		num_classes = max(sample_classes) + 1
		sample_counts = np.zeros(num_classes, dtype=np.int32)

		for target in sample_classes:
			sample_counts[target] += 1

		train_class_weights = [float(num_samples) / float(cnt) for cnt in sample_counts]
		train_sample_weights = [train_class_weights[target] for target in sample_classes]
		dataset_sampler = torch.utils.data.sampler.WeightedRandomSampler(train_sample_weights, epoch_size)
	else:
		dataset_sampler = torch.utils.data.sampler.RandomSampler(data_source=dataset_object, replacement=True, num_samples=epoch_size)

	return dataset_sampler


class GrayTo3Channel():
	def __init__(self, probability=0.2):
		self.probability = probability
	
	def __call__(self, img):
		if random.random() < self.probability:
			img_gray = img.convert("L")
			np_img = np.array(img_gray)
			np_img_3channel = np.stack([np_img, np_img, np_img], axis=2)
			return Image.fromarray(np_img_3channel, 'RGB')
		else:
			return img


class AddPadding():
	def __init__(self, fill=0, padding_mode='constant'):
		self.fill = fill
		self.padding_mode = padding_mode

	def __call__(self, img):
		width, height = img.size

		if width != height:
			pad = int((height - width) // 2)
			img = torchvision.transforms.Pad(padding=(pad, 0, pad, 0), fill=self.fill, padding_mode=self.padding_mode)(img)
		return img
	

def get_train_transform(args):
	"""
	Original image --> [RandomResizedCrop | RandomCrop] --> AutoAugment
	 --> RandomResizedCrop:
		--> Crop a random area with 'range scale' and 'range ratio';
		--> Resize to 'img_crop'size without maintaining aspect ratio.
	 --> Apply 'AutoAugment'.
	 --> Apply 'RandomErasing'.
	"""
	reverse_channels = xvision.transforms.ReverseChannels() if args.input_channel_reverse else xvision.transforms.Bypass()

	train_resize_crop_transform = torchvision.transforms.RandomResizedCrop(size=args.img_crop, scale=args.rand_scale) \
		if (args.rand_scale[0] != -1) else torchvision.transforms.RandomCrop(size=args.img_crop)
	
	train_transform_list = [reverse_channels,
							# AddPadding(fill=0, padding_mode='constant'),	# custom TODO
							train_resize_crop_transform,
							torchvision.transforms.RandomHorizontalFlip()]
	
	if args.auto_augument is not None:
		# AutoAugmentPolicy is a enum - so we can construct it with value
		auto_augument_policy = torchvision.transforms.autoaugment.AutoAugmentPolicy(args.auto_augument)
		train_transform_list += [torchvision.transforms.autoaugment.AutoAugment(auto_augument_policy)]

	multi_color_transform = xvision.transforms.MultiColor(args.multi_color_modes) if (args.multi_color_modes is not None) else xvision.transforms.Bypass()
	normalize = xvision.transforms.NormalizeMeanScale(mean=args.image_mean, scale=args.image_scale) \
		if (args.image_mean is not None and args.image_scale is not None) else xvision.transforms.Bypass()
	
	train_transform_list += [multi_color_transform,
							 GrayTo3Channel(probability=0.1),	# custom TODO
							 xvision.transforms.ToFloat(),
							 torchvision.transforms.ToTensor(),
							 normalize]
	
	# RandomErasing operates on tensors - not PIL.Image - so do it after ToTensor
	if args.random_erasing is not None and args.random_erasing > 0:
		train_transform_list += [
			torchvision.transforms.RandomErasing(p=args.random_erasing, scale=(0.02, 0.33), ratio=(0.3, 3.3), value='random')
			] # value = ['random', 0, 255,]; scale_s=(0.01, 0.05) scale_l=(0.02, 0.33)
	
	train_transform = torchvision.transforms.Compose(train_transform_list)

	# Show train transformer
	print(f"{'*'*100}\n\n\n{train_transform}\n\n\n{'*'*100}")
	return train_transform


def get_validation_transform(args):
	"""
	Original image --> Resize --> CenterCrop
	 --> Resizes the smaller edge to 'img_resize' while maintaining aspect ratio -->
	 --> Apply CenterCrop with 'img_crop' size.
	"""
	reverse_channels = xvision.transforms.ReverseChannels() if args.input_channel_reverse else xvision.transforms.Bypass()
	multi_color_transform = xvision.transforms.MultiColor(args.multi_color_modes) if (args.multi_color_modes is not None) else xvision.transforms.Bypass()
	normalize = xvision.transforms.NormalizeMeanScale(mean=args.image_mean, scale=args.image_scale) \
		if (args.image_mean is not None and args.image_scale is not None) else xvision.transforms.Bypass()

	# pass tuple to Resize() to resize to exact size without respecting aspect ratio (typical caffe style)
	val_resize_crop_transform = torchvision.transforms.Resize(size=args.img_resize) if args.img_resize else xvision.transforms.Bypass()
	val_transform = torchvision.transforms.Compose([reverse_channels,
													# AddPadding(fill=0, padding_mode='constant'),	# custom TODO
													val_resize_crop_transform,
													torchvision.transforms.CenterCrop(size=args.img_crop),
													multi_color_transform,
													xvision.transforms.ToFloat(),
													torchvision.transforms.ToTensor(),
													normalize])
	return val_transform


def get_transforms(args):
	# Provision to train with val transform - provide rand_scale as (0, 0)
	# Fixing the train-test resolution discrepancy, https://arxiv.org/abs/1906.06423
	always_use_val_transform = (args.rand_scale[0] == 0)
	train_transform = get_validation_transform(args) if always_use_val_transform else get_train_transform(args)
	val_transform = get_validation_transform(args)
	return (train_transform, val_transform)


def get_data_loaders(args):
	train_transform, val_transform = get_transforms(args) if args.transforms is None else (args.transforms[0], args.transforms[1])
	train_dataset, val_dataset = xvision.datasets.__dict__[args.dataset_name](args.dataset_config, args.data_path, transforms=(train_transform,val_transform))

	if args.distributed:
		train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
		val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
	else:
		train_sampler = get_dataset_sampler(train_dataset, args.epoch_size) if args.epoch_size != 0 else None
		val_sampler = get_dataset_sampler(val_dataset, args.epoch_size_val) if args.epoch_size_val != 0 else None

	train_shuffle = args.shuffle and (train_sampler is None)
	train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=train_shuffle,
											   num_workers=args.workers, pin_memory=True, sampler=train_sampler)

	val_shuffle = args.shuffle_val and (val_sampler is None)
	val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=args.batch_size, shuffle=val_shuffle,
											 num_workers=args.workers, pin_memory=True, drop_last=False, sampler=val_sampler)

	return train_loader, val_loader


if __name__ == '__main__':
	main()
