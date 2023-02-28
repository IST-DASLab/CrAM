"""
Managing class for training a model with custom optimizers.
    * the type of optimizer, hyperparameters and learning rate scheduler can be passed directly from a yaml config file
    * supports multiple choices of models and datasets
"""
from models import get_model
from load_training_configs import build_training_schedule_from_config

from utils import read_config, get_datasets
from utils import enable_running_stats, disable_running_stats
from utils import (preprocess_for_device,
                   load_checkpoint,
                   save_checkpoint,
                   TrainingProgressTracker)
from utils import one_shot_prune, apply_masks, test_sparsity

import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.onnx
import time
import os
import logging
import sys


class Manager:
    """
    Class for training models
    """

    def __init__(self, args):
        args = preprocess_for_device(args)

        torch.manual_seed(args.manual_seed)
        torch.cuda.manual_seed(args.manual_seed)
        np.random.seed(args.manual_seed)

        self.model_config = {'arch': args.arch, 'dataset': args.dset}
        self.model = get_model(args.arch, dataset=args.dset, pretrained=args.pretrained)

        self.config = args.config_path if isinstance(args.config_path, dict) else read_config(args.config_path)

        self.logging_function = print
        if args.use_wandb:
            import wandb
            self.logging_function = wandb.log

        self.data = (args.dset, args.dset_path)
        self.n_epochs = args.epochs
        self.num_workers = args.workers
        self.batch_size = args.batch_size
        self.steps_per_epoch = args.steps_per_epoch
        self.device = args.device
        self.initial_epoch = 0
        self.best_val_acc = 0.
        self.use_train_val_split = args.train_val_split
        
        self.training_stats_freq = args.training_stats_freq
 
        # Define datasets
        if self.use_train_val_split:
            print('Using custom validation set for evaluation!')
        self.data_train, self.data_val = get_datasets(*self.data, use_val_split=self.use_train_val_split)
        self.train_loader = DataLoader(self.data_train, batch_size=self.batch_size, shuffle=True,
                                       num_workers=self.num_workers)
        self.test_loader = DataLoader(self.data_val, batch_size=self.batch_size, shuffle=False,
                                      num_workers=self.num_workers)

        self.only_model = args.only_model
        optimizer_dict, lr_scheduler_dict = None, None
        if args.from_checkpoint_path is not None:
            epoch, model, opt_dict, lr_sch_dict = load_checkpoint(args.from_checkpoint_path)
            self.model = model
            if not self.only_model:
                optimizer_dict, lr_scheduler_dict = opt_dict, lr_sch_dict

        if args.device.type == 'cuda':
            if len(args.gpus) > 1:
                self.model = torch.nn.DataParallel(self.model, device_ids=args.gpus)
            self.model.to(args.device)
        
        self.optimizer, self.lr_scheduler, self.use_sam = build_training_schedule_from_config(self.model, self.config,
                                                                                              sam_rho=args.sam_rho)
        print('==> Optimizer: ', self.optimizer)
        print('==> LR Scheduler: ', self.lr_scheduler)

        if optimizer_dict is not None and lr_scheduler_dict is not None:
            self.initial_epoch = epoch
            self.optimizer.load_state_dict(optimizer_dict)
            self.lr_scheduler.load_state_dict(lr_scheduler_dict)

        self.loss_fn = F.cross_entropy
        print('==> Loss function: ', self.loss_fn)

        self.logging_level = args.logging_level
        self.checkpoint_freq = args.checkpoint_freq
        self.exp_dir = args.exp_dir
        self.run_dir = args.run_dir

        # this is used when finetuning the sparse model after one-shot pruning
        self.train_sparse = args.train_sparse
        self.sparsity = args.sparsity
        if self.train_sparse and (self.sparsity == 0.):
            raise ValueError('Sparsity needs to be > 0 if training sparse')
        
        # if sparse training, do one shot pruning and save the masks
        self.masks_dict = None
        if self.train_sparse:
            self.masks_dict = one_shot_prune(self.model, self.sparsity)
            torch.save(self.masks_dict, os.path.join(self.run_dir, 'weights_masks.pth'))

        self.eval_only = args.eval_only
        if self.eval_only and (args.from_checkpoint_path is None):
            raise ValueError("Must provide a checkpoint for validation")
        
        self.export_onnx = args.export_onnx
        if self.export_onnx and (args.from_checkpoint_path is None):
            self.onnx_nick = args.onnx_nick
            raise ValueError("Must provide a checkpoint for ONNX export")

        self.training_progress = TrainingProgressTracker(self.initial_epoch,
                                                         len(self.train_loader),
                                                         len(self.test_loader.dataset),
                                                         self.training_stats_freq)

    def eval_model(self, loader):
        self.model.eval()
        eval_loss = 0
        correct = 0
        with torch.no_grad():
            for in_tensor_eval, target_eval in loader:
                in_tensor_eval, target_eval = in_tensor_eval.to(self.device), target_eval.to(self.device)
                output = self.model(in_tensor_eval)
                eval_loss += F.cross_entropy(output, target_eval, reduction='sum').item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target_eval.view_as(pred)).sum().item()
        eval_loss /= len(loader.dataset)
        eval_acc = correct / len(loader.dataset)
        return eval_loss, eval_acc, correct

    def log_eval_stats(self, epoch, eval_loss, eval_acc, eval_correct, dset_type='val'):
        # log validation stats
        self.logging_function({'epoch': epoch, dset_type + ' loss': eval_loss, dset_type + ' acc': eval_acc})
        logging.info({'epoch': epoch, dset_type + ' loss': eval_loss, dset_type + ' acc': eval_acc})
        if dset_type == "val":
            self.training_progress.val_info(epoch, eval_loss, eval_correct)

    def get_model_sparsity(self):
        total_zeros = 0.
        total_params = 0.
        for n, p in self.model.named_parameters():
            zero_p = (p.data==0.).float().sum()
            n_p = p.data.numel()
            print(f'{n}: {zero_p/n_p}')
            total_zeros += zero_p
            total_params += n_p
        print(f'total sparsity: {total_zeros/total_params}')
        print("============================================")

    def optimize_minibatch(self, minibatch):
        self.optimizer.zero_grad()
        in_tensor, target = minibatch
        in_tensor, target = in_tensor.to(self.device), target.to(self.device)
        if self.use_sam:
            # first forward-backward step
            predictions = self.model(in_tensor)
            loss = self.loss_fn(predictions, target)
            loss.backward()

            def closure():
                closure_preds = self.model(in_tensor)
                closure_loss = self.loss_fn(closure_preds, target)
                closure_loss.backward()
                return closure_loss

            # second forward-backward step
            disable_running_stats(self.model) # disable BN stats for perturbed model (only computed on the clean model)
            self.optimizer.step(closure)
            enable_running_stats(self.model)
        else:
            predictions = self.model(in_tensor)
            loss = self.loss_fn(predictions, target)
            loss.backward()
            self.optimizer.step()

        with torch.no_grad():
            pred = predictions.argmax(dim=1, keepdim=True)
            correct = pred.eq(target.view_as(pred)).sum().item()
            acc = 1.0 * correct / target.size(0)
        return loss.item(), acc

    @property
    def optim_lr(self):
        return list(self.optimizer.param_groups)[0]['lr']

    def run(self):
        if self.eval_only:
            eval_loss, eval_acc, eval_correct = self.eval_model(self.test_loader)
            print(f'Validation loss: {eval_loss} \t Top-1 validation accuracy: {eval_acc}')
            return 
        
        if self.export_onnx is True:
            self.model.eval()
            onnx_batch = 1
            x = torch.randn(onnx_batch, 3, 224, 224, requires_grad=True).to(self.device)
            if self.onnx_nick:
                onnx_nick = self.onnx_nick
            else:
                onnx_nick = 'my_model.onnx'

            torch.onnx.export(self.model.module,         # model being run
                              x,                         # model input (or a tuple for multiple inputs)
                              onnx_nick,                 # where to save the model (can be a file or file-like object)
                              export_params=True,        # store the trained parameter weights inside the model file
                              opset_version=10,          # the ONNX version to export the model to
                              do_constant_folding=True,  # whether to execute constant folding for optimization
                              input_names=['input'],   # the model's input names
                              output_names=['output'],
                              dynamic_axes={'input': {0: 'batch_size'},    # variable length axes
                                            'output': {0: 'batch_size'}})

            print("ONNX EXPORT COMPLETED. EXITING")
            sys.exit()

        val_correct = 0.

        for epoch in range(self.initial_epoch, self.n_epochs):

            logging.info(f"Starting epoch {epoch} with number of batches {self.steps_per_epoch or len(self.train_loader)}")

            epoch_trn_loss, epoch_trn_acc = 0., 0.
            n_samples = len(self.train_loader.dataset)

            self.model.train()

            for i, batch in enumerate(self.train_loader):
                start = time.time()
                # optimize over a minibatch
                # if SAM/CrAM is used, the loss and accuracy are calculated on the un-perturbed model
                loss_batch, acc_batch = self.optimize_minibatch(batch)    
                epoch_trn_acc += acc_batch * batch[0].size(0) / n_samples
                epoch_trn_loss += loss_batch * batch[0].size(0) / n_samples
                
                if self.train_sparse:
                    apply_masks(self.model, self.masks_dict)

                # tracking the training statistics
                self.training_progress.step(loss=loss_batch,
                                            acc=acc_batch,
                                            time=time.time() - start,
                                            lr=self.optim_lr)
            
            current_lr = self.optim_lr
            self.logging_function({'epoch': epoch, 'lr': current_lr})
            
            # update the LR
            self.lr_scheduler.step()

            # log train stats
            self.logging_function({'epoch': epoch, 'train loss': epoch_trn_loss, 'train acc': epoch_trn_acc})
            
            if self.train_sparse:
                real_sparsity = test_sparsity(self.model)
                self.logging_function({'epoch': epoch, 'sparsity': real_sparsity})

            # validate the model
            val_loss, val_acc, val_correct = self.eval_model(self.test_loader)

            # save checkpoint
            scheduled = False
            is_best = False
            if (val_acc > self.best_val_acc) and self.use_train_val_split:
                is_best = True
                self.best_val_acc = val_acc
                logging.info(f"Best validation accuracy: {self.best_val_acc}")
            if (epoch + 1) % self.checkpoint_freq == 0:
                logging.info("scheduled checkpoint")
                scheduled = True
            save_checkpoint(epoch, self.model_config, self.model, self.optimizer,
                            self.lr_scheduler, self.run_dir, is_best=is_best,
                            is_scheduled_checkpoint=scheduled)

            # log validation stats
            self.log_eval_stats(epoch, val_loss, val_acc, val_correct, dset_type='val')

        return val_correct, len(self.test_loader.dataset)
