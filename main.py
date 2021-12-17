#! /usr/bin/env python
# -*- coding: utf-8 -*-

import os
import time
import random
import argparse
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F


from lipreading.utils import get_save_folder
from lipreading.utils import load_model, CheckpointSaver
from lipreading.utils import get_logger, update_logger_batch
from lipreading.utils import showLR, AverageMeter
from lipreading.model import myLipreading
from lipreading.mixup import mixup_data, mixup_criterion
from lipreading.optim_utils import get_optimizer, CosineScheduler
from lipreading.dataloaders import get_data_loaders, get_preprocessing_pipelines


def load_args(default_config=None):
    parser = argparse.ArgumentParser(description='Pytorch Lipreading ')
    # -- directory
    parser.add_argument('--data-dir', default='./datasets/visual_data_50', help='Loaded data directory')
    parser.add_argument('--label-path', type=str, default='./labels/words_50.txt', help='Path to txt file with labels')
    parser.add_argument('--annonation-direc', default=None, help='Loaded data directory')

    # -- train
    parser.add_argument('--training-mode', default='tcn', help='tcn')
    parser.add_argument('--batch-size', type=int, default=16, help='Mini-batch size')
    parser.add_argument('--optimizer',type=str, default='sgd', choices = ['adam','sgd','adamw'])    #adamw, 3e-4
    parser.add_argument('--lr', default=1e-4, type=float, help='initial learning rate')
    parser.add_argument('--init-epoch', default=0, type=int, help='epoch to start at')
    parser.add_argument('--epochs', default=80, type=int, help='number of epochs')
    parser.add_argument('--test', default=False, action='store_true', help='training mode')
    parser.add_argument('--alpha', default=0.4, type=float, help='interpolation strength (uniform=1., ERM=0.)')
    parser.add_argument('--extract-feats', default=False, action='store_true', help='Feature extractor')

    # -- test
    parser.add_argument('--model-path', type=str, default=None, help='Pretrained model pathname') #./train_logs/tcn/senet_mstcn_50cls_v1_2021_12_10T15_56_14/ckpt.best.pth.tar
    parser.add_argument('--allow-size-mismatch', default=False, action='store_true',
                        help='If True, allows to init from model with mismatching weight tensors. Useful to init from model with diff. number of classes')
    # -- other vars
    parser.add_argument('--interval', default=50, type=int, help='display interval')
    parser.add_argument('--workers', default=8, type=int, help='number of data loading workers')
    # paths
    parser.add_argument('--logging-dir', type=str, default='./train_logs', help = 'path to the directory in which to save the log file')

    args = parser.parse_args()
    return args


args = load_args()

torch.manual_seed(1)
np.random.seed(1)
random.seed(1)
torch.backends.cudnn.benchmark = True


def extract_feats(model):
    """
    :rtype: FloatTensor
    """
    model.eval()
    preprocessing_func = get_preprocessing_pipelines()['test']
    data = preprocessing_func(np.load(args.mouth_patch_path)['data'])  # data: TxHxW
    return model(torch.FloatTensor(data)[None, None, :, :, :].cuda(), lengths=[data.shape[0]])


def evaluate(model, dset_loader, criterion):

    model.eval()

    running_loss = 0.
    running_corrects = 0.

    with torch.no_grad():
        for batch_idx, (input, lengths, labels) in enumerate(tqdm(dset_loader)):
            logits = model(input.unsqueeze(1).cuda(), lengths=lengths)
            _, preds = torch.max(F.softmax(logits, dim=1).data, dim=1)
            running_corrects += preds.eq(labels.cuda().view_as(preds)).sum().item()

            loss = criterion(logits, labels.cuda())
            running_loss += loss.item() * input.size(0)

    print('{} in total\tCR: {}'.format( len(dset_loader.dataset), running_corrects/len(dset_loader.dataset)))
    return running_corrects/len(dset_loader.dataset), running_loss/len(dset_loader.dataset)


def train(model, dset_loader, criterion, epoch, optimizer, logger):
    data_time = AverageMeter()
    batch_time = AverageMeter()

    lr = showLR(optimizer)

    logger.info('-' * 10)
    logger.info('Epoch {}/{}'.format(epoch, args.epochs - 1))
    logger.info('Current learning rate: {}'.format(lr))

    model.train()
    running_loss = 0.
    running_corrects = 0.
    running_all = 0.

    end = time.time()
    for batch_idx, (input, lengths, labels) in enumerate(dset_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # --
        input, labels_a, labels_b, lam = mixup_data(input, labels, args.alpha)
        labels_a, labels_b = labels_a.cuda(), labels_b.cuda()

        optimizer.zero_grad()

        logits = model(input.unsqueeze(1).cuda(), lengths=lengths)

        loss_func = mixup_criterion(labels_a, labels_b, lam)
        loss = loss_func(criterion, logits)

        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        # -- compute running performance
        _, predicted = torch.max(F.softmax(logits, dim=1).data, dim=1)
        running_loss += loss.item()*input.size(0)
        running_corrects += lam * predicted.eq(labels_a.view_as(predicted)).sum().item() + (1 - lam) * predicted.eq(labels_b.view_as(predicted)).sum().item()
        running_all += input.size(0)
        # -- log intermediate results
        if batch_idx % args.interval == 0 or (batch_idx == len(dset_loader)-1):
            update_logger_batch( args, logger, dset_loader, batch_idx, running_loss, running_corrects, running_all, batch_time, data_time )

    return model


def get_model(num_classes):
    print("load model from myLipreading...")
    backbone_type = "resnet"
    backend_type = "mstcn"
    relu_type = "relu"
    tcn_options = {'num_layers': 4,
                   'kernel_size': [3, 5, 7],
                   'dropout': 0.2,
                   'dwpw': False,
                   'width_mult': 1,
                   }
    model = myLipreading(num_classes=num_classes,
                         tcn_options=tcn_options,
                         backbone_type=backbone_type,
                         backend_type=backend_type,
                         relu_type=relu_type,

                         ms3d=False,
                         extract_feats=args.extract_feats).cuda()
    return model


def main():
    # -- logging
    num_classes = 50
    save_path = get_save_folder(args)
    print("Model and log being saved in: {}".format(save_path))
    logger = get_logger(args, save_path, num_classes)
    ckpt_saver = CheckpointSaver(save_path)

    # -- get model
    model = get_model(num_classes)
    # -- get dataset iterators
    dset_loaders = get_data_loaders(args)
    # -- get loss function
    criterion = nn.CrossEntropyLoss()
    # -- get optimizer
    optimizer = get_optimizer(args, optim_policies=model.parameters())
    # -- get learning rate scheduler
    if args.optimizer == 'sgd':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    else:
        scheduler = CosineScheduler(args.lr, args.epochs)

    if args.model_path:
        assert args.model_path.endswith('.tar') and os.path.isfile(args.model_path), \
            "'.tar' model path does not exist. Path input: {}".format(args.model_path)
        # resume from checkpoint
        if args.init_epoch > 0:
            model, optimizer, epoch_idx, ckpt_dict = load_model(args.model_path, model, optimizer)
            args.init_epoch = epoch_idx
            ckpt_saver.set_best_from_ckpt(ckpt_dict)
            logger.info('Model and states have been successfully loaded from {}'.format( args.model_path ))
        # init from trained model
        else:
            model = load_model(args.model_path, model, allow_size_mismatch=args.allow_size_mismatch)
            logger.info('Model has been successfully loaded from {}'.format( args.model_path ))

        # if test-time, performance on test partition and exit. Otherwise, performance on validation and continue (sanity check for reload)
        if args.test:
            acc_avg_test, loss_avg_test = evaluate(model, dset_loaders['test'], criterion)
            logger.info('Test-time performance on partition {}: Loss: {:.4f}\tAcc:{:.4f}'.format( 'test', loss_avg_test, acc_avg_test))
            return

    # print summary
    print(model)

    # -- fix learning rate after loading the ckeckpoint (latency)
    if args.optimizer != 'sgd' and args.model_path and args.init_epoch > 0:
        scheduler.adjust_lr(optimizer, args.init_epoch-1)

    epoch = args.init_epoch

    while not args.test and epoch < args.epochs:
        model = train(model, dset_loaders['train'], criterion, epoch, optimizer, logger)
        acc_avg_val, loss_avg_val = evaluate(model, dset_loaders['val'], criterion)
        logger.info('{} Epoch:\t{:2}\tLoss val: {:.4f}\tAcc val:{:.4f}, LR: {}'.format('val', epoch, loss_avg_val, acc_avg_val, showLR(optimizer)))
        # -- save checkpoint
        save_dict = {
            'epoch_idx': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }
        ckpt_saver.save(save_dict, acc_avg_val)
        if args.optimizer == 'sgd':
            scheduler.step()
        else:
            scheduler.adjust_lr(optimizer, epoch)
        epoch += 1

    # -- evaluate best-performing epoch on test partition
    best_fp = os.path.join(ckpt_saver.save_dir, ckpt_saver.best_fn)
    _ = load_model(best_fp, model)
    acc_avg_test, loss_avg_test = evaluate(model, dset_loaders['test'], criterion)
    logger.info('Test time performance of best epoch: {} (loss: {})'.format(acc_avg_test, loss_avg_test))

if __name__ == '__main__':
    main()
