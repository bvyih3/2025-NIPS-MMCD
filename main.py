import os
import sys
import json
import argparse
from pprint import pprint
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import utils.utils as utils
import utils.config as config
from train import train, evaluate
import modules.base_model_arcface as base_model
from utils.dataset import Dictionary, VQAFeatureDataset
from utils.losses import Plain


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=20,
                        help='number of running epochs')
    parser.add_argument('--lr', type=float, default=0.002,
                        help='learning rate for adamax')
    parser.add_argument('--loss-fn', type=str, default='Plain',
                        help='chosen loss function')
    parser.add_argument('--num-hid', type=int, default=1024,
                        help='number of dimension in last layer')
    parser.add_argument('--base_model', type=str, default='updn',
                        choices=["updn", "SAN", "SMRL", "SAN_MEVF", "BAN_MEVF"], help='base model')
    parser.add_argument('--baseline', action='store_true',
                        help='use the baseline model')
    parser.add_argument('--model', type=str, default='updn_att',
                        help='model structure')
    parser.add_argument('--name', type=str, default='exp0.pth',
                        help='saved model name')
    parser.add_argument('--name-new', type=str, default=None,
                        help='combine with fine-tune')
    parser.add_argument('--batch-size', type=int, default=512,
                        help='training batch size')
    parser.add_argument('--fine-tune', action='store_true',
                        help='fine tuning with our loss')
    parser.add_argument('--resume', action='store_true',
                        help='whether resume from checkpoint')
    parser.add_argument('--not-save', action='store_true',
                        help='do not overwrite the old model')
    parser.add_argument('--test', dest='test_only', action='store_true',
                        help='test one time')
    parser.add_argument('--eval-only', action='store_true',
                        help='evaluate on the val set one time')
    parser.add_argument("--gpu", type=str, default='0',
                        help='gpu card ID')
    parser.add_argument("--tau1", type=float, default=1.0,
                        help='temperature for contrastive loss')
    parser.add_argument("--tau2", type=float, default=0.2,
                        help='temperature for confidence margin')
    parser.add_argument("--w", type=float, default=10,
                        help='warm-up epoch for instance difficulty')
    parser.add_argument('--c', type=float, default=0.1,
                        help='sensitivity of difficulty model')
    parser.add_argument('--alpha', type=float, default=0.6,
                        help='ensemble parameter for difficulties')
    parser.add_argument('--beta', type=float, default=0.5,
                        help='difficulty update parameter')
    parser.add_argument('--lambda1', type=float, default=0.1,
                        help='hyper-parameter for the confidence margin')
    parser.add_argument('--lambda2', type=float, default=0.05,
                        help='hyper-parameter for the difficulty margin')
    parser.add_argument('--lambda3', type=float, default=2,
                        help='hyper-parameter for DCL')
    parser.add_argument('--lambda4', type=float, default=1,
                        help='hyper-parameter for the ce loss')
    parser.add_argument('--add', action='store_true', default=False,
                        help='add the difficulty margin')
    parser.add_argument(
        "--dataset",
        default="vqacp-v2",
        choices=["vqacp-v1", "vqacp-v2", "vqa-v2", "vqace", "gqaood"],
        help="choose dataset",
    )
    args = parser.parse_args()
    return args



if __name__ == '__main__':
    args = parse_args()
    dataset = args.dataset
    config.dataset = dataset
    config.update_paths(args.dataset)
    if not args.test_only:
        args.name = dataset
    print(args)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    cudnn.benchmark = True

    seed = 1111
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    if 'log' not in args.name:
        args.name = 'logs/' + args.name
    if args.test_only or args.fine_tune or args.eval_only:
        args.resume = True
    if args.resume and not args.name:
        raise ValueError("Resuming requires folder name!")
    if args.resume:
        logs = torch.load(args.name)
        print("loading logs from {}".format(args.name))

    # ------------------------DATASET CREATION--------------------
    dictionary = Dictionary.load_from_file(config.dict_path)
    if args.test_only:
        eval_dset = VQAFeatureDataset('test', dictionary, args)
        args.datasetSize = len(eval_dset.entries)
    else:
        train_dset = VQAFeatureDataset('train', dictionary, args)
        args.datasetSize = len(train_dset.entries)
        if dataset != "vqace":
            eval_dset = VQAFeatureDataset('test', dictionary, args)
        else:
            eval_dset = VQAFeatureDataset('all', dictionary, args)
            easy_dset = VQAFeatureDataset('easy', dictionary, args)
            hard_dset = VQAFeatureDataset('hard', dictionary, args)
            cou_dset = VQAFeatureDataset('cou', dictionary, args)
    # if config.train_set == 'train+val' and not args.test_only:
    #     train_dset = train_dset + eval_dset
    #     eval_dset = VQAFeatureDataset('test', dictionary)
    # if args.eval_only:
    #     eval_dset = VQAFeatureDataset('val', dictionary)

    tb_count = 0
    # writer = SummaryWriter() # for visualization
    writer = None

    if not config.train_set == 'train+val' and 'LM' in args.loss_fn:
        utils.append_bias(train_dset, eval_dset, len(eval_dset.label2ans))

    # ------------------------MODEL CREATION------------------------
    print('base_model: {}'.format(args.base_model))
    if not args.baseline:
        args.model = '{}_att'.format(args.base_model)
        constructor = 'build_{}'.format(args.model)
        print('model:{}'.format(args.model))
        model, metric_fc = getattr(base_model, constructor)(eval_dset, args)
        model = model.cuda()
        metric_fc = metric_fc.cuda()
    else:
        constructor = 'build_{}'.format(args.base_model)
        model = getattr(base_model, constructor)(eval_dset, args)
        model = model.cuda()
        metric_fc = None
    if not args.base_model == 'SAN' and 'MEVF' not in args.base_model:
        model.w_emb.init_embedding(config.glove_embed_path)

    # model = nn.DataParallel(model).cuda()
    if not args.baseline:
        optim = torch.optim.Adamax([{'params': model.parameters()}, {'params': metric_fc.parameters()}], lr=args.lr)
    else:
        optim = torch.optim.Adamax(model.parameters(), lr=args.lr)

    if args.loss_fn == 'Plain':
        loss_fn = Plain()
    else:
        raise RuntimeError('not implement for {}'.format(args.loss_fn))

    # ------------------------STATE CREATION------------------------
    eval_score, best_val_score, start_epoch, best_epoch = 0.0, 0.0, 0, 0
    tracker = utils.Tracker()
    if args.resume:
        model.load_state_dict(logs['model_state'])
        metric_fc.load_state_dict(logs['margin_model_state'])
        optim.load_state_dict(logs['optim_state'])
        if 'loss_state' in logs:
            loss_fn.load_state_dict(logs['loss_state'])
        start_epoch = logs['epoch']
        best_epoch = logs['epoch']
        best_val_score = logs['best_val_score']
        if args.fine_tune:
            print('best accuracy is {:.2f} in baseline'.format(100 * best_val_score))
            args.epochs = start_epoch + 10 # 10 more epochs
            for params in optim.param_groups:
                params['lr'] = config.ft_lr

            # if you want save your model with a new name
            if args.name_new:
                if 'log' not in args.name_new:
                    args.name = 'logs/' + args.name_new
                else:
                    args.name = args.name_new

    eval_loader = DataLoader(eval_dset,
                    args.batch_size, shuffle=False, num_workers=4)
    if dataset == "vqace":
        easy_loader = DataLoader(easy_dset,
                    args.batch_size, shuffle=False, num_workers=4)
        hard_loader = DataLoader(hard_dset,
                    args.batch_size, shuffle=False, num_workers=4)
        cou_loader = DataLoader(cou_dset,
                    args.batch_size, shuffle=False, num_workers=4)
    if args.test_only or args.eval_only:
        model.eval()
        metric_fc.eval()
        evaluate(model, metric_fc, eval_loader, args, write=False)
        if dataset == "vqace":
            evaluate(model, metric_fc, easy_loader, args, write=False)
            evaluate(model, metric_fc, hard_loader, args, write=False)
            evaluate(model, metric_fc, cou_loader, args, write=False)
    else:
        train_loader = DataLoader(
            train_dset, args.batch_size, shuffle=True, num_workers=4)
        for epoch in range(start_epoch, args.epochs):
            print("training epoch {:03d}".format(epoch))
            tb_count = train(model, metric_fc, optim, train_loader, loss_fn, tracker, writer, tb_count, epoch, args)
            if not (config.train_set == 'train+val' and epoch in range(args.epochs - 3)):
                # save for the last three epochs
                write = True if config.train_set == 'train+val' else False
                print("validating after epoch {:03d}".format(epoch))
                model.train(False)
                if not args.baseline:
                    metric_fc.train(False)
                eval_score = evaluate(model, metric_fc, eval_loader, args, epoch, write=write)
                if dataset == "vqace":
                    easy_score = evaluate(model, metric_fc, easy_loader, args, epoch, write=False)
                    hard_score = evaluate(model, metric_fc, hard_loader, args, epoch, write=False)
                    cou_score = evaluate(model, metric_fc, cou_loader, args, epoch, write=False)
                model.train(True)
                if not args.baseline:
                    metric_fc.train(True)
                print("eval score: {:.2f} \n".format(100 * eval_score))

            if eval_score > best_val_score:
                best_val_score = eval_score
                best_epoch = epoch
                if not args.baseline:
                    results = {
                        'epoch': epoch + 1,
                        'best_val_score': best_val_score,
                        'model_state': model.state_dict(),
                        'optim_state': optim.state_dict(),
                        'loss_state': loss_fn.state_dict(),
                        'margin_model_state': metric_fc.state_dict()
                    }
                else:
                    results = {
                        'epoch': epoch + 1,
                        'best_val_score': best_val_score,
                        'model_state': model.state_dict(),
                        'optim_state': optim.state_dict(),
                        'loss_state': loss_fn.state_dict(),
                    }
                if not args.not_save:
                    torch.save(results, args.name)
        print("best accuracy {:.2f} on epoch {:03d}".format(
            100 * best_val_score, best_epoch))
