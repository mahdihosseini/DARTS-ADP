import os
import sys
import glob
import numpy as np
import torch
import utils
import logging
import argparse
import torch.nn as nn
import genotypes
import torch.utils
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn

from torch.autograd import Variable
from model import DARTS_ADP_N2, DARTS_ADP_N3, DARTS_ADP_N4
# for ADP dataset
from ADP_utils.classesADP import classesADP



parser = argparse.ArgumentParser("adp")
parser.add_argument('--data', type=str, default='./data', help='location of the data corpus')
parser.add_argument('--dataset', type=str, default='ADP', help='choose dataset: ADP, BCSS, BACH, OS')
parser.add_argument('--batch_size', type=int, default=96, help='batch size')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--model_path', type=str, default='./pretrained/ADP/darts_adp_n4.pt', help='path of pretrained model')
parser.add_argument('--auxiliary', action='store_true', default=False, help='use auxiliary tower')
parser.add_argument('--drop_path_prob', type=float, default=0.2, help='drop path probability')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--arch', type=str, default='DARTS_ADP_N4', help='choose network architecture: DARTS_ADP_N2, DARTS_ADP_N3, DARTS_ADP_N4')
parser.add_argument('--image_size', type=int, default=272, help='ADP image size')
# ADP only
parser.add_argument('--adp_level', type=str, default='L3', help='ADP level')
args = parser.parse_args()

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')

if args.dataset == 'ADP':
    n_classes = classesADP[args.adp_level]['numClasses']
elif args.dataset == 'BCSS':
    n_classes = 10
elif args.dataset == 'BACH' or args.dataset == 'OS':
    n_classes = 4
else:
    logging.info('Unknown dataset!')
    sys.exit(1)

def main():
    if not torch.cuda.is_available():
        logging.info('no gpu device available')
        sys.exit(1)

    np.random.seed(args.seed)
    torch.cuda.set_device(args.gpu)
    cudnn.benchmark = True
    torch.manual_seed(args.seed)
    cudnn.enabled=True
    torch.cuda.manual_seed(args.seed)
    logging.info('gpu device = %d' % args.gpu)
    logging.info("args = %s", args)

    # dataset
    if args.dataset == 'ADP':
        train_transform, test_transform = utils._data_transforms_adp(args)
        train_data = utils.ADP_dataset(level=args.adp_level, transform=train_transform, root=args.data, split='train')
        test_data = utils.ADP_dataset(level=args.adp_level, transform=test_transform, root=args.data, split='test')
    elif args.dataset == 'BCSS':
        train_transform, test_transform = utils._data_transforms_bcss(args)
        train_data = utils.BCSSDataset(root=args.data, split='train', transform=train_transform)
        test_data = utils.BCSSDataset(root=args.data, split='test', transform=test_transform)
    elif args.dataset == 'BACH':
        train_transform, test_transform = utils._data_transforms_bach(args)
        train_data = utils.BACH_transformed(root=args.data, split='train', transform=train_transform)
        test_data = utils.BACH_transformed(root=args.data, split='test', transform=test_transform)
    elif args.dataset == 'OS':
        train_transform, test_transform = utils._data_transforms_os(args)
        train_data = utils.OS_transformed(root=args.data, split='train', transform=train_transform)
        test_data = utils.OS_transformed(root=args.data, split='test', transform=test_transform)

    train_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=2)

    test_queue = torch.utils.data.DataLoader(
        test_data, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=2)

    dataset_size = len(train_queue.dataset)

    # criterion
    # ADP and BCSS are multi-label datasets
    # Use MultiLabelSoftMarginLoss
    if args.dataset == 'ADP' or args.dataset == 'BCSS':
        train_class_counts = np.sum(train_queue.dataset.class_labels, axis=0)
        weightsBCE = dataset_size / train_class_counts
        weightsBCE = torch.as_tensor(weightsBCE, dtype=torch.float32).to(int(args.gpu))
        criterion = torch.nn.MultiLabelSoftMarginLoss(weight=weightsBCE).cuda()
    # BACH and OS are single-label datasets
    # Use CrossEntropyLoss
    elif args.dataset == 'BACH' or args.dataset == 'OS':
        criterion = nn.CrossEntropyLoss()
        criterion = criterion.cuda()

    # model
    if args.arch == 'DARTS_ADP_N2':
        model = DARTS_ADP_N2(n_classes, args.auxiliary)
    elif args.arch == 'DARTS_ADP_N3':
        model = DARTS_ADP_N3(n_classes, args.auxiliary)
    elif args.arch == 'DARTS_ADP_N4':
        model = DARTS_ADP_N4(n_classes, args.auxiliary)
    else:
        logging.info('Unknown architecture!')
        sys.exit(1)
    utils.load(model, args.model_path)
    model.drop_path_prob = args.drop_path_prob
    logging.info("param size = %fM", utils.count_parameters_in_MB(model))

    test_acc1, test_acc5, test_obj = infer(test_queue, model, criterion)
    logging.info('test_acc_1 %f, test_acc_5 %f', test_acc1, test_acc5)


def infer(valid_queue, model, criterion):
    objs = utils.AverageMeter()
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()
    model.eval()

    infered_data_size = 0
    with torch.no_grad():
        for step, (input, target) in enumerate(valid_queue):
            input = Variable(input, volatile=True).cuda()
            target = Variable(target, volatile=True).cuda(async=True)

            logits, _ = model(input)
            loss = criterion(logits, target)

            n = input.size(0)
            infered_data_size += n

            if args.dataset == 'ADP' or args.dataset == 'BCSS':
                m = nn.Sigmoid()
                preds = (m(logits) > 0.5).int()
                prec1, prec5 = utils.accuracyADP(preds, target)
                objs.update(loss.item(), n)
                top1.update(prec1.double(), n)
                top5.update(prec5.double(), n)
            elif args.dataset == 'BACH' or args.dataset == 'OS':
                prec1, prec5 = utils.accuracy(logits, target, topk=(1, min(5, n_classes)))
                objs.update(loss.item(), n)
                top1.update(prec1.item(), n)
                top5.update(prec5.item(), n)
 
            # report validation loss
            if step % args.report_freq == 0:
                if args.dataset == 'ADP' or args.dataset == 'BCSS':
                    top1_avg = (top1.sum_accuracy.cpu().item() / (infered_data_size * n_classes))
                    top5_avg = (top5.sum_accuracy.cpu().item() / infered_data_size) 
                elif args.dataset == 'BACH' or args.dataset == 'OS':
                    top1_avg = top1.avg
                    top5_avg = top5.avg 
                logging.info('valid %03d %e %f %f', step, objs.avg, top1_avg, top5_avg)
    print('infered_data_size:', infered_data_size)
    print('valid_data_size:', len(valid_queue.dataset))

    if args.dataset == 'ADP' or args.dataset == 'BCSS':  
        top1_avg = (top1.sum_accuracy.cpu().item() / (len(valid_queue.dataset) * n_classes))
        top5_avg = (top5.sum_accuracy.cpu().item() / len(valid_queue.dataset))
    elif args.dataset == 'BACH' or args.dataset == 'OS':
        top1_avg = top1.avg
        top5_avg = top5.avg
        
    return top1_avg, top5_avg, objs.avg

if __name__ == '__main__':
  main()

