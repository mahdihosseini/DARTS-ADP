import os
import sys
import time
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
from model import NetworkADP as Network

# for ADP dataset only
from ADP_utils.classesADP import classesADP

parser = argparse.ArgumentParser("adp")
parser.add_argument('--data', type=str, default='../data', help='location of the data corpus')
parser.add_argument('--dataset', type=str, default='ADP', help='choose dataset: ADP, BCSS, BACH, OS')
parser.add_argument('--batch_size', type=int, default=96, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.025, help='init learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--epochs', type=int, default=600, help='num of training epochs')
parser.add_argument('--init_channels', type=int, default=36, help='num of init channels')
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--auxiliary', action='store_true', default=False, help='use auxiliary tower')
parser.add_argument('--auxiliary_weight', type=float, default=0.4, help='weight for auxiliary loss')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--drop_path_prob', type=float, default=0.2, help='drop path probability')
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--arch', type=str, default='DARTS_ADP_N4', help='choose network architecture: DARTS_ADP_N2, DARTS_ADP_N3, DARTS_ADP_N4')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--image_size', type=int, default=272, help='ADP image size')
args = parser.parse_args()

args.save = 'eval-{}-{}-size-{}-{}'.format(args.save, args.arch, args.adp_size, time.strftime("%Y%m%d-%H%M%S"))
utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

if args.dataset == 'ADP':
    n_classes = classesADP['L3']['numClasses']
elif args.dataset == 'BCSS':
    n_classes = 10
elif args.dataset == 'BACH' or args.dataset == 'OS':
    n_classes = 4
else:
    logging.info('Unknown dataset')
    sys.exit(1)

def main():
    if not torch.cuda.is_available():
        logging.info('no gpu device available')
        sys.exit(1)

    np.random.seed(args.seed)
    torch.cuda.set_device(args.gpu)
    cudnn.benchmark = True
    torch.manual_seed(args.seed)
    cudnn.enabled = True
    torch.cuda.manual_seed(args.seed)
    logging.info('gpu device = %d' % args.gpu)
    logging.info("args = %s", args)

    genotype = eval("genotypes.%s" % args.arch)
    logging.info('genotype = %s', genotype)

    # dataset
    if args.dataset == 'ADP':
        train_transform, valid_transform = utils._data_transforms_adp(args)
        train_data = utils.ADP_dataset(level=args.adp_level, transform=train_transform, root=args.data, split='train')
        valid_data = utils.ADP_dataset(level=args.adp_level, transform=valid_transform, root=args.data, split='valid')
        test_data = utils.ADP_dataset(level=args.adp_level, transform=valid_transform, root=args.data, split='test')
    elif args.dataset == 'BCSS':
        train_transform, valid_transform = utils._data_transforms_bcss(args)
        train_data = utils.BCSSDataset(root=args.data, split='train', transform=train_transform)
        valid_data = utils.BCSSDataset(root=args.data, split='valid', transform=valid_transform)
        test_data = utils.BCSSDataset(root=args.data, split='test', transform=valid_transform)
    elif args.dataset == 'BACH':
        train_transform, valid_transform = utils._data_transforms_bach(args)
        train_data = utils.BACH_transformed(root=args.data, split='train', transform=train_transform)
        valid_data = utils.BACH_transformed(root=args.data, split='valid', transform=valid_transform)
        test_data = utils.BACH_transformed(root=args.data, split='test', transform=valid_transform)
    elif args.dataset == 'OS':
        train_transform, valid_transform = utils._data_transforms_os(args)
        train_data = utils.OS_transformed(root=args.data, split='train', transform=train_transform)
        valid_data = utils.OS_transformed(root=args.data, split='valid', transform=valid_transform)
        test_data = utils.OS_transformed(root=args.data, split='test', transform=valid_transform)

    train_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=2)

    valid_queue = torch.utils.data.DataLoader(
        valid_data, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=2)

    test_queue = torch.utils.data.DataLoader(
        test_data, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=2)

    dataset_size = len(train_queue.dataset)
    print('train dataset size:', len(train_queue.dataset))
    print('valid dataset size:', len(valid_queue.dataset))
    print('test dataset size:', len(test_queue.dataset))

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
    model = Network(args.init_channels, n_classes, 4, args.auxiliary, genotype)
    model = model.cuda()

    logging.info("param size = %fM", utils.count_parameters_in_MB(model))

    # optimizer and scheduler
    optimizer = torch.optim.SGD(
        model.parameters(),
        args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(args.epochs))

    # train
    best_acc = 0
    for epoch in range(args.epochs):
        scheduler.step()
        logging.info('epoch %d lr %e', epoch, scheduler.get_lr()[0])
        model.drop_path_prob = args.drop_path_prob * epoch / args.epochs

        train_acc_1, train_acc_5, train_obj = train(train_queue, model, criterion, optimizer)
        logging.info('train_acc_1 %f, train_acc_5 %f', train_acc_1, train_acc_5)

        valid_acc_1, valid_acc_5, valid_obj = infer(valid_queue, model, criterion)
        logging.info('valid_acc_1 %f, valid_acc_5 %f', valid_acc_1, valid_acc_5)

        if valid_acc_1 > best_acc:
            best_acc = valid_acc_1
            utils.save(model, os.path.join(args.save, 'best_weights.pt'))

        utils.save(model, os.path.join(args.save, 'last_weights.pt'))
    
    # test
    # use last weights
    logging.info("Test using last weights ...")
    model_test = Network(args.init_channels, n_classes, 4, args.auxiliary, genotype)
    model_test = model_test.cuda()
    utils.load(model_test, os.path.join(args.save, 'last_weights.pt'))
    model_test.drop_path_prob = args.drop_path_prob
    test_acc1, test_acc5, test_obj = infer(test_queue, model_test, criterion)
    logging.info('test_acc_1 %f, test_acc_5 %f', test_acc1, test_acc5)
    # use best weights on valid set
    logging.info("Test using best weights ...")
    model_test = Network(args.init_channels, n_classes, 4, args.auxiliary, genotype)
    model_test = model_test.cuda()
    utils.load(model_test, os.path.join(args.save, 'best_weights.pt'))
    model_test.drop_path_prob = args.drop_path_prob
    test_acc1, test_acc5, test_obj = infer(test_queue, model_test, criterion)
    logging.info('test_acc_1 %f, test_acc_5 %f', test_acc1, test_acc5)

def train(train_queue, model, criterion, optimizer):
    objs = utils.AverageMeter()
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()
    model.train()

    trained_data_size = 0
    for step, (input, target) in enumerate(train_queue):
        input = Variable(input).cuda()
        target = Variable(target).cuda(async=True)

        optimizer.zero_grad()
        logits, logits_aux = model(input)
        loss = criterion(logits, target)
        if args.auxiliary:
            loss_aux = criterion(logits_aux, target)
            loss += args.auxiliary_weight * loss_aux
        loss.backward()
        nn.utils.clip_grad_norm(model.parameters(), args.grad_clip)
        optimizer.step()
        
        n = input.size(0)
        trained_data_size += n

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

        # report training loss
        if step % args.report_freq == 0:
            if args.dataset == 'ADP' or args.dataset == 'BCSS':
                top1_avg = (top1.sum_accuracy.cpu().item() / (trained_data_size * n_classes))
                top5_avg = (top5.sum_accuracy.cpu().item() / trained_data_size) 
            elif args.dataset == 'BACH' or args.dataset == 'OS':
                top1_avg = top1.avg
                top5_avg = top5.avg 
            logging.info('train %03d %e %f %f', step, objs.avg, top1_avg, top5_avg)

    if args.dataset == 'ADP' or args.dataset == 'BCSS':  
        top1_avg = (top1.sum_accuracy.cpu().item() / (len(train_queue.dataset) * n_classes))
        top5_avg = (top5.sum_accuracy.cpu().item() / len(train_queue.dataset)) 
    elif args.dataset == 'BACH' or args.dataset == 'OS':
        top1_avg = top1.avg
        top5_avg = top5.avg

    return top1_avg, top5_avg, objs.avg

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

    if args.dataset == 'ADP' or args.dataset == 'BCSS':  
        top1_avg = (top1.sum_accuracy.cpu().item() / (len(valid_queue.dataset) * n_classes))
        top5_avg = (top5.sum_accuracy.cpu().item() / len(valid_queue.dataset))
    elif args.dataset == 'BACH' or args.dataset == 'OS':
        top1_avg = top1.avg
        top5_avg = top5.avg
        
    return top1_avg, top5_avg, objs.avg


if __name__ == '__main__':
    main()
