


import os
import time
from tqdm import tqdm
import math 
import shutil
import argparse
import logging
from contextlib import suppress

import torch
from torch import nn
import torchvision
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import MultiStepLR

from thop import profile, clever_format
#from ptflops import get_model_complexity_info

from utils import Cutout, AverageMeter, ProgressMeter, save_checkpoint, get_accuracy 
from utils import adjust_learning_rate, CrossEntropyLabelSmooth
from model import pdenet, resnet32, resnet_original, pdenet_original
from wideresnet import wide_resnet
from densenet import DenseNet
from darts import DARTS_NetworkCIFAR



torch.backends.cudnn.benchmark = True
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--K', default=3, type=int,
                    metavar='K', help='Number of iterations in the Global feature extractor block (default: 3)')
parser.add_argument('--non_linear', default=False, action='store_true')
parser.add_argument('--pde_state', default=0, type=int,
                    metavar='N', help='PDE State so that we can try out multiple(default: 0)')
parser.add_argument('-dxy', '--constant_Dxy', default=False, action='store_true')
parser.add_argument('--use_silu', default=False, action='store_true')
parser.add_argument('--use_res', default=False, action='store_true')
parser.add_argument('--init_h0_h', default=False, action='store_true')
parser.add_argument('-nof', '--no_f', default=False, action='store_true')
parser.add_argument('--dt', type=float, default=0.2, help='Random erase prob (default: 0.)')
parser.add_argument('--dx', type=int, default=1, help='Random erase prob (default: 0.)')
parser.add_argument('--dy', type=int, default=1, help='Random erase prob (default: 0.)')
parser.add_argument('--cDx', type=float, default=1., help='Random erase prob (default: 0.)')
parser.add_argument('--cDy', type=float, default=1., help='Random erase prob (default: 0.)')

parser.add_argument('-ct', '--cutout', default=False, action='store_true')
parser.add_argument('-o', '--original', default=False, action='store_true')
parser.add_argument('-r', '--restart', default=False, action='store_true')
parser.add_argument('-rk', '--restart_known', default=False, action='store_true')
parser.add_argument('-rmn', '--restart_model_name', type=str, help='model to restart from')
parser.add_argument('--warmup', action='store_true', help='set lower initial learning rate to warm up the training.')
parser.add_argument('--separable', action='store_true', help='using separable convolutions.')
parser.add_argument('-dc', '--lr_decay', default='cos', type=str)
parser.add_argument('-p', '--print-freq', default=500, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('-m', '--resnet-m', default=2, type=int,
                    metavar='N', help='Number of repeats in one resnet block (default: 2)')
parser.add_argument('-wdt', '--width', default=2, type=int,
                    metavar='N', help='Width wide resnet block (default: 2)')
parser.add_argument('-ds', '--dataset', default='CIFAR-10', type=str, 
                    help='dataset ( CIFAR-10/CIFAR-100/Imagenet-1000 )')
parser.add_argument('--drop', type=float, default=0.2, metavar='PCT',
                    help='Dropout rate (default: 0.)')
parser.add_argument('--model', default='Resnet-Global', type=str, 
                    help='architecture ( Resnet-Global/CIFAR )')
parser.add_argument('-d', '--data', default='/home/anilkag/code/compact-vision-nets-PDE-Feature-Generator/data/',
                   type=str, metavar='DIR', help='path to dataset')
parser.add_argument('-n', '--n_holes', default=1, type=int, metavar='N',
                    help='number of holes in cutout augmentation')
parser.add_argument('-l', '--length', default=16, type=int, metavar='N',
                    help='length of each hole in cutout augmentation')
parser.add_argument('-e', '--epochs', default=300, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-ek', '--efficient_k', default=0, type=int, metavar='N',
                    help='Efficient Net variant (0-8)')

parser.add_argument('--opt', default='sgd', type=str, metavar='OPTIMIZER',
                    help='Optimizer (default: "sgd"')
parser.add_argument('--opt-eps', default=None, type=float, metavar='EPSILON',
                    help='Optimizer Epsilon (default: None, use opt default)')
parser.add_argument('-lr', '--learning_rate', default=0.1, type=float,
                    help='initial learning rate', dest='lr')
parser.add_argument('-wd', '--weight-decay', default=5e-5, type=float,
                    metavar='W', help='weight decay (default: 5e-5)',
                    dest='weight_decay')

parser.add_argument('--reprob', type=float, default=0., metavar='PCT',
                    help='Random erase prob (default: 0.)')
parser.add_argument('--remode', type=str, default='const',
                    help='Random erase mode (default: "const")')
parser.add_argument('--recount', type=int, default=1,
                    help='Random erase count (default: 1)')
parser.add_argument('--resplit', action='store_true', default=False,
                    help='Do not random erase first (clean) augmentation split')

# Learning rate schedule parameters
parser.add_argument('--sched', default='step', type=str, metavar='SCHEDULER',
                    help='LR scheduler (default: "step"')
#parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
#                    help='learning rate (default: 0.01)')
parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                    help='learning rate noise on/off epoch percentages')
parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                    help='learning rate noise limit percent (default: 0.67)')
parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                    help='learning rate noise std-dev (default: 1.0)')
parser.add_argument('--lr-cycle-mul', type=float, default=1.0, metavar='MULT',
                    help='learning rate cycle len multiplier (default: 1.0)')
parser.add_argument('--lr-cycle-limit', type=int, default=1, metavar='N',
                    help='learning rate cycle limit')
parser.add_argument('--warmup-lr', type=float, default=0.0001, metavar='LR',
                    help='warmup learning rate (default: 0.0001)')
parser.add_argument('--min-lr', type=float, default=1e-5, metavar='LR',
                    help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')
#parser.add_argument('--epochs', type=int, default=200, metavar='N',
#                    help='number of epochs to train (default: 2)')
parser.add_argument('--epoch-repeats', type=float, default=0., metavar='N',
                    help='epoch repeat multiplier (number of times to repeat dataset epoch per train epoch).')
parser.add_argument('--start-epoch', default=None, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--decay-epochs', type=float, default=30, metavar='N',
                    help='epoch interval to decay LR')
parser.add_argument('--warmup-epochs', type=int, default=3, metavar='N',
                    help='epochs to warmup LR, if scheduler supports')
parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',
                    help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
parser.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                    help='patience epochs for Plateau LR scheduler (default: 10')
parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                    help='LR decay rate (default: 0.1)')


parser.add_argument('--amp', action='store_true', default=False,
                    help='use NVIDIA Apex AMP or Native AMP for mixed precision training')

parser.add_argument('-b', '--batch-size', default=32, type=int,
                    metavar='N',
                    help='mini-batch size (default: 32), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('-tb', '--test-batch-size', default=512, type=int,
                    metavar='N',
                    help='[test] mini-batch size (default: 512)')

parser.add_argument('--n1', default=64, type=int,
                    help='number of total filters in first convolutional layer (CIFAR-10/100)')
parser.add_argument('--n2', default=128, type=int,
                    help='number of total filters in second convolutional layer (CIFAR-10/100)')
parser.add_argument('--n3', default=128, type=int,
                    help='number of total filters in third convolutional layer (CIFAR-10/100)')
parser.add_argument('--n4', default=128, type=int,
                    help='number of total filters in third convolutional layer (CIFAR-10/100)')


# Model Exponential Moving Average
parser.add_argument('--model-ema', action='store_true', default=False,
                    help='Enable tracking moving average of model weights')
parser.add_argument('--model-ema-force-cpu', action='store_true', default=False,
                    help='Force ema to be tracked on CPU, rank=0 node only. Disables EMA validation.')
parser.add_argument('--model-ema-decay', type=float, default=0.9998,
                    help='decay factor for model weights moving average (default: 0.9998)')

parser.add_argument('--log_dir', type=str, default='./logs/', help='Log Directory')
parser.add_argument('--model_dir', type=str, default='./models/', help='Model Checkpoint Directory')

args = parser.parse_args()
args.learning_rate = args.lr
print('args = ', args)


if not os.path.exists(args.log_dir):
    os.makedirs( args.log_dir )
if not os.path.exists(args.model_dir):
    os.makedirs( args.model_dir )

n_class = 10
exp_name = args.dataset + '-' + args.model + '-e-'+str(args.epochs) + '-lr-'+str(args.lr)+ '-m-'+str(args.resnet_m)+\
           '-wd-'+ str(args.weight_decay) + '-b-'+str(args.batch_size)+'-K-' + str(args.K)+'-' +\
           '-dxy-' + str(int(args.constant_Dxy)) +\
           '-silu-' + str(int(args.use_silu)) +\
           '-nof-' + str(int(args.no_f)) +\
           '-dt-' + str(args.dt) +\
           '-dx-' + str(args.dx) +\
           '-dy-' + str(args.dy) +\
           '-cDx-' + str(args.cDx) +\
           '-cDy-' + str(args.cDy) +\
           '-h0_h-' + str(int(args.init_h0_h)) 

if args.pde_state != 0:
    exp_name += '-pde-' + str(args.pde_state) + '-'
if args.dataset in ['CIFAR-10', 'CIFAR-100']:
    exp_name += str(args.n1) + '-'+str(args.n2) + '-'+str(args.n3)+ '-'+str(args.n4)+'-sep-'+str(args.separable)
    if args.model in ['WideResnet', 'WideResnet-Global'] : 
        exp_name += '-wide-' + str(args.width)
    if args.non_linear:
        exp_name += '-nonlin-'
elif args.dataset == 'MNIST-10':
    exp_name += str(args.n1) + str(args.separable)
elif args.dataset == 'Imagenet-1000':
    if args.model in ['EfficientNet', 'EfficientNet-Global']:
        exp_name += 'B-'+str(args.efficient_k) 

logger = logging.getLogger('trainer')

file_log_handler = logging.FileHandler( './logs/logfile-' + exp_name + '.log')
logger.addHandler(file_log_handler)

stderr_log_handler = logging.StreamHandler()
logger.addHandler(stderr_log_handler)

# nice output format
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_log_handler.setFormatter(formatter)
stderr_log_handler.setFormatter(formatter)

logger.setLevel( 'DEBUG' )
logger.info( args )

def compute_params_flops(logger, cnn, n_channels=3, n_size=32):
    dummy_input = torch.randn(1, n_channels, n_size, n_size)
    macs, params = profile(cnn, inputs=(dummy_input, ), verbose=False) #, custom_ops=custom_ops) 
    macs, params = clever_format([macs, params], "%.3f")
    message = 'macs, params = ' + str(macs) + ', ' + str(params)
    logger.info(message)

    #macs, params = get_model_complexity_info(cnn, (n_channels, n_size, n_size), as_strings=True,
    #                                       print_per_layer_stat=False, verbose=False)
    #macs, params = get_model_complexity_info(cnn, (n_channels, n_size, n_size), as_strings=True,
    #                                       print_per_layer_stat=True, verbose=True)
    #msg = '{:<30}  {:<8}'.format('Computational complexity: ', macs) + '\n'
    #msg += '{:<30}  {:<8}'.format('Number of parameters: ', params)
    #logger.info(msg)

    #summary(cnn, torch.zeros((1, n_channels, n_size, n_size)))


def get_architecture_for_dataset(args, n_class=10, aux=True):
    global_ft = '-Global' in args.model
    if args.dataset in ['CIFAR-10', 'CIFAR-100']:
        if args.model == 'Resnet':
            if args.original:
                cnn = resnet_original( num_classes=n_class, m=args.resnet_m, n1=args.n1, n2=args.n2, n3=args.n3, n4=args.n4, args=args )
            else:
                cnn = resnet32( num_classes=n_class, m=args.resnet_m, n1=args.n1, n2=args.n2, n3=args.n3, n4=args.n4, args=args )
        elif args.model == 'Resnet-Global':
            if args.original:
                cnn = pdenet_original( num_classes=n_class, m=args.resnet_m, n1=args.n1, n2=args.n2, n3=args.n3, n4=args.n4, args=args )
            else:
                cnn = pdenet( num_classes=n_class, m=args.resnet_m, n1=args.n1, n2=args.n2, n3=args.n3, n4=args.n4, args=args )
        elif args.model in [ 'Densenet', 'Densenet-Global' ]:
            cnn = DenseNet( num_classes=n_class, block_config=(args.n1, args.n2, args.n3), global_ft=global_ft, args=args )
        elif args.model in [ 'WideResnet', 'WideResnet-Global' ]:
            cnn = wide_resnet( num_classes=n_class, m=args.resnet_m, n1=args.n1, n2=args.n2, n3=args.n3, n4=args.n4, global_ft=global_ft, args=args )
        elif args.model in [ 'DARTS', 'DARTS-Global' ]:
            from collections import namedtuple
            Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')

            DARTS_V2 = Genotype(normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 1), ('skip_connect', 0), ('skip_connect', 0), ('dil_conv_3x3', 2)], normal_concat=[2, 3, 4, 5], reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('skip_connect', 2), ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('skip_connect', 2), ('skip_connect', 2), ('max_pool_3x3', 1)], reduce_concat=[2, 3, 4, 5])
            cnn = DARTS_NetworkCIFAR(args.n1, n_class, args.resnet_m, True, DARTS_V2, args=args, global_ft=global_ft )
        else:
            raise ValueError('Incorrect model architecture.')
    elif args.dataset == 'MNIST-10':
        from odenet_mnist import  get_odenet_model
        if args.model in [ 'm_resnet', 'm_odenet' ]:
            cnn = get_odenet_model( args.model )
        elif args.model == 'm_global':
            cnn = get_odenet_model( args.model, args=args )
        else:
            raise ValueError('Incorrect model architecture.')
    else:
        raise ValueError('Dataset not supported.')

    return cnn

if args.dataset in ['CIFAR-10', 'CIFAR-100']:
    n_channels, n_size = 3, 32
    train_aug_list = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]
    if args.cutout:
        print('Enabling cutout data augmentation in CIFAR-10.')
        train_aug_list.append( Cutout(n_holes=args.n_holes, length=args.length) )

    transform_train = transforms.Compose(train_aug_list)

    # Normalize the test set same as training set without augmentation
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    if args.dataset == 'CIFAR-10':
        trainset = torchvision.datasets.CIFAR10(root=args.data, train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR10(root=args.data, train=False, download=True, transform=transform_test)
        n_class = 10
    elif args.dataset == 'CIFAR-100':
        trainset = torchvision.datasets.CIFAR100(root=args.data, train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR100(root=args.data, train=False, download=True, transform=transform_test)
        n_class = 100

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=2)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, pin_memory=True, num_workers=2)
    
elif args.dataset == 'MNIST-10':
    n_channels, n_size, n_class = 1, 28, 10

    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    trainset = torchvision.datasets.MNIST(root=args.data, train=True, download=True, transform=transform)
    testset = torchvision.datasets.MNIST(root=args.data, train=False, transform=transform)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=1)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, pin_memory=True, num_workers=1)
else:
    print('Not supported currently.')
    exit(1)

cnn = get_architecture_for_dataset(args, n_class, aux=False)
compute_params_flops(logger, cnn, n_channels=n_channels, n_size=n_size)
del cnn

cnn = get_architecture_for_dataset(args, n_class, aux=True)
cnn = cnn.cuda()

criterion = nn.CrossEntropyLoss().cuda()
criterion_smooth = CrossEntropyLabelSmooth().cuda()
cnn_optimizer = torch.optim.SGD(cnn.parameters(), lr=args.lr, momentum=0.9, nesterov=True, weight_decay=args.weight_decay)
scheduler = MultiStepLR(cnn_optimizer, milestones=[150, 225], gamma=0.1)

if args.dataset == 'MNIST-10':
    scheduler = MultiStepLR(cnn_optimizer, milestones=[60, 100, 140], gamma=0.1)

if args.model in ['DARTS', 'DARTS-Global']:
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(cnn_optimizer, float(args.epochs))

def test(loader, cnn, model_name):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(loader),
        [batch_time, losses, top1, top5],
        prefix='Test('+ model_name + '): ', logger=logger)

    cnn.eval()    # Change model to 'eval' mode (BN uses moving mean/var).
    correct = 0.
    total = 0.
    end = time.time()

    aux_available = False
    if args.model not in ['Resnet-Global', 'Resnet', 'Resnet-Global-Res', 'Densenet', 'Densenet-Global', 'Compact', 'WideResnet', 'WideResnet-Global' , 'm_resnet', 'm_odenet', 'm_global' ]:
        aux_available = True

    for i, (images, labels) in enumerate(loader):
        images = images.cuda()
        labels = labels.cuda()

        with torch.no_grad():
                if not aux_available: #args.model in ['Resnet-Global', 'Resnet']:
                    pred = cnn(images)
                else:
                    pred, _ = cnn(images)

        loss = criterion(pred, labels)

        # measure accuracy and record loss
        acc1, acc5 = get_accuracy(pred, labels, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)

        pred = torch.max(pred.data, 1)[1]
        total += labels.size(0)
        correct += (pred == labels).sum().item()
    progress.display(i)

    message = ' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'.format(top1=top1, top5=top5)
    # this should also be done with the ProgressMeter
    print(message)
    logger.info(message)

    val_acc = correct / total
    cnn.train()
    return val_acc

start_epoch = 0
if args.dataset == 'Imagenet-1000' and args.restart:
    suffix = './models/' + exp_name 
    if args.restart_known:
        suffix = args.restart_model_name
        
    msg = 'loading from ' + suffix +'-checkpoint.pth.tar'
    logger.info( msg )

    state = torch.load(suffix + '-checkpoint.pth.tar')
    
    start_epoch = state['epoch']
    best_acc1 = state['best_acc1']
    cnn.load_state_dict(state['state_dict'])
    cnn_optimizer.load_state_dict(state['optimizer'])

    if args.amp:
        loss_scaler.load_state_dict( state['loss_scaler.state_dict'] )

    acc1 = test(test_loader, cnn, model_name='Original')
    msg = 'best_acc1 = ' + str( best_acc1) + ' -- current_acc = ' + str(acc1) 
    logger.info( msg )

#total_params = sum(p.numel() for p in cnn.parameters() if p.requires_grad)
#print('Total params = ', total_params)
best_acc1 = 0.0
drop_path_prob = 0.2
for epoch in range(start_epoch, args.epochs):

    if args.model in ['DARTS', 'DARTS-Global']:
        cnn.drop_path_prob = drop_path_prob * epoch / args.epochs

    xentropy_loss_avg = 0.
    correct = 0.
    total = 0.

    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch),
        logger=logger)

    xcriterion = criterion
    if args.dataset == 'Imagenet-1000':
        xcriterion = criterion_smooth

    end = time.time()
    train_loader_len = len(train_loader)
    for i, (images, labels) in enumerate(train_loader):

        if args.dataset == 'Imagenet-1000':
            if scheduler is None:
                adjust_learning_rate(cnn_optimizer, epoch, i, train_loader_len, args)

        data_time.update(time.time() - end)

        #progress_bar.set_description('Epoch ' + str(epoch))

        images = images.cuda()
        labels = labels.cuda()

        cnn.zero_grad()

        aux_available = False
        if args.model not in ['Resnet-Global', 'Resnet', 'Resnet-Global-Res', 'Densenet', 'Densenet-Global', 'Compact', 'WideResnet', 'WideResnet-Global', 'm_resnet', 'm_odenet', 'm_global' ]:
            aux_available = True
	
        if not aux_available: #args.model in ['Resnet-Global', 'Resnet']:
                pred = cnn(images)
                xentropy_loss = xcriterion(pred, labels) #+ 0.4*criterion(aux_pred, labels)      
        else:
                pred, aux_pred = cnn(images)
                xentropy_loss = xcriterion(pred, labels) + 0.4 * xcriterion(aux_pred, labels)      
        
        # measure accuracy and record loss
        acc1, acc5 = get_accuracy(pred, labels, topk=(1, 5))
        losses.update(xentropy_loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))
        
        xentropy_loss_avg += xentropy_loss.item()
        
        xentropy_loss.backward()
        if args.model in ['DARTS', 'DARTS-Global']:
                torch.nn.utils.clip_grad_norm_(cnn.parameters(), 5.0)
        cnn_optimizer.step()

        # Calculate running average of accuracy
        pred = torch.max(pred.data, 1)[1]
        total += labels.size(0)
        correct += (pred == labels.data).sum().item()
        accuracy = correct / total

        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)

    progress.display(i)

    test_acc = test(test_loader, cnn, model_name='Original')

    scheduler.step()     # Use this line for PyTorch >=1.4
    
    # remember best acc@1 and save checkpoint
    acc1 = test_acc
    is_best = acc1 > best_acc1
    best_acc1 = max(acc1, best_acc1)

    model_train_dict = {
            'epoch': epoch + 1,
            'arch': "GlobalLayer-" + args.model,
            'state_dict': cnn.state_dict(),
            'best_acc1': best_acc1,
            'optimizer' : cnn_optimizer.state_dict(),
    }

    save_checkpoint(model_train_dict, is_best, EXP_NAME='./models/' + exp_name)

