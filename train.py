'''
Training script for CIFAR-10/100
Copyright (c) Wei YANG, 2017
'''
from __future__ import print_function

import argparse
import os
import shutil
import time
import random
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision import transforms, utils

from models import SqueezeNet, F
from dataset.FaceLandmarksDataset import Rescale, RandomBrightness, RandomCrop, RandomFlip, \
        RandomContrast, RandomLightingNoise, Normalize, ToTensor, FaceLandmarksDataset

from utils import Bar, Logger, AverageMeter,normalizedME, mkdir_p, savefig

parser = argparse.ArgumentParser(description='PyTorch face landmark Training')
# Datasets
parser.add_argument('-d', '--dataset', default='face76', type=str)
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
# Optimization options
parser.add_argument('--epochs', default=72, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--train-batch', default=128, type=int, metavar='N',
                    help='train batchsize')
parser.add_argument('--test-batch', default=128, type=int, metavar='N',
                    help='test batchsize')
parser.add_argument('--lr', '--learning-rate', default=0.0001, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--drop', '--dropout', default=0.3, type=float,
                    metavar='Dropout', help='Dropout ratio')
parser.add_argument('--schedule', type=int, nargs='+', default=[60,100],
                        help='Decrease learning rate at these epochs.')
parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule.')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
# Checkpoints
parser.add_argument('-c', '--checkpoint', default='checkpoint/1011/', type=str, metavar='PATH',
                    help='path to save checkpoint (default: checkpoint)')
#parser.add_argument('--resume', default='/home/foto1/workspace/zuoxin/face_landmark/checkpoint/0918/facelandmark_squeezenet_128_55.pth.tar', type=str, metavar='PATH',
#                    help='path to latest checkpoint (default: none)')
parser.add_argument('--resume', default='checkpoint/1011/facelandmark_squeezenet_64_67.pth.tar', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
# Architecture
parser.add_argument('--depth', type=int, default=104, help='Model depth.')
parser.add_argument('--cardinality', type=int, default=8, help='Model cardinality (group).')
parser.add_argument('--widen-factor', type=int, default=4, help='Widen factor. 4 -> 64, 8 -> 128, ...')
parser.add_argument('--growthRate', type=int, default=12, help='Growth rate for DenseNet.')
parser.add_argument('--compressionRate', type=int, default=2, help='Compression Rate (theta) for DenseNet.')
# Miscs
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
#Device options
parser.add_argument('--gpu_id', default='0', type=str,
help='id(s) for CUDA_VISIBLE_DEVICES')

args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}


# Use CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
use_cuda = torch.cuda.is_available()

# Random seed
if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
if use_cuda:
    torch.cuda.manual_seed_all(args.manualSeed)

best_acc= 99  # best test accuracy
start_epoch = 0
fig, ax = plt.subplots()
nx = 224
ny = 224
sc = ax.scatter(nx,ny, s=10, marker='.', c='r')
sc1 = ax.scatter(nx,ny, s=10, marker='.', c='g')
trash_data = np.zeros((nx, ny))
im = plt.imshow(trash_data, cmap='gist_gray_r', vmin=0, vmax=1)

def main():
    global best_acc
    global start_epoch
    start_epoch = args.start_epoch  # start from epoch 0 or last checkpoint epoch

    if not os.path.isdir(args.checkpoint):
        mkdir_p(args.checkpoint)

    # Data
    print('==> Preparing dataset %s' % args.dataset)
    transform_train = transforms.Compose([
        Rescale((250,250)),
        RandomCrop((224,224)),
        #RandomFlip(),
        #RandomContrast(),
        #RandomBrightness(),
        #RandomLightingNoise(),
        Normalize(),
        ToTensor(),
    ])

    transform_test = transforms.Compose([
        #SmartRandomCrop(),
        Rescale((224,224)),
        Normalize(),
        ToTensor(),
    ])

    trainset = FaceLandmarksDataset(csv_file='dataset/dd/training_frames_keypoints.csv', transform=transform_train,root_dir='dataset/dd/training')
    trainloader = data.DataLoader(trainset, batch_size=args.train_batch, shuffle=True, num_workers=args.workers, pin_memory=True)

    testset = FaceLandmarksDataset(csv_file='dataset/dd/test_frames_keypoints.csv', transform=transform_test,root_dir='dataset/dd/test')
    testloader = data.DataLoader(testset, batch_size=args.train_batch, shuffle=True, num_workers=args.workers, pin_memory=True)
    model = SqueezeNet(136).cuda()
    cudnn.benchmark = True
    print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))

    criterion = nn.MSELoss().cuda()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    ignored_params = list(map(id, model.fc.parameters()))
    base_params = filter(lambda p: id(p) not in ignored_params,model.parameters())
    params = [
        {'params': base_params, 'lr': args.lr},
        {'params': model.fc.parameters(), 'lr': args.lr * 10}
    ]
    #model = model.cuda()
    optimizer = optim.Adam(params=params, lr=args.lr, weight_decay=args.weight_decay)

    # Resume
    title = 'facelandmark_squeezenet_64'
    if True:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isfile(args.resume), 'Error: no checkpoint directory found!'
        args.checkpoint = os.path.dirname(args.resume)
        checkpoint = torch.load(args.resume)
        best_acc = checkpoint['best_acc']
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        if os.path.exists(os.path.join(args.checkpoint, title+'_log.txt')):
            logger = Logger(os.path.join(args.checkpoint, title+'_log.txt'), title=title, resume=True)
        else:
            logger = Logger(os.path.join(args.checkpoint, title+'_log.txt'), title=title)
            logger.set_names(['Learning Rate', 'Train Loss', 'Valid Loss', 'Train Acc.', 'Valid Acc.'])
    else:
        logger = Logger(os.path.join(args.checkpoint, title+'_log.txt'), title=title)
        logger.set_names(['Learning Rate', 'Train Loss', 'Valid Loss', 'Train Acc.', 'Valid Acc.'])


    if args.evaluate:
        print('\nEvaluation only')
        test_loss, test_acc , _ , __ , ___= test(testloader, model, criterion, start_epoch, use_cuda)
        print(' Test Loss:  %.8f, Test Acc:  %.2f' % (test_loss, test_acc))
        return

    # Train and val
    current_epoch = start_epoch
    print(args.epochs - current_epoch)
    anim = animation.FuncAnimation(fig, animate, fargs=(current_epoch, model, criterion, optimizer, trainloader, testloader, logger, best_acc, title,), frames=(args.epochs - current_epoch - 2),  interval=100, blit=True, repeat=False)
    plt.show()
    
    logger.close()
    #logger.plot()
    savefig(os.path.join(args.checkpoint, 'log.eps'))

    print('Best acc:')
    print(best_acc)

def train(trainloader, model, criterion, optimizer, epoch, use_cuda):
    # switch to train mode
    model.train()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    #NormMS = AverageMeter()
    end = time.time()

    bar = Bar('Processing', max=len(trainloader))
    for batch_idx, batch_data in enumerate(trainloader):
        # measure data loading time
        data_time.update(time.time() - end)
        inputs = batch_data['image']
        targets = batch_data['landmarks']
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda(async=True)
        inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)
        #if batch_idx == 3:
        #            im_temp, sc_temp = show_landmarks_batch({'image': inputs, 'outputs_landmarks': targets ,'targets_landmarks': targets})

        # compute output
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # measure accuracy and record loss
        #nms= normalizedME(outputs.data,targets.data,64,64)
        losses.update(loss.item(), inputs.size(0))
        #NormMS.update(nms[0], inputs.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f}'.format(
                    batch=batch_idx + 1,
                    size=len(trainloader),
                    data=data_time.avg,
                    bt=batch_time.avg,
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    loss=losses.avg,
                    )
        bar.next()
    bar.finish()
    return (losses.avg,0)

def test(testloader, model, criterion, epoch, use_cuda):
    global best_acc

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    #retuen im in data
    # switch to evaluate mode
    model.eval()
    with torch.no_grad():
        total_accuracy = 0.0
        end = time.time()
        bar = Bar('Processing', max=len(testloader))
        for batch_idx, batch_data in enumerate(testloader):
            # measure data loading time
            data_time.update(time.time() - end)

            inputs = batch_data['image']
            targets = batch_data['landmarks']
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda(async=True)
            inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)

            # compute output
            outputs = model(inputs)
            if batch_idx == 5:
                im_temp, sc_temp, sc1_temp = show_landmarks_batch({'image': inputs, 'outputs_landmarks': outputs, 'targets_landmarks': targets})
            
            loss = criterion(outputs, targets)

            # measure accuracy and record loss
            losses.update(loss.item(), inputs.size(0))

            total_accuracy += F.mse_loss(outputs, targets)
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # plot progress
            bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} '.format(
                        batch=batch_idx + 1,
                        size=len(testloader),
                        data=data_time.avg,
                        bt=batch_time.avg,
                        total=bar.elapsed_td,
                        eta=bar.eta_td,
                        loss=losses.avg,
                        )
            bar.next()
        bar.finish()
    return (losses.avg, total_accuracy, im_temp, sc_temp, sc1_temp)

def trainAndValidaion (epoch, model, criterion, optimizer, trainloader, testloader, logger, best_acc, title):
    adjust_learning_rate(optimizer, epoch)

    print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, args.epochs, state['lr']))

    train_loss, train_acc = train(trainloader, model, criterion, optimizer, epoch, use_cuda)
    test_loss, test_acc, im_temp, sc_temp, sc1_temp = test(testloader, model, criterion, epoch, use_cuda)

    # append logger file
    logger.append([state['lr'], train_loss, test_loss, train_acc, test_acc])

    # save model
    print(test_acc.data.cpu().numpy())
    is_best = test_acc.data.cpu().numpy() < best_acc
    best_acc = min(test_acc.data.cpu().numpy(), best_acc)
    save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'acc': test_acc,
            'best_acc': best_acc,
            'optimizer' : optimizer.state_dict(),
        }, is_best, checkpoint=args.checkpoint,filename=title+'_'+str(epoch)+'.pth.tar')
    global start_epoch
    start_epoch = start_epoch + 1
    return [im_temp, sc_temp, sc1_temp]

def show_landmarks_batch(sample_batched):
    """Show image with landmarks for a batch of samples."""
    #plt.ion()

    number_of_plotted_images = 1
    images_batch, outputs_landmarks_batch, targets_landmarks_batch = \
    sample_batched['image'][0: (number_of_plotted_images), :, :, :], sample_batched['outputs_landmarks'][0: (number_of_plotted_images), :, :], sample_batched['targets_landmarks'][0: (number_of_plotted_images), :, :]
    
    batch_size = len(images_batch)
    im_size = images_batch.size(2)

    grid = utils.make_grid(images_batch)
    im.set_data(grid.cpu().numpy().transpose((1, 2, 0)))
    for i in range(0, batch_size):
        outputs_landmarks_batch_single = outputs_landmarks_batch[i]
        targets_landmarks_batch_single = targets_landmarks_batch[i]

        outputs_landmarks_batch_single = outputs_landmarks_batch_single.view(-1, 2)
        sc.set_offsets(np.c_[(outputs_landmarks_batch_single[:, 0].cpu().detach().numpy() * 50 + 100) + i * im_size,
                    outputs_landmarks_batch_single[:, 1].cpu().detach().numpy() * 50 + 100])
        
        targets_landmarks_batch_single = targets_landmarks_batch_single.view(-1, 2)
        sc1.set_offsets(np.c_[(targets_landmarks_batch_single[:, 0].cpu().detach().numpy() * 50 + 100) + i * im_size,
                    targets_landmarks_batch_single[:, 1].cpu().detach().numpy() * 50 + 100])

        plt.title('Batch from dataloader')
    return im, sc , sc1


def save_checkpoint(state, is_best, checkpoint='checkpoint', filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    if is_best:
        torch.save(state, filepath)
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))

def adjust_learning_rate(optimizer, epoch):
    global state
    if epoch in args.schedule:
        state['lr'] *= args.gamma
        for param_group in optimizer.param_groups:
            param_group['lr'] = state['lr']

def animate(i, epoch, model, criterion, optimizer, trainloader, testloader, logger, best_acc, title):
    global start_epoch
    epoch = start_epoch
    return trainAndValidaion(epoch, model, criterion, optimizer, trainloader, testloader, logger, best_acc, title)

if __name__ == '__main__':
    main()