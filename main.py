# -*- coding: utf-8 -*-
from __future__ import print_function
import argparse
import math
import matplotlib.pyplot as plt
import os

import torch
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

from model import LeNet_standard, LeNet_dropout

# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--mode', type=int, default=0, metavar='N',
                    help='train mode (0) test mode (1)'
                    'uncertainty test mode (2) (default: 0)')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=30, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='interval of logging training status')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)


kwargs = {'num_workers': 4, 'pin_memory': True} if args.cuda else {}
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=args.test_batch_size, shuffle=True, **kwargs)


def train(model, opt, epoch):
    model.train()
    lr = args.lr * (0.1 ** (epoch // 10))
    opt.param_groups[0]['lr'] = lr
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        opt.zero_grad()
        output = model(data)
        loss = F.nll_loss(F.log_softmax(output), target)
        loss.backward()
        opt.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)] lr: {}\tLoss: {:.6f}'
                  .format(epoch, batch_idx * len(data),
                          len(train_loader.dataset),
                          100. * batch_idx / len(train_loader),
                          lr, loss.data[0]))


def test(model):
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        test_loss += F.nll_loss(F.log_softmax(output), target, size_average=False).data[0]  # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def mcdropout_test(model):
    model.train()
    test_loss = 0
    correct = 0
    T = 100
    for data, target in test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output_list = []
        for i in xrange(T):
            output_list.append(torch.unsqueeze(model(data), 0))
        output_mean = torch.cat(output_list, 0).mean(0)
        test_loss += F.nll_loss(F.log_softmax(output_mean), target, size_average=False).data[0]  # sum up batch loss
        pred = output_mean.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('\nMC Dropout Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def uncertainty_test(model):
    model.train()
    T = 100
    rotation_list = range(0, 180, 10)
    for data, target in test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output_list = []
        image_list = []
        unct_list = []
        for r in rotation_list:
            rotation_matrix = Variable(torch.Tensor([[[math.cos(r/360.0*2*math.pi), -math.sin(r/360.0*2*math.pi), 0],
                                                    [math.sin(r/360.0*2*math.pi), math.cos(r/360.0*2*math.pi), 0]]]).cuda(),
                                       volatile=True)
            grid = F.affine_grid(rotation_matrix, data.size())
            data_rotate = F.grid_sample(data, grid)
            image_list.append(data_rotate)

            for i in xrange(T):
                output_list.append(torch.unsqueeze(F.softmax(model(data_rotate)), 0))
            output_mean = torch.cat(output_list, 0).mean(0)
            output_variance = torch.cat(output_list, 0).var(0).mean().data[0]
            confidence = output_mean.data.cpu().numpy().max()
            predict = output_mean.data.cpu().numpy().argmax()
            unct_list.append(output_variance)
            print ('rotation degree', str(r).ljust(3), 'Uncertainty : {:.4f} Predict : {} Softmax : {:.2f}'.format(output_variance, predict, confidence))

        plt.figure()
        for i in range(len(rotation_list)):
            ax = plt.subplot(2, len(rotation_list)/2, i+1)
            plt.text(0.5, -0.5, "{0:.3f}".format(unct_list[i]),
                     size=12, ha="center", transform=ax.transAxes)
            plt.axis('off')
            plt.gca().set_title(str(rotation_list[i])+u'\xb0')
            plt.imshow(image_list[i][0, 0, :, :].data.cpu().numpy())
        plt.show()
        print ()


def main():

    model_standard = LeNet_standard()
    model_dropout = LeNet_dropout()
    if args.cuda:
        model_standard.cuda()
        model_dropout.cuda()

    if not os.path.exists('checkpoint'):
        os.makedirs('checkpoint')

    if args.mode == 0:
        optimizer_standard = optim.SGD(model_standard.parameters(), lr=args.lr, momentum=args.momentum)
        optimizer_dropout = optim.SGD(model_dropout.parameters(), lr=args.lr, momentum=args.momentum)

        print ('Train standard Lenet')
        for epoch in range(1, args.epochs + 1):
            train(model_standard, optimizer_standard, epoch)
        test(model_standard)

        print ('Train Lenet with dropout at all layer')
        for epoch in range(1, args.epochs + 1):
            train(model_dropout, optimizer_dropout, epoch)
        mcdropout_test(model_dropout)

        print ('Save checkpoint/'+'LeNet_stadard'+str(epoch)+'.pth.tar')
        state = {'state_dict': model_standard.state_dict()}
        filename = 'checkpoint/'+'LeNet_stadard'+str(epoch)+'.pth.tar'
        torch.save(state, filename)

        print ('Save checkpoint/'+'LeNet_dropout'+str(epoch)+'.pth.tar')
        state = {'state_dict': model_dropout.state_dict()}
        filename = 'checkpoint/'+'LeNet_dropout'+str(epoch)+'.pth.tar'
        torch.save(state, filename)

    elif args.mode == 1:
        ckpt_standard = torch.load('checkpoint/LeNet_stadard30.pth.tar')
        model_standard.load_state_dict(ckpt_standard['state_dict'])
        test(model_standard)

        ckpt_dropout = torch.load('checkpoint/LeNet_dropout30.pth.tar')
        model_dropout.load_state_dict(ckpt_dropout['state_dict'])
        mcdropout_test(model_dropout)

    elif args.mode == 2:
        ckpt_dropout = torch.load('checkpoint/LeNet_dropout30.pth.tar')
        model_dropout.load_state_dict(ckpt_dropout['state_dict'])
        uncertainty_test(model_dropout)
    else:
        print ('--mode argument is invalid \ntrain mode (0) or test mode (1) uncertainty test mode (2)')


if __name__ == '__main__':
    main()
