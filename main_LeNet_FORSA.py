# -*- coding: utf-8 -*-
"""
Created on Mon Apr  3 20:06:17 2023

@author: Huruy LeNet_FORSA
"""

#%% Importing libraries
import numpy as np
from fxpmath import Fxp
from numpy import random
from numpy import *

from utils_FORSA import *
from torchinfo import summary
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data

import torchvision.transforms as transforms
import torchvision.datasets as datasets

from sklearn import decomposition
from sklearn import manifold
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
#from tqdm.notebook import tqdm, trange
from tqdm import tqdm, trange
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import datetime
from nvitop import CudaDevice
from nvitop import Device, ResourceMetricCollector, collect_in_background
from nvitop.callbacks.tensorboard import add_scalar_dict
writer = SummaryWriter()
#collector = ResourceMetricCollector(devices=CudaDevice.all(),  # log all visible CUDA devices and use the CUDA ordinal
#                                    root_pids={os.getpid()},   # only log the descendant processes of the current process
#                                    interval=1.0)              # snapshot interval for background daemon thread

ResourceMetricCollector(Device.cuda.all()).daemonize(
    on_collect,
    interval=1.0,
    on_stop=on_stop,
)

import copy
import random
import time

import scipy.io
import pandas as pd
import argparse

parser = argparse.ArgumentParser(description='PyTorch MNIST Nemo')
parser.add_argument('--epochs', default=30, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--batchsize', '--batch-size', default=64, type=int, metavar='N', help='mini-batch size (default: 128),only used for train')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float, metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true', help='evaluate model on validation set')
parser.add_argument('--init', help='initialize form pre-trained floating point model', type=str, default='')
parser.add_argument('--bit', default=8, type=int, help='the bit-width of the quantized network')
parser.add_argument('--word_size', default=8, type=int, help='Number of bits required to represent a number')
parser.add_argument('--frac_size', default=6, type=int, help='Number of bits required to represent a number after decimal place')
parser.add_argument('--train', default=False, type=str, help='True for training and fine-tuning')
args = parser.parse_args()


########################### Loading dataset ###########################
SEED = 1234

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True


def main():

    global args, best_prec
    
    kwargs = {'num_workers': 8, 'pin_memory': True} if torch.cuda.is_available() else {}
    if args.frac_size == 24: #This frac size needs to be changed for every run based on bit width
        data_sw = {
        'Experiment (#Switching)': ['Layer 1 before', 'Layer 1 after', 'Layer 2 before', 'Layer 2 after',
                                    'Layer 3 FC  before', 'Layer 3 FC  after', 'Layer 4 FC before', 'Layer 4 FC after'],
        'sw_' + str(args.bit) + '_' + str(args.frac_size): [0., 0., 0., 0., 0., 0., 0., 0.]}
        row_labels = [0, 1, 2, 3, 4, 5, 6, 7]
        df = pd.DataFrame(data=data_sw, index=row_labels)
    else:
        data_sw = pd.read_csv('./FORSA_sw_activities.csv')
        data_sw['sw_' + str(args.bit) + '_' + str(args.frac_size)] = [0., 0., 0., 0., 0., 0., 0., 0.]
        df = pd.DataFrame(data=data_sw)


    
    
    print(df)
    ROOT = './data'

    train_data = datasets.MNIST(root=ROOT,
                                train=True,
                                download=True)

    mean_train_data = train_data.data.float().mean() / 255
    std_train_data = train_data.data.float().std() / 255

    #print(f'Calculated mean: {mean_train_data}')
    #print(f'Calculated std: {std_train_data}')

    train_transforms = transforms.Compose([
                                transforms.RandomRotation(5, fill=(0,)),
                                transforms.RandomCrop(28, padding=2),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[mean_train_data], std=[std_train_data])
                                          ])

    test_transforms = transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize(mean=[mean_train_data], std=[std_train_data])
                                         ])
    train_data = datasets.MNIST(root=ROOT,
                                train=True,
                                download=True,
                                transform=train_transforms)

    test_data = datasets.MNIST(root=ROOT,
                               train=False,
                               download=True,
                               transform=test_transforms)
    #%% Dataset preprocessing 

    VALID_RATIO = 0.9

    n_train_examples = int(len(train_data) * VALID_RATIO)
    n_valid_examples = len(train_data) - n_train_examples

    train_data, valid_data = data.random_split(train_data,
                                               [n_train_examples, n_valid_examples])

    valid_data = copy.deepcopy(valid_data)
    valid_data.dataset.transform = test_transforms

    print(f'Number of training examples: {len(train_data)}')
    print(f'Number of validation examples: {len(valid_data)}')
    print(f'Number of testing examples: {len(test_data)}')

    #%% 
    BATCH_SIZE = args.batchsize #64

    train_iterator = data.DataLoader(train_data,
                                     shuffle=True,
                                     batch_size=BATCH_SIZE)

    valid_iterator = data.DataLoader(valid_data,
                                     batch_size=BATCH_SIZE)

    test_iterator = data.DataLoader(test_data,
                                    batch_size=BATCH_SIZE)

    #%% 
    OUTPUT_DIM = 10

    model = LeNet(OUTPUT_DIM)
    params = [print(param.shape) for param in model.parameters()]

    batch_size = 1
    summary(model, input_size=(batch_size, 1, 28, 28))
    print(f'The model has {count_parameters(model):,} trainable parameters')

    #Training config
    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # device = 'cpu'
    print(device)

    model = model.to(device)
    criterion = criterion.to(device)
    epochs = args.epochs #5

    # Load pretrained model
    if args.init:
        model.load_state_dict(torch.load('./Lenet5.pt'))
        # device = torch.device('cpu')
        test_loss, test_acc = evaluate(model, test_iterator, criterion, device)
        print(f'Test Loss of pretrained model: {test_loss:.3f} | Test Acc: {test_acc * 100:.2f}%')
    
    df_nvitop = pd.DataFrame()
    best_valid_loss = float('inf')
    if args.train:
        for epoch in trange(epochs, desc="Epochs"):
            with collector(tag='resources'):

                start_time = time.monotonic()

                train_loss, train_acc = train(model, train_iterator, optimizer, criterion, device)
                
                metrics = collector.collect()
                df_metrics = pd.DataFrame.from_records(metrics, index=[len(df_nvitop)])
                df_nvitop = pd.concat([df_nvitop, df_metrics], ignore_index=True)
                # Flush to CSV file ...
                
                valid_loss, valid_acc = evaluate(model, valid_iterator, criterion, device)

                if valid_loss < best_valid_loss:
                    best_valid_loss = valid_loss
                    torch.save(model.state_dict(), './Lenet5.pt')

                end_time = time.monotonic()

                epoch_mins, epoch_secs = epoch_time(start_time, end_time)

                print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
                print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
                print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')
        df.insert(0, 'time', df['resources/timestamp'].map(datetime.datetime.fromtimestamp))
        df.to_csv('results.csv', index=False)
        #%% Load pretrained model
        model.load_state_dict(torch.load('./Lenet5.pt'))
        # device = torch.device('cpu')
        test_loss, test_acc = evaluate(model, test_iterator, criterion, device)

        print(f'Test Loss before sorting: {test_loss:.3f} | Test Acc: {test_acc*100:.2f}%')
        #Sort filters
        #Sort filters and evaluate to compare results
        # word_size  = 8
        # frac_size = 6
        ################################################# LAYER 1 #######################
        conv1_weight = model.conv1.weight
        conv1_weight_reshaped=conv1_weight.reshape((conv1_weight.shape[0], conv1_weight.shape[2]*conv1_weight.shape[3]))
        sw_before = sum(count_switching(np.asarray(conv1_weight_reshaped.to('cpu').detach().numpy()), word_size=args.bit, frac_size=args.frac_size))
        print('Sw activity of Layer 1 before sorting', sw_before)
        df.loc[0, 'sw_' + str(args.bit) + '_' + str(args.frac_size)] = sw_before
        conv1_weight_reshaped, new_indx1 = sortFullMatrix(np.asarray(conv1_weight_reshaped.to('cpu').detach().numpy()), word_size=args.bit, frac_size=args.frac_size)
        sw_after = sum(count_switching(np.asarray(conv1_weight_reshaped), word_size=args.bit, frac_size=args.frac_size))
        print('Sw activity of Layer 1 after sorting', sw_after)
        df.loc[1, 'sw_' + str(args.bit) + '_' + str(args.frac_size)] = sw_after
        # new_indx1 = torch.tensor([ 3,  6, 23, 13,  8,  0, 18, 11, 21, 20, 14, 25, 17,  4, 31, 26, 16, 10, 30, 12,  9,  1, 19, 24,  2, 29, 22, 27, 28,  5, 15,  7])
        new_conv1_weight = model.conv1.weight[new_indx1, :, :, :] ########## To be used in the new model
        new_conv1_bias = model.conv1.bias[new_indx1]              ########## To be used in the new model
        
        ################################################## LAYER 2 #######################
        conv2_weight = model.conv2.weight
        # Using the new filter index to rearrange layer 2 filters (Channelwise rearrange)
        new_conv2_weight = model.conv2.weight[:, new_indx1, :, :] ########## To adjust the sorting the previous layer
        new_conv2_weight_reshaped = new_conv2_weight.reshape((new_conv2_weight.shape[0], new_conv2_weight.shape[1]*new_conv2_weight.shape[2]*new_conv2_weight.shape[3]))
        sw_before = sum(count_switching(np.asarray(new_conv2_weight_reshaped.to('cpu').detach().numpy()), word_size=args.bit, frac_size=args.frac_size))
        print('Sw activity of Layer 2 before sorting', sw_before)
        df.loc[2, 'sw_' + str(args.bit) + '_' + str(args.frac_size)] = sw_before
        new_conv2_weight_reshaped, new_indx2 = sortFullMatrix(np.asarray(new_conv2_weight_reshaped.to('cpu').detach().numpy()), word_size=args.bit, frac_size=args.frac_size)
        sw_after = sum(count_switching(new_conv2_weight_reshaped, word_size=args.bit, frac_size=args.frac_size))
        print('Sw activity of Layer 2 after sorting', sw_after)
        df.loc[3, 'sw_' + str(args.bit) + '_' + str(args.frac_size)] = sw_after
        # new_indx2 = torch.tensor([12,  9, 21,  7, 13,  1, 30, 31, 10, 26, 25,  3, 15,  8, 11, 19, 14, 24, 22, 16,  2,  4, 20, 28,  6, 17, 23, 18, 27, 29,  0,  5])
        new_conv2_weight = new_conv2_weight[new_indx2, :, :, :]
        new_conv2_bias = model.conv2.bias[new_indx2]

        ################################################## LAYER 3 FC_1 #######################
        #
        fc1_weight = model.fc_1.weight
        # fc1_weight = torch.reshape(fc1_weight, (120, 32, 4, 4)) # to be transposed to [120, 32, 4, 4]
        fc1_weight = fc1_weight.view(120, 32, 4, 4)
        new_fc1_weight = fc1_weight[:, new_indx2, :, :] ########## To adjust the sorting the previous layer
        # fc1_weight.reshape([120, 4, 4, 32]) # to be transposed to [120, 32, 4, 4]
        # new_indx3 = torch.randperm(fc1_weight.shape[0])

        # new_indx3 = torch.tensor([   10,  34,  97, 110,  71,  49,  48,   2, 100,  96,   3,   7,  41,  31,
        #                               50,  20,  89,   4,  52,  21,  86,  32,  30, 119,  92,  91,  76,  59,
        #                               38, 101,  95,  67, 109,  56,  47,  18, 114,  15, 118,  37,  61,  44,
        #                               42,  23,  68,  83,  94,  29,  80,  64, 107,  58,  79,  27,  60,  24,
        #                             106,   6,  12,  82,  19,   1,  77,  26,  57,  74,  99,  78,  25,  13,
        #                               65,  40,   9, 111,  70,  85,  22,  16,  36,  51,  81,  88, 113,  17,
        #                             103, 102, 115,  93,  11,  14,  54,  55, 108, 112,  75,  62,  35,   0,
        #                             117,  73, 104,   8,  33, 116,  43,  98,  69,  63, 105,  72,  39,  87,
        #                               5,  53,  46,  84,  90,  28,  45,  66])
        new_fc1_weight = new_fc1_weight.view(new_fc1_weight.shape[0], -1)   #torch.reshape(new_fc1_weight, (120, 512))
        sw_before = sum(count_switching(np.asarray(new_fc1_weight.to('cpu').detach().numpy()), word_size=args.bit, frac_size=args.frac_size))
        print('Sw activity of Layer 3 FC1 before sorting', sw_before)
        df.loc[4, 'sw_' + str(args.bit) + '_' + str(args.frac_size)] = sw_before
        new_fc1_weight_reshaped, new_indx3 = sortFullMatrix(np.asarray(new_fc1_weight.to('cpu').detach().numpy()), word_size=args.bit, frac_size=args.frac_size)
        sw_after = sum(count_switching(new_fc1_weight_reshaped, word_size=args.bit, frac_size=args.frac_size))
        print('Sw activity of Layer 3 FC1 after sorting', sw_after)
        df.loc[5, 'sw_' + str(args.bit) + '_' + str(args.frac_size)] = sw_after

        new_fc1_weight = new_fc1_weight[new_indx3, :]
        new_fc1_bias = model.fc_1.bias[new_indx3]

        ################################################## LAYER 4 FC_2 #######################
        fc2_weight = model.fc_2.weight
        # fc2_weight = torch.reshape(fc2_weight, (84, 120)) # to be transposed to [120, 32, 4, 4]
        new_fc2_weight = fc2_weight[:, new_indx3] ########## To adjust the sorting the previous layer
        # fc1_weight.reshape([120, 4, 4, 32]) # to be transposed to [120, 32, 4, 4]
        # new_indx4 = torch.randperm(fc2_weight.shape[0])
        # print(new_fc2_weight.shape)
        new_fc2_weight_reshaped = new_fc2_weight
        sw_before = sum(count_switching(np.asarray(new_fc2_weight.to('cpu').detach().numpy()), word_size=args.bit, frac_size=args.frac_size))
        print('Sw activity of Layer 4 FC_2 before sorting', sw_before)
        df.loc[6, 'sw_' + str(args.bit) + '_' + str(args.frac_size)] = sw_before
        new_fc2_weight_reshaped, new_indx4 = sortFullMatrix(np.asarray(new_fc2_weight.to('cpu').detach().numpy()), word_size=args.bit, frac_size=args.frac_size)
        sw_after = sum(count_switching(new_fc2_weight_reshaped, word_size=args.bit, frac_size=args.frac_size))
        print('Sw activity of Layer 4 FC_2 after sorting', sw_after)
        df.loc[7, 'sw_' + str(args.bit) + '_' + str(args.frac_size)] = sw_after

        # new_indx4 = torch.tensor([  60, 57, 46,  8, 14, 33, 17, 82, 44, 50, 62, 22, 11, 13,  1, 61, 10, 71,
        #                             45,  4, 78, 24, 66, 74, 35,  5, 36, 67,  7,  2, 20, 23, 26, 34, 42, 79,
        #                             77, 80, 29, 37, 38, 51, 25, 30, 32, 65, 70, 69, 58,  3, 47, 54, 83, 56,
        #                               9, 64, 81, 75, 40, 72, 28, 39, 59, 27, 16, 63, 15,  6, 52, 73, 55, 48,
        #                             76, 31, 18, 21, 68, 41, 19,  0, 12, 53, 43, 49])

        new_fc2_weight = new_fc2_weight[new_indx4, :]
        # new_fc2_weight = torch.reshape(new_fc2_weight, (84, 512))
        new_fc2_bias = model.fc_2.bias[new_indx4]
        # print(new_fc2_weight.shape)

        ################################################## LAYER 5 FC_3 #######################

        fc3_weight = model.fc_3.weight
        # fc3_weight = torch.reshape(fc3_weight, (10, 84)) # to be transposed to [120, 32, 4, 4]
        new_fc3_weight = fc3_weight[:, new_indx4] ########## To adjust the sorting the previous layer
        # new_fc3_weight_reshaped = new_fc3_weight
        # sw_before = count_switching(np.asarray(new_fc3_weight.to('cpu').detach().numpy()), word_size=8, frac_size=6)
        # print('Sw activity before sorting', sw_before)
        # df.loc[8, 'sw_' + str(args.bit) + str(args.frac_size)] = sw_before
        # new_fc3_weight_reshaped, new_indx5 = sortFullMatrix(np.asarray(new_fc3_weight.to('cpu').detach().numpy()), word_size=8, frac_size=6)
        # print('Sw activity after sorting', count_switching(new_fc3_weight_reshaped, word_size=8, frac_size=6))

        # new_indx5 = torch.tensor([2, 6, 7, 0, 4, 1, 3, 5, 9, 8])
        new_fc3_weight = new_fc3_weight#[new_indx5, :]
        # new_fc3_weight = new_fc3_weight.reshape(-1)
        new_fc3_bias = model.fc_3.bias#[new_indx5]

        #Layer 1
        # x.unsqueeze(dim=0)
        model.conv1.weight.data = new_conv1_weight
        model.conv1.bias.data   = new_conv1_bias

        #Layer 2
        model.conv2.weight.data = new_conv2_weight
        model.conv2.bias.data   = new_conv2_bias

        #Layer 3
        model.fc_1.weight.data = new_fc1_weight
        # model.fc_1.bias.data = new_fc1_bias

        #Layer 4
        model.fc_2.weight.data = new_fc2_weight
        model.fc_2.bias.data = new_fc2_bias

        #Layer 5
        model.fc_3.weight.data = new_fc3_weight
        model.fc_3.bias.data = new_fc3_bias
        # Verification of the accuracy
        # df.to_csv('./FORSA_' + str(args.bit) + '.csv', index=False, header=True)
        # df.to_csv('./FORSA_' + str(args.bit) + '_' + str(args.frac_size) + '.csv', index=False, header=True)
        df.to_csv('./FORSA_sw_activities.csv', index=False, header=True)
        test_loss, test_acc = evaluate(model, test_iterator, criterion, device)
        
        print(f'Test Loss after sorting: {test_loss:.3f} | Test Acc: {test_acc*100:.2f}%')

# Model defintion
class LeNet(nn.Module):
    def __init__(self, output_dim):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=1,
                               out_channels=32,
                               kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=32,
                               out_channels=32,
                               kernel_size=5)

        self.fc_1 = nn.Linear(32 * 4 * 4, 120)
        self.fc_2 = nn.Linear(120, 84)
        self.fc_3 = nn.Linear(84, output_dim)

    def forward(self, x):
        x_image = x
        x_image = x_image.to('cpu').detach().numpy()
#         save('x_image.npy', x_image)
#         scipy.io.savemat('x_image.mat', {'x_image': x_image})

        x = self.conv1(x)
        x_act = x
        x_act = x_act.to('cpu').detach().numpy()
#         save('x_act.npy', x_act)
#         scipy.io.savemat('x_act.mat', {'x_act': x_act})
        
        x = F.max_pool2d(x, kernel_size=2)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.max_pool2d(x, kernel_size=2)
        x = F.relu(x)
#         print('before flattening', x.shape)
        x = x.view(x.shape[0], -1)
        h = x
        
#         print('after flattening', x.shape)
        x = self.fc_1(x)
        x = F.relu(x)
        x = self.fc_2(x)
        x = F.relu(x)
        x = self.fc_3(x)
        return x, h
    

def count_parameters(model):
    return np.sum(p.numel() for p in model.parameters() if p.requires_grad)

def calculate_accuracy(y_pred, y):
    top_pred = y_pred.argmax(1, keepdim=True)
    correct = top_pred.eq(y.view_as(top_pred)).sum()
    acc = correct.float() / y.shape[0]
    return acc

def train(model, iterator, optimizer, criterion, device):

    epoch_loss = 0
    epoch_acc = 0

    model.train()

    for (x, y) in tqdm(iterator, desc="Training", leave=False):

        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        y_pred, _ = model(x)

        loss = criterion(y_pred, y)

        acc = calculate_accuracy(y_pred, y)

        loss.backward()

        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def evaluate(model, iterator, criterion, device):

    epoch_loss = 0
    epoch_acc = 0

    model.eval()

    with torch.no_grad():

        for (x, y) in tqdm(iterator, desc="Evaluating", leave=False):

            x = x.to(device)
            y = y.to(device)

            y_pred, idx = model(x)
#             print(idx)

            loss = criterion(y_pred, y)

            acc = calculate_accuracy(y_pred, y)

            epoch_loss += loss.item()
            epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

if __name__=='__main__':
    main()