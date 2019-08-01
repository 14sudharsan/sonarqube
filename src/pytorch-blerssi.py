from __future__ import print_function

import argparse
import os

from tensorboardX import SummaryWriter
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from builtins import range
from torch.autograd import Variable
torch.manual_seed(1234)
import pandas as pd
import numpy as np
import shutil
import os
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score

WORLD_SIZE = int(os.environ.get('WORLD_SIZE', 1))
criterion = nn.CrossEntropyLoss()
class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(13, 65)
        self.relu1=nn.ReLU()
        self.fc2 = nn.Linear(65, 65)
        self.relu2=nn.ReLU()
        self.fc3 = nn.Linear(65, 105)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.fc3(out)
        return out

def train(args, model,  optimizer, epoch, writer):
    BLE_RSSI = pd.read_csv('iBeacon_RSSI_Labeled.csv')
    COLUMNS = list(BLE_RSSI.columns)
    FEATURES = COLUMNS[2:]
    LABEL = [COLUMNS[0]]
    df_full = pd.read_csv('iBeacon_RSSI_Labeled.csv') 

  
    df_full = df_full.drop(['date'],axis = 1)
    df_full[FEATURES] = (df_full[FEATURES]-df_full[FEATURES].mean())/df_full[FEATURES].std()

 
    dict = {'O02': 0,'P01': 1,'P02': 2,'R01': 3,'R02': 4,'S01': 5,'S02': 6,'T01': 7,'U02': 8,'U01': 9,'J03': 10,'K03': 11,'L03': 12,'M03': 13,'N03': 14,'O03': 15,'P03': 16,'Q03': 17,'R03': 18,'S03': 19,'T03': 20,'U03': 21,'U04': 22,'T04': 23,'S04': 24,'R04': 25,'Q04': 26,'P04': 27,'O04': 28,'N04': 29,'M04': 30,'L04': 31,'K04': 32,'J04': 33,'I04': 34,'I05': 35,'J05': 36,'K05': 37,'L05': 38,'M05': 39,'N05': 40,'O05': 41,'P05': 42,'Q05': 43,'R05': 44,'S05': 45,'T05': 46,'U05': 47,'S06': 48,'R06': 49,'Q06': 50,'P06': 51,'O06': 52,'N06': 53,'M06': 54,'L06': 55,'K06': 56,'J06': 57,'I06': 58,'F08': 59,'J02': 60,'J07': 61,'I07': 62,'I10': 63,'J10': 64,'D15': 65,'E15': 66,'G15': 67,'J15': 68,'L15': 69,'R15': 70,'T15': 71,'W15': 72,'I08': 73,'I03': 74,'J08': 75,'I01': 76,'I02': 77,'J01': 78,'K01': 79,'K02': 80,'L01': 81,'L02': 82,'M01': 83,'M02': 84,'N01': 85,'N02': 86,'O01': 87,'I09': 88,'D14': 89,'D13': 90,'K07': 91,'K08': 92,'N15': 93,'P15': 94,'I15': 95,'S15': 96,'U15': 97,'V15': 98,'S07': 99,'S08': 100,'L09': 101,'L08': 102,'Q02': 103,'Q01': 104}
    df_full['location'] = df_full['location'].map(dict)
    df_train=df_full.sample(frac=0.8,random_state=200)
    df_test=df_full.drop(df_train.index)

    location_counts = BLE_RSSI.location.value_counts()
    train_X, test_X, train_y, test_y = train_test_split(df_full[df_full.columns[1:14]].values,
                                                    df_full.location.values, test_size=0.20)
    train_X = Variable(torch.Tensor(train_X).float())
    test_X = Variable(torch.Tensor(test_X).float())
    train_y = Variable(torch.Tensor(train_y).long())
    test_y = Variable(torch.Tensor(test_y).long())
    model.train()
    for epoch in range(3000):
        out = model(train_X)
        loss = criterion(out, train_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch % 100 == 0:
            print('number of epoch', epoch, 'loss', loss.data)

    predict_out = model(test_X)
    _, predict_y = torch.max(predict_out, 1)

    print('\naccuracy={:.4f}\n'.format(accuracy_score(test_y.data, predict_y.data)))
    writer.add_scalar('accuracy', accuracy_score(test_y.data, predict_y.data), epoch)


def should_distribute():
    return dist.is_available() and WORLD_SIZE > 1


def is_distributed():
    return dist.is_available() and dist.is_initialized()

def main():

    parser = argparse.ArgumentParser(description='PyTorch BLERSSI Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=1, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--beta1', type=float, default=0.1,
                        help='Beta1 value')
    parser.add_argument('--beta2', type=float, default=0.5,
                        help='Beta2 value')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=True,
                        help='For Saving the current Model')
    parser.add_argument('--dir', default='logs', metavar='L',
                        help='directory where summary logs are stored')
    if dist.is_available():
        parser.add_argument('--backend', type=str, help='Distributed backend',
                            choices=[dist.Backend.GLOO, dist.Backend.NCCL, dist.Backend.MPI],
                            default=dist.Backend.GLOO)
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    if use_cuda:
        print('Using CUDA')

    writer = SummaryWriter(args.dir)

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    if should_distribute():
        print('Using distributed PyTorch with {} backend'.format(args.backend))
        dist.init_process_group(backend=args.backend)

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    model = Net().to(device)

    if is_distributed():
        Distributor = nn.parallel.DistributedDataParallel if use_cuda \
            else nn.parallel.DistributedDataParallelCPU
        model = Distributor(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
    for epoch in range(1, args.epochs + 1):
        train(args, model, optimizer, epoch, writer)

    if (args.save_model):
        torch.save(model.state_dict(),"/var/blerssi_cnn.pt")
        print("Model Saved")


if __name__ == '__main__':
    main()

