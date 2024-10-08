import numpy as np
import pandas as pd
from torchvision.transforms import Normalize
import xarray as xr
import torch as pt
from torch import nn
import torch.utils.data as Data
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim.lr_scheduler import ExponentialLR
import csv

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import time
import torch
import random


seed = 42

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

start_time = time.time()


epochs = 100
batchsize = 128
inChan = 32

folder_paths = [
    '/root/autodl-tmp/input_row_net+/C1/sla/',
    '/root/autodl-tmp/input_row_net+/C1/sst/',
    '/root/autodl-tmp/input_row_net+/C1/net/',
    '/root/autodl-tmp/input_row_net+/C1/uwnd/',
    '/root/autodl-tmp/input_row_net+/C1/vwnd/'
    ]
folder_tensors = []
for folder_path in folder_paths:
    folder_name = os.path.basename(folder_path.rstrip('/'))
    csv_tensors = []

    for i in range(1, 7):
        csv_file = folder_path + f'{folder_name}_input_{i:02d}.csv'
        df = pd.read_csv(csv_file)
        df = df.iloc[:, 3:]
        matrix = df.values
        tensor = torch.tensor(matrix).unsqueeze(0)
        csv_tensors.append(tensor)

    folder_tensor = torch.cat(csv_tensors, dim=0)
    folder_tensors.append(folder_tensor)

X = torch.stack(folder_tensors, dim=1)
X = np.transpose(X, (2, 1, 0, 3))
print(X.shape)

X_train = X[:106272, :, :, :]
X_test = X[106272:, :, :, :]
num_train = np.size(X_train, 0)
print(X_train[0, 0, 0, 0])
print(X_train[-1, 0, 0, 0])
num_test = np.size(X_test, 0)
print(X_test[0, 0, 0, 0])
print(X_test[-1, 0, 0, 0])
print(X_train.shape)
print(X_test.shape)

Pnum = num_train
Pind = np.arange(0, num_train, 1)

X_train = np.reshape(X_train, [num_train, 5, 6, 23, 23])
X_test = np.reshape(X_test, [num_test, 5, 6, 23, 23])
print(X_train.shape)
print(X_test.shape)
Pnum = num_train

df = pd.read_csv('/root/autodl-tmp/output_row_net/output_soda_day8_C1.csv')
y = df.iloc[:, 3:].values
y_train = y[:106272, :]
y_test = y[106272:, :]
print(y_train[0, 0])
print(y_train[-1, 0])
print(y_test[0, 0])
print(y_test[-1, 0])
print(y.shape)
print(y_train.shape)
print(y_test.shape)

####################################################################################################

P0 = X_train
P2 = X_test
T0 = np.array(y_train)
T2 = np.array(y_test)
P = X

print(P0.shape)
print(P2.shape)

P0 = pt.tensor(P0, dtype=pt.float32)
T0 = pt.tensor(T0, dtype=pt.float32)
P2 = pt.tensor(P2, dtype=pt.float32).to(device)
T2 = pt.tensor(T2, dtype=pt.float32).to(device)
P = pt.tensor(P, dtype=pt.float32)

channels = 5
dataset0 = Data.TensorDataset(P0, T0)
loader = Data.DataLoader(
    dataset=dataset0,
    batch_size=batchsize,
    shuffle=True,
    num_workers=0
)


class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)
        self.fc1 = nn.Conv3d(in_channels, in_channels // reduction_ratio, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv3d(in_channels // reduction_ratio, in_channels, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu(self.fc1(self.max_pool(x))))
        out = self.sigmoid(avg_out + max_out)
        return out


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv3d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv1(out)
        out = self.sigmoid(out)
        return out


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=[1, 1, 1], padding=1):
        super(BasicBlock, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=(3, 3, 3), stride=(stride[0], stride[1], stride[2]),
                      padding=(padding, padding, padding), bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=(3, 3, 3), stride=(stride[0], stride[0], stride[0]),
                      padding=(padding, padding, padding), bias=False),
            nn.BatchNorm3d(out_channels)
        )
        self.ca = ChannelAttention(out_channels)
        self.sa = SpatialAttention()

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=(stride[0], stride[1], stride[2]),
                          bias=False),
                nn.BatchNorm3d(out_channels)
            )

    def forward(self, x):
        out = self.layer(x)
        out = self.ca(out) * out
        out = self.sa(out) * out
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class BasicBlock2(nn.Module):
    def __init__(self, in_channels, out_channels, stride=[1, 1, 1], padding=1):
        super(BasicBlock2, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=(3, 3, 3), stride=(stride[0], stride[1], stride[2]),
                      padding=(padding, padding, padding), bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=(3, 3, 3), stride=(stride[0], stride[0], stride[0]),
                      padding=(padding, padding, padding), bias=False),
            nn.BatchNorm3d(out_channels)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=(stride[0], stride[1], stride[2]),
                          bias=False),
                nn.BatchNorm3d(out_channels)
            )

    def forward(self, x):
        out = self.layer(x)
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet14(nn.Module):
    def __init__(self, BasicBlock, num_classes=1) -> None:
        super(ResNet14, self).__init__()
        self.in_channels = inChan
        self.conv1 = nn.Sequential(
            nn.Conv3d(channels, inChan, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False),
            nn.BatchNorm3d(inChan),
            nn.ReLU(inplace=True)
        )
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.conv2 = self._make_layer(BasicBlock, inChan, [[1, 1, 1], [1, 1, 1]])
        self.conv3 = self._make_layer(BasicBlock2, inChan * 2, [[1, 2, 2], [1, 1, 1]])
        self.conv4 = self._make_layer(BasicBlock, inChan * 4, [[1, 2, 2], [1, 1, 1]])
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(4 * inChan, 8)

    def _make_layer(self, block, out_channels, strides):
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.maxpool(out)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.avgpool(out)
        out = out.reshape(x.shape[0], -1)
        out = self.fc(out)
        return out


# network training

print("device: ", device)


model = ResNet14(BasicBlock)
model = model.to(device)
print(model)
lossfunction = pt.nn.SmoothL1Loss()
optimizer = pt.optim.Adam(model.parameters(), lr=0.001)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=False, threshold=0.00000001,threshold_mode='rel', min_lr=0, eps=1e-10)#自适应调整学习率


test_mse_list = []
losses = [];
losses1 = [];
minloss = 999;
cor2 = 0;
for epoch in range(epochs):
    print("Epoch {}/{}".format(epoch + 1, epochs))
    for step, (xbatch, ybatch) in enumerate(loader):
        xbatch = xbatch.to(device)
        ybatch = ybatch.to(device)
        model.train()
        out = model(xbatch)
        loss = lossfunction(out, ybatch)
        losses.append(loss.item())

        cor = np.corrcoef(ybatch.detach().cpu().numpy().transpose(1, 0), out.detach().cpu().numpy().transpose(1, 0))[
            0, 1]
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        del xbatch, ybatch, out
        torch.cuda.empty_cache()

    model.eval()
    with torch.no_grad():
        out2 = model(P2)
        loss2 = lossfunction(out2, T2)
        mse2 = np.sum(
            (T2.detach().cpu().numpy().transpose(1, 0) - out2.detach().cpu().numpy().transpose(1,
                                                                                               0)) ** 2) / num_test  # 均方误差 (MSE)
        test_mse_list.append(mse2)
        cor2 = np.corrcoef(np.round(T2.detach().cpu().numpy().transpose(1, 0), 6),
                           np.round(out2.detach().cpu().numpy().transpose(1, 0), 6))[
            0, 1]
        correlations = []
        mse_list = []
        #
        for i in range(T2.detach().cpu().numpy().transpose(1, 0).shape[0]):
            correlation = \
                np.corrcoef(out2.detach().cpu().numpy().transpose(1, 0)[i],
                            T2.detach().cpu().numpy().transpose(1, 0)[i])[
                    0, 1]
            correlations.append(correlation)
        for i in range(T2.detach().cpu().numpy().transpose(1, 0).shape[0]):
            mses = np.sum((T2.detach().cpu().numpy().transpose(1, 0)[i] - out2.detach().cpu().numpy().transpose(1, 0)[
                i]) ** 2) / num_test
            mse_list.append(mses)
        print('Train:', loss, ' Cor:', cor)
        print('Test:', loss2, ' Cor:', cor2, '- MSE:', mse2)
        print(' Cor_9time:', correlations)
        print(' MSE_9time:', mse_list)
        del out2, mse2, correlations, mse_list
        torch.cuda.empty_cache()



pt.save(model.state_dict(), "/root/autodl-tmp/Resnet_model/model_resnet_C1_529_6time.pth")
model.load_state_dict(pt.load("/root/autodl-tmp/Resnet_model/model_resnet_C1_529_6time.pth"))

model.eval()
with torch.no_grad():
    df_out2 = pd.read_csv("/root/autodl-tmp/output_row_net/output_soda_day8_C1.csv")
    df_out2 = df_out2.iloc[106272:, :]
    df_out2 = df_out2.iloc[:, 3:]
    out2_values = df_out2.values.astype(np.float32)
    predictions = model(P2)
    correlations = []
    for i in range(out2_values.transpose(1, 0).shape[0]):
        correlation = \
        np.corrcoef(predictions.detach().cpu().numpy().transpose(1, 0)[i], out2_values.transpose(1, 0)[i])[0, 1]
        correlations.append(correlation)
    print(correlations)


    for i, correlation in enumerate(correlations):
        print("Column {} and Column {}: Correlation = {}".format(i + 1, i + 1, correlation))

    df = pd.DataFrame(np.round(predictions.detach().cpu().numpy(), 6))

    df.to_csv(r"/root/autodl-tmp/Resnet_prediction/prediction_model_resnet_C1_529_6time.csv",
              index=False)
    
    end_time = time.time()
    execution_time = end_time - start_time

    hours = int(execution_time // 3600)
    minutes = int((execution_time % 3600) // 60)

    print("Execution time: {} hours, {} minutes".format(hours, minutes))
    exit()

