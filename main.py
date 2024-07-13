import torch
from torch import nn, optim
from loader import UCSD_train_loader, UCSD_test_loader
from model import my_stae
from torch.utils.data import DataLoader, Dataset
import numpy as np
from os.path import exists
from os import makedirs
from plot import plot_loss
from tensorboardX import SummaryWriter


pretrain_model_path = './pretrained_model/'
log_path = './log/'

if not exists(pretrain_model_path):
    makedirs(pretrain_model_path)

if not exists(log_path):
    makedirs(log_path)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
writer = SummaryWriter(log_path)
train_data = UCSD_train_loader(1123)
loader_train_data = DataLoader(dataset=train_data, batch_size=4, shuffle=True, num_workers=0)

epochs = 20
learning_rate = 1e-5
model = my_stae(1).to(device)
optimizer = optim.Adam(model.parameters(), learning_rate)
criterion = nn.MSELoss()

loss_least = np.inf
model_best = model
loss_list = []

for epoch in range(epochs):
    loss_epoch = 0
    for batch_idx, x in enumerate(loader_train_data):
        x = x.to(device)
        x_hat = model(x)
        loss = criterion(x_hat, x)
        loss_epoch += loss.item()

        # back propagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    loss_list.append(loss_epoch)
    loss_avg = loss_epoch / (batch_idx+1)
    print(f'epoch: {epoch+1} / {epochs}; loss: {loss_avg}')
    writer.add_scalar('Loss/train', loss_avg, epoch)

    if loss_epoch < loss_least:
        loss_epoch = loss_least
        model_best = model

torch.save(model, pretrain_model_path + 'model_best' + '.pt')

plot_loss(epochs, loss_list)
writer.close()






