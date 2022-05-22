import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from dataloader import DataGenerator
from dataloader import EEGDataset
from models import LSTM_net
from torch.utils.data import DataLoader


eeg_data = EEGDataset()
model = LSTM_net()
device = torch.device('cpu')
loss_func = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
x_train, y_train, x_test, y_test = eeg_data.leave_one_dataset(0)
x_data = x_train[:3394]
y_data = y_train[:3394]
X_train, X_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=42)
train_data = DataGenerator(X_train, y_train, seq_len=8)
test_data = DataGenerator(X_test, y_test, seq_len=8)
train_loader = DataLoader(train_data, 32)
test_loader = DataLoader(test_data, 32)

for ep in range(1, 11):
    hx, cx = torch.zeros(32, 256), torch.zeros(32, 256)
    total_steps = len(train_loader)
    for i, batch in enumerate(train_loader):
        batch_x, batch_y = batch
        logits = model(batch_x)
        loss = loss_func(logits, batch_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f'\r[LSTM baseline][Epoch {ep}/{10}] > {i + 1}/{total_steps} Loss: {loss.item():.3f}', end='')

    y_true, y_pred = np.empty(0, dtype=int), np.empty(0, dtype=int)
    hx, cx = torch.zeros(32, 256), torch.zeros(32, 256)
    for i, batch in enumerate(test_loader):
        batch_x, batch_y = batch
        logits = model(batch_x)
        logits = logits.data.cpu().numpy()
        pred = np.argmax(logits, axis=1) - 1
        y_true = np.append(y_true, batch_y - 1)
        y_pred = np.append(y_pred, pred)

    acc = accuracy_score(y_true, y_pred)
    print('[LSTM baseline] Test done! Acc %.4f' % (acc))