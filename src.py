import numpy
import pandas as pd
import torch
import torch.nn as nn

train_data = pd.read_csv('data/train.csv')
test_data = pd.read_csv('data/test.csv')
all_features = pd.concat((train_data.iloc[:, 1:-1], test_data.iloc[:, 1:]))
numeric_features = all_features.dtypes[all_features.dtypes != 'object'].index

all_features[numeric_features] = all_features[numeric_features].apply(lambda x: (x - x.mean()) / x.std())
all_features[numeric_features] = all_features[numeric_features].fillna(0)
all_features = pd.get_dummies(all_features, dummy_na=True)

n_train = len(train_data)

train_features = torch.tensor(all_features[:n_train].values, dtype=torch.float32)
test_features = torch.tensor(all_features[n_train:].values, dtype=torch.float32)

train_labels = torch.tensor(train_data.iloc[:, -1].values, dtype=torch.float32)

loss = nn.MSELoss()

in_features = train_features.shape[1]
out_features = 1

def get_net():
  net = nn.Sequential(nn.Linear(in_features, out_features))
  return net


def log_rmse(net, features, labels):
  clipped_preds = torch.clamp(net(features), 1, float('inf'))
  #     print(clipped_preds, labels)
  rmse = torch.sqrt(loss(torch.log(clipped_preds), torch.log(labels)))
  return rmse.item()


def load_array(data_array, batch_size):
  train_dataset = torch.utils.data.TensorDataset(*(data_array))
  train_dataloader = torch.utils.data.DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
  return train_dataloader


def train(net, train_features, train_labels, test_features, test_labels, num_epochs, learning_rate, weight_decay,
          batch_size):

  train_ls, test_ls = [], []
  train_iter = load_array((train_features, train_labels), batch_size)
  optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=weight_decay)
  print(f"[trace] train func: num_epochs = {num_epochs}")
  for epoch in range(num_epochs):
    if (epoch % 100 == 0):
      print(f"[trace] current progress {epoch}/{num_epochs}")
    for X, y in train_iter:
      y_hat = net(X)
      #             print(y_hat.shape, y.shape)
      l = loss(y_hat, y.unsqueeze(1))
      optimizer.zero_grad()
      l.backward()
      optimizer.step()
    train_ls.append(log_rmse(net, train_features, train_labels))
    if (test_labels is not None):
      test_ls.append(log_rmse(net, test_features, test_labels))
  return train_ls, test_ls


net = get_net()
test_labels = None
num_epochs = 10000
learning_rate = 0.003
weight_decay = 0
batch_size = 64

def get_k_fold_data(K, i, X, y):
  X_train, X_valid, y_train, y_valid = None, None, None, None
  len_of_slice = len(X) // K

  for j in range(K):
    index = slice(j * len_of_slice, (j + 1) * len_of_slice)
    if j == i:
      X_valid, y_valid = X[index, :], y[index]
    elif X_train is None:
      X_train, y_train = X[index, :], y[index]
    else:
      X_train = torch.cat([X_train, X[index, :]], 0)
      y_train = torch.cat([y_train, y[index]], 0)

  return X_train.cuda(), y_train.cuda(), X_valid.cuda(), y_valid.cuda()


for X, y in load_array((train_features, train_labels), batch_size):
  print(get_k_fold_data(10, 2, X, y))
  break

import matplotlib.pyplot as plt


def k_fold(k, X_train, y_train, num_epochs, learning_rate, weight_decay, batch_size):
  train_l_sum, valid_l_sum = 0, 0
  for i in range(k):
    data = get_k_fold_data(k, i, X_train, y_train)

    net = get_net().cuda()
    train_ls, valid_ls = train(net, *data, num_epochs, learning_rate, weight_decay, batch_size)

    train_l_sum += train_ls[-1]
    valid_l_sum += valid_ls[-1]

    if i == 0:
      plt.plot(list(range(1, num_epochs + 1)), train_ls)
      plt.plot(list(range(1, num_epochs + 1)), valid_ls)
      plt.show()

    print(f'fold {i + 1}, train log rmse {float(train_ls[-1]):f}, '
          f'valid log rmse {float(valid_ls[-1]):f}')
  return train_l_sum / k, valid_l_sum / k

k, num_epochs, lr, weight_decay, batch_size = 10, 2000, 5, 0, 64

train_l, valid_l = k_fold(k, train_features, train_labels, num_epochs, learning_rate, weight_decay, batch_size)
print(f'{k}-fold validation: avg train log rmse: {float(train_l):f}, '
      f'avg valid log rmse: {float(valid_l):f}')


def train_and_pred(train_features, test_feature, train_labels, test_data,
                   num_epochs, lr, weight_decay, batch_size):
  net = get_net()
  train_ls, _ = train(net, train_features, train_labels, None, None, num_epochs, lr, weight_decay, batch_size)

  plt.plot(range(1, num_epochs + 1), train_ls)
  plt.show()
  print(f"train log rmse : {float(train_ls[-1])}")

  preds = net(test_features).detach().numpy()
  print(preds, preds.reshape(1, -1))
  test_data['SalePrice'] = pd.Series(preds.reshape(1, -1)[0])
  submission = pd.concat([test_data['Id'], test_data['SalePrice']], axis=1)
  submission.to_csv('submission.csv', index=False)

import numpy as np
train_and_pred(train_features, test_features, train_labels, test_data,
               num_epochs, lr, weight_decay, batch_size)

