import datetime
import numpy as np
import os
import random
import torch
import torch.nn.functional as F
from copy import deepcopy
from torch.utils.data import DataLoader, TensorDataset
from utils.file import write

def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def make_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def split_data(x, y, trainval_ratios):
    n_train, n_val = [int(len(x) * trainval_ratio) for trainval_ratio in trainval_ratios]
    x_train, y_train = x[:n_train], y[:n_train]
    x_val, y_val = x[n_train:n_train + n_val], y[n_train:n_train + n_val]
    x_mean, x_sd = x_train.mean(0), x_train.std(0)
    x_train = (x_train - x_mean) / x_sd
    x_val = (x_val - x_mean) / x_sd
    if sum(trainval_ratios) == 1:
        return (x_train, y_train), (x_val, y_val)
    else:
        x_test, y_test = x[n_train + n_val:], y[n_train + n_val:]
        x_test = (x_test - x_mean) / x_sd
        return (x_train, y_train), (x_val, y_val), (x_test, y_test)

def make_dataloaders(data_train, data_val, data_test, batch_size):
    data_train = DataLoader(TensorDataset(*data_train), batch_size=batch_size, shuffle=True)
    data_val = DataLoader(TensorDataset(*data_val), batch_size=batch_size)
    data_test = DataLoader(TensorDataset(*data_test), batch_size=batch_size)
    return data_train, data_val, data_test

def train_epoch_vanilla(train_data, model, optimizer):
    device = make_device()
    model.train()
    loss_epoch = []
    for x_batch, y_batch in train_data:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        loss_batch = F.binary_cross_entropy(torch.sigmoid(model(x_batch)), y_batch)
        loss_batch.backward()
        loss_epoch.append(loss_batch.item())
        optimizer.step()
    return np.mean(loss_epoch)

def eval_epoch_vanilla(eval_data, model):
    device = make_device()
    model.eval()
    loss_epoch = []
    with torch.no_grad():
        for x_batch, y_batch in eval_data:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            loss_batch = F.binary_cross_entropy(torch.sigmoid(model(x_batch)), y_batch)
            loss_epoch.append(loss_batch.item())
    return np.mean(loss_epoch)

def posterior_kldiv(mu, logvar):
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)

def elbo(x, x_reconst, mu, logvar):
    loss_reconst = F.binary_cross_entropy_with_logits(x_reconst, x, reduction="none")
    loss_reconst = loss_reconst.sum(dim=1)
    return loss_reconst, posterior_kldiv(mu, logvar)

def train_epoch_vae(train_data, model, optimizer, epoch, reconst_mult, n_anneal_epochs):
    n_batches = len(train_data)
    device = make_device()
    model.train()
    loss_reconst_epoch, loss_kldiv_epoch, loss_epoch = [], [], []
    for batch_idx, (x_batch, y_batch) in enumerate(train_data):
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        x_reconst, mu, logvar = model(x_batch, y_batch)
        loss_reconst_batch, loss_kldiv_batch = elbo(x_batch, x_reconst, mu, logvar)
        loss_reconst_epoch.append(loss_reconst_batch.mean().item())
        loss_kldiv_epoch.append(loss_kldiv_batch.mean().item())
        anneal_mult = (batch_idx + epoch * n_batches) / (n_anneal_epochs * n_batches) if epoch < n_anneal_epochs else 1
        loss_batch = (reconst_mult * loss_reconst_batch + anneal_mult * loss_kldiv_batch).mean()
        loss_batch.backward()
        loss_reconst_epoch.append(loss_reconst_batch.mean().item())
        loss_kldiv_epoch.append(loss_kldiv_batch.mean().item())
        loss_epoch.append(loss_batch.item())
        optimizer.step()
    return np.mean(loss_reconst_epoch), np.mean(loss_kldiv_epoch), np.mean(loss_epoch)

def eval_epoch_vae(eval_data, model):
    device = make_device()
    model.eval()
    loss_reconst_epoch, loss_kldiv_epoch, loss_epoch = [], [], []
    with torch.no_grad():
        for x_batch, y_batch in eval_data:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            x_reconst, mu, logvar = model(x_batch, y_batch)
            loss_reconst_batch, loss_kldiv_batch = elbo(x_batch, x_reconst, mu, logvar)
            loss_batch = (loss_reconst_batch + loss_kldiv_batch).mean()
            loss_reconst_epoch.append(loss_reconst_batch.mean().item())
            loss_kldiv_epoch.append(loss_kldiv_batch.mean().item())
            loss_epoch.append(loss_batch.item())
    return np.mean(loss_reconst_epoch), np.mean(loss_kldiv_epoch), np.mean(loss_epoch)

def train_eval_loop(data_train, data_val, data_test, model, optimizer, train_f, eval_f, dpath, n_epochs):
    train_fpath = os.path.join(dpath, "train_summary.txt")
    val_fpath = os.path.join(dpath, "val_summary.txt")
    test_fpath = os.path.join(dpath, "test_summary.txt")
    min_val_loss = np.inf
    optimal_weights = deepcopy(model.load_state_dict)
    for epoch in range(n_epochs):
        train_loss_reconst, train_loss_kldiv, train_loss = train_f(data_train, model, optimizer, epoch)
        val_loss_reconst, val_loss_kldiv, val_loss = eval_f(data_val, model)
        train_loss_str = f"{train_loss_reconst:.6f}{train_loss_kldiv:.6f}, {train_loss:.6f}"
        val_loss_str = f"{val_loss_reconst:.6f}, {val_loss_kldiv:.6f}, {val_loss:.6f}"
        write(train_fpath, f"{timestamp()}, {epoch}, {train_loss_str}")
        write(val_fpath, f"{timestamp()}, {epoch}, {val_loss_str}")
        if val_loss < min_val_loss:
            optimal_weights = deepcopy(model.state_dict())
    torch.save(optimal_weights, os.path.join(dpath, "optimal_weights.pt"))
    model.load_state_dict(optimal_weights)
    test_loss_reconst, test_loss_kldiv, test_loss = eval_f(data_test, model)
    test_loss_str = f"{test_loss_reconst:.6f}, {test_loss_kldiv:.6f}, {test_loss:.6f}"
    write(test_fpath, f"{timestamp()}, {test_loss_str}")

def timestamp():
    return datetime.datetime.now().strftime('%H:%M:%S')