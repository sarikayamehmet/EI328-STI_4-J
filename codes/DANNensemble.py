import argparse
import math
import os
import random
from os.path import join

import numpy as np
from scipy import stats
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score
from torch.utils.data import DataLoader
from utils import topk_element_in_ndarray

from dataloader import DomainDataGenerator, EEGDatasetWithDomain
from models import DANN
from utils import AverageMeter


def parse_args():
    parser = argparse.ArgumentParser()

    p = parser.add_argument_group("General")
    p.add_argument("--device", type=str, default='cpu')

    p = parser.add_argument_group("Model")
    p.add_argument("--load_path", type=str, default='./saved_models/DANNensemble')
    p.add_argument("--save_path", type=str, default='./saved_models/DANNensemble')

    p = parser.add_argument_group("Train")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--epoch", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--domain_loss_rate", type=float, default=0.5)
    p.add_argument("--reinforce_rate", type=float, default=0.3)

    p = parser.add_argument_group("Predict")
    p.add_argument("--predict_only", default=False, action='store_true')

    return parser.parse_args()


def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def calculate_losses_on_dataset(dataset: DomainDataGenerator, model: DANN, device) -> np.ndarray:
    """ 计算给定数据集上每个样本的loss, 返回array """
    model.eval()
    data_loader = DataLoader(dataset, batch_size=1, shuffle=True)
    losses = np.zeros(len(dataset))

    with torch.no_grad():
        for idx, batch in enumerate(data_loader):
            params = batch
            for param_idx, param in enumerate(params):
                params[param_idx] = param.to(device)
            batch_x, batch_y, batch_d, batch_xt, batch_dt = params

            task_predict, domain_predict = model(batch_x)
            task_loss = task_loss_func(task_predict, batch_y)
            domain_loss = domain_loss_func(domain_predict, batch_d.unsqueeze(-1))
            _, domain_predict_t = model(batch_xt)
            domain_loss_t = domain_loss_func(domain_predict_t, batch_dt.unsqueeze(-1))
            loss = task_loss + (domain_loss + domain_loss_t) * args.domain_loss_rate
            loss = loss.detach().item()

            losses[idx] = loss
    return losses

def data_reinforce(dataset: DomainDataGenerator, n_samples: int, rate: float, model: DANN, device):
    dataset.x, dataset.y, dataset.ds, dataset.xt, dataset.dt = \
        dataset.x[:n_samples], dataset.y[:n_samples], dataset.ds[:n_samples], dataset.xt[:n_samples], dataset.dt[:n_samples]
    losses = calculate_losses_on_dataset(dataset, model, device)
    sample_idx = topk_element_in_ndarray(losses, topk=int(rate * n_samples))[1][0]
    
    dataset.x = np.append(dataset.x, [dataset.x[i] for i in sample_idx], axis=0) 
    dataset.y = np.append(dataset.y, [dataset.y[i] for i in sample_idx], axis=0)
    dataset.ds = np.append(dataset.ds, [dataset.ds[i] for i in sample_idx], axis=0)
    dataset.xt = np.append(dataset.xt, [dataset.xt[i] for i in sample_idx], axis=0)
    dataset.dt = np.append(dataset.dt, [dataset.dt[i] for i in sample_idx], axis=0)
    loader = DataLoader(dataset, args.batch_size, shuffle=True)
    return loader


if __name__ == '__main__':
    args = parse_args()
    if args.device == 'gpu':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")
    seed_everything(args.seed)
    train_cnt = AverageMeter()

    eeg_data = EEGDatasetWithDomain()
    task_loss_func = nn.CrossEntropyLoss()
    domain_loss_func = nn.BCEWithLogitsLoss()
    total_y_true, total_y_pred = np.empty(0, dtype=int), np.empty(0, dtype=int)
    
    for target_idx in range(15):
        print('[Ensemble][Target %d] ==============================' % target_idx)
        x_target, y_target, d_target = eeg_data.choose_one_dataset(target_idx, is_src=False)
        test_data = DomainDataGenerator(x_target, y_target, d_target, seq_len=8)
        test_loader = DataLoader(test_data, args.batch_size)
        srcs = np.delete(np.arange(15), target_idx)
        this_y_pred = np.empty(0, dtype=int)
        
        for source_idx in srcs:
            model = DANN().to(device)
            x_source, y_source, d_source = eeg_data.choose_one_dataset(source_idx, is_src=True)
            train_data = DomainDataGenerator(x_source, y_source, d_source, x_target, d_target, seq_len=8)
            train_loader = DataLoader(train_data, args.batch_size, shuffle=True)

            if not args.predict_only:
                print('[Ensemble][Src %d -> Tar %d] Train begin!' % (source_idx, target_idx))
                optimizer = torch.optim.Adam(model.parameters(), lr=args.lr) 
                n_train_samples = len(train_data)

                for ep in range(1, args.epoch + 1):
                    model.train()
                    train_cnt.reset()

                    for i, batch in enumerate(train_loader):
                        total_steps = len(train_loader)
                        train_cnt.update(0, 1)
                        p = float(train_cnt.count + ep * len(train_loader)) / (args.epoch * len(train_loader))
                        lambda_ = 2. / (1. + np.exp(-10 * p)) - 1
                        model.set_lambda(lambda_)
                        params = batch

                        for param_idx, param in enumerate(params):
                            params[param_idx] = param.to(device)
                        batch_x, batch_y, batch_d, batch_xt, batch_dt = params
                        # Train model using source
                        task_predict, domain_predict = model(batch_x)
                        task_loss = task_loss_func(task_predict, batch_y)
                        domain_loss = domain_loss_func(domain_predict, batch_d.unsqueeze(-1))
                        # Train model using target
                        _, domain_predict_t = model(batch_xt)
                        domain_loss_t = domain_loss_func(domain_predict_t, batch_dt.unsqueeze(-1))
                        loss = task_loss + (domain_loss + domain_loss_t) * args.domain_loss_rate
                        
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        print(
                            f'\r[Ensemble][Src {source_idx} -> Tar {target_idx}][Epoch {ep}/{args.epoch}] > {i + 1}/{total_steps} task Loss: {task_loss.item():.3f}, domain Loss: {domain_loss.item():.3f}, target domain Loss: {domain_loss_t.item():.3f}',
                            end='')

                    if (args.reinforce_rate > 0) and (ep != args.epoch) and (ep % 5 == 0):
                        train_loader = data_reinforce(train_data, n_train_samples, args.reinforce_rate, model, device)

                path = join(args.save_path, 'model_%d_%d.bin' % (source_idx, target_idx))
                if not os.path.exists(args.save_path):
                    os.makedirs(args.save_path)
                torch.save(model.state_dict(), path)
                print()

            print('[Ensemble][Src %d -> Tar %d] Test begin!' % (source_idx, target_idx))
            path = join(args.save_path, 'model_%d_%d.bin' % (source_idx, target_idx))
            model.load_state_dict(torch.load(path))
            model.eval()
            y_true, y_pred = np.empty(0, dtype=int), np.empty(0, dtype=int)
            for i, batch in enumerate(test_loader):
                batch_x, batch_y = batch
                batch_x = batch_x.to(device)
                batch_y = batch_y.data.numpy() - 1

                task_logits, _ = model(batch_x)
                task_logits = task_logits.data.cpu().numpy()
                pred = np.argmax(task_logits, axis=1) - 1
                y_true = np.append(y_true, batch_y)
                y_pred = np.append(y_pred, pred)

            acc = accuracy_score(y_true, y_pred)
            print('[Ensemble][Src %d -> Tar %d] Test done! Acc %.4f' % (source_idx, target_idx, acc))
            this_y_pred = np.append(this_y_pred, y_pred)

        # Ensemble
        this_y_pred = np.reshape(this_y_pred, (14, -1))
        ensembled_pred = stats.mode(this_y_pred)[0][0]
        this_acc = accuracy_score(y_target, ensembled_pred)
        print('[Ensemble][Target %d] Ensemble done! On dataset %d: Acc %.4f' % (target_idx, target_idx, this_acc))
        total_y_true = np.append(total_y_true, y_target)
        total_y_pred = np.append(total_y_pred, ensembled_pred)  

    acc = accuracy_score(total_y_true, total_y_pred)
    print('[Ensemble] All tests done! Total acc: %.4f' % acc)
    disp = ConfusionMatrixDisplay.from_predictions(
        total_y_true,
        total_y_pred,
        normalize='all',
        values_format='.3f'
    )
    plt.savefig('figures/DANNensemble.png')
    plt.show()
