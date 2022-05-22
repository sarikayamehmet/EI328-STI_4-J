import argparse
import os
import random
from os.path import join

import numpy as np
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score
from torch.utils.data import DataLoader

from dataloader import DomainDataGenerator, EEGDatasetWithDomain
from models import DANN
from utils import AverageMeter


def parse_args():
    parser = argparse.ArgumentParser()

    p = parser.add_argument_group("General")
    p.add_argument("--device", type=str, default='cpu')

    p = parser.add_argument_group("Model")
    p.add_argument("--load_path", type=str, default='./saved_models/DANN')
    p.add_argument("--save_path", type=str, default='./saved_models/DANN')

    p = parser.add_argument_group("Train")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--epoch", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)

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


if __name__ == '__main__':
    args = parse_args()
    if args.device == 'gpu':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")
    seed_everything(args.seed)
    train_cnt = AverageMeter()

    eeg_data = EEGDatasetWithDomain()
    models = [DANN() for i in range(15)]
    task_loss_func = nn.CrossEntropyLoss()
    domain_loss_func = nn.BCEWithLogitsLoss()
    total_y_true, total_y_pred = np.empty(0, dtype=int), np.empty(0, dtype=int)
    for idx, model in enumerate(models):
        model.to(device)
        x_train, y_train, x_test, y_test, d_source, d_target = eeg_data.leave_one_dataset(idx)
        train_data = DomainDataGenerator(x_train, y_train, d_source, x_test, d_target, seq_len=8)
        test_data = DomainDataGenerator(x_test, y_test, d_target, seq_len=8)
        train_loader = DataLoader(train_data, args.batch_size, shuffle=True)
        test_loader = DataLoader(test_data, args.batch_size)

        if not args.predict_only:
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
            total_steps = len(train_loader)
            print('[DANN][Leave %d] Train begin!' % idx)
            model.train()
            for ep in range(1, args.epoch + 1):
                train_cnt.reset()

                for i, batch in enumerate(train_loader):
                    train_cnt.update(0, 1)
                    p = float(train_cnt.count + ep * len(train_loader)) / (args.epoch * len(train_loader))
                    lambda_ = 2. / (1. + np.exp(-10 * p)) - 1
                    model.set_lambda(lambda_)
                    params = batch
                    # batch_x = batch_x.to(device)
                    # batch_y = batch_y.to(device)
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
                    loss = task_loss + (domain_loss + domain_loss_t) * 0.5
                    
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    print(
                        f'\r[DANN][Leave {idx}][Epoch {ep}/{args.epoch}] > {i + 1}/{total_steps} task Loss: {task_loss.item():.3f}, domain Loss: {domain_loss.item():.3f}, target domain Loss: {domain_loss_t.item():.3f}',
                        end='')

            path = join(args.save_path, 'model_leave%d.bin' % idx)
            if not os.path.exists(args.save_path):
                os.makedirs(args.save_path)
            torch.save(model.state_dict(), path)
            print()

        print('\n[DANN][Leave %d] Test begin!' % idx)
        path = join(args.save_path, 'model_leave%d.bin' % idx)
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
        print('[DANN][Leave %d] Test done! On dataset #%d: Acc %.4f' % (idx, idx, acc))
        total_y_true = np.append(total_y_true, y_true)
        total_y_pred = np.append(total_y_pred, y_pred)

    acc = accuracy_score(total_y_true, total_y_pred)
    print('[DANN] All tests done! Total acc: %.4f' % acc)
    disp = ConfusionMatrixDisplay.from_predictions(
        total_y_true,
        total_y_pred,
        normalize='all',
        values_format='.3f'
    )
    plt.savefig('figures/DANN.png')
    plt.show()
