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
from models import ADDAmodel
from utils import AverageMeter


def parse_args():
    parser = argparse.ArgumentParser()

    p = parser.add_argument_group("General")
    p.add_argument("--device", type=str, default='cpu')

    p = parser.add_argument_group("Model")
    p.add_argument("--load_path", type=str, default='./saved_models/ADDA')
    p.add_argument("--save_path", type=str, default='./saved_models/ADDA')

    p = parser.add_argument_group("Train")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--epoch", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--classifier_only", default=False, action='store_true')
    p.add_argument("--extractor_only", default=False, action='store_true')

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

    eeg_data = EEGDatasetWithDomain()
    models = [ADDAmodel() for i in range(15)]
    label_loss_func = nn.CrossEntropyLoss()
    domain_loss_func = nn.BCEWithLogitsLoss()
    total_y_true, total_y_pred = np.empty(0, dtype=int), np.empty(0, dtype=int)
    
    for idx, model in enumerate(models):
        # if idx == 0: continue
        model.to_device(device)
        x_train, y_train, x_test, y_test, d_source, d_target = eeg_data.leave_one_dataset(idx)
        train_data = DomainDataGenerator(x_train, y_train, d_source, x_test, d_target, seq_len=8)
        test_data = DomainDataGenerator(x_test, y_test, d_target, seq_len=8)
        train_loader = DataLoader(train_data, args.batch_size, shuffle=True)
        test_loader = DataLoader(test_data, args.batch_size)

        if not args.predict_only:
            print('[ADDA][Leave %d] ==============================' % idx)
            if not args.extractor_only:
                # Step 1: Train label classifier
                optimizer = torch.optim.Adam(
                    list(model.src_extractor.parameters()) + list(model.label_classifier.parameters()), 
                    lr=args.lr
                )
                total_steps = len(train_loader)
                print('[ADDA][Leave %d][Label classifier] Train begin!' % idx)
                model.src_extractor.train()
                model.label_classifier.train()
                model.domain_discriminator.requires_grad_(False)
                model.tar_extractor.requires_grad_(False)
                for ep in range(1, args.epoch + 1):
                    for i, batch in enumerate(train_loader):
                        batch_x, batch_y, batch_d, batch_xt, batch_dt = batch
                        batch_x = batch_x.to(device)
                        batch_y = batch_y.to(device)
                        src_features = model.src_extractor(batch_x)
                        label_logits = model.label_classifier(src_features)
                        loss = label_loss_func(label_logits, batch_y)

                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        print(
                            f'\r[ADDA][Leave {idx}][Label classifier][Epoch {ep}/{args.epoch}] > {i + 1}/{total_steps} Loss: {loss.item():.3f}',
                            end=''
                        )
                path = join(args.save_path, 'model_clf_leave%d.bin' % idx)
                if not os.path.exists(args.save_path):
                    os.makedirs(args.save_path)
                model.save_classifier(path)
                print()
            
                model.src_extractor.eval()
                model.label_classifier.eval()
                y_true, y_pred = np.empty(0, dtype=int), np.empty(0, dtype=int)
                for i, batch in enumerate(train_loader):
                    batch_x, batch_y, batch_d, batch_xt, batch_dt = batch
                    batch_x = batch_x.to(device)
                    batch_y = batch_y.data.numpy() - 1

                    src_features = model.src_extractor(batch_x)
                    label_logits = model.label_classifier(src_features)
                    label_logits = label_logits.data.cpu().numpy()
                    pred = np.argmax(label_logits, axis=1) - 1
                    y_true = np.append(y_true, batch_y)
                    y_pred = np.append(y_pred, pred)
                acc = accuracy_score(y_true, y_pred)
                print('[ADDA][Leave %d][Label classifier] Test on classifier: Acc %.4f' % (idx, acc))
            
            if not args.classifier_only:
                # Step 2: Train target extractor
                path = join(args.save_path, 'model_clf_leave%d.bin' % idx)
                model.load_classifier(path)
                tar_optimizer = torch.optim.Adam(model.tar_extractor.parameters(), lr=1e-8)
                dis_optimizer = torch.optim.Adam(model.domain_discriminator.parameters(), lr=1e-4)
                total_steps = len(train_loader)
                print('[ADDA][Leave %d][Target extractor] Train begin!' % idx)
                model.src_extractor.eval()
                model.src_extractor.requires_grad_(False)
                model.label_classifier.eval()
                model.label_classifier.requires_grad_(False)
                model.update_para()
                model.domain_discriminator.train()
                model.tar_extractor.train()
                for ep in range(1, args.epoch + 1):
                    for i, batch in enumerate(train_loader):
                        batch_x, batch_y, batch_d, batch_xt, batch_dt = batch
                        batch_x = batch_x.to(device)
                        batch_d = batch_d.to(device)
                        batch_xt = batch_xt.to(device)
                        batch_dt = batch_dt.to(device)

                        # Step 2.1: Train discriminator
                        model.domain_discriminator.requires_grad_(True)
                        model.tar_extractor.requires_grad_(False)
                        for _ in range(1):
                            src_features = model.src_extractor(batch_x).detach()
                            domain_logits_s = model.domain_discriminator(src_features)
                            loss_s = domain_loss_func(domain_logits_s, batch_d.unsqueeze(-1))
                            tar_features = model.tar_extractor(batch_xt).detach()
                            domain_logits_t = model.domain_discriminator(tar_features)
                            loss_t = domain_loss_func(domain_logits_t, batch_dt.unsqueeze(-1))
                            loss = loss_s + loss_t
                            dis_optimizer.zero_grad()
                            loss.backward()
                            dis_optimizer.step()

                        # Step 2.2: Train target extractor
                        model.domain_discriminator.requires_grad_(False)
                        model.tar_extractor.requires_grad_(True)
                        dis_optimizer.zero_grad()
                        for _ in range(3):
                            tar_features = model.tar_extractor(batch_xt)
                            domain_logits_t = model.domain_discriminator(tar_features)
                            loss_adv = domain_loss_func(domain_logits_t, batch_d.unsqueeze(-1)) # deceive
                            loss = loss_adv
                            tar_optimizer.zero_grad()
                            loss.backward()
                            tar_optimizer.step()

                        print(
                            f'\r[ADDA][Leave {idx}][Target extractor][Epoch {ep}/{args.epoch}] > {i + 1}/{total_steps} source Loss: {loss_s.item():.3f}, target Loss: {loss_t.item():.3f}, adv Loss: {loss_adv.item():.3f}',
                            end=''
                        )
                print()

            path = join(args.save_path, 'model_leave%d.bin' % idx)
            if not os.path.exists(args.save_path):
                os.makedirs(args.save_path)
            model.save_model(path)
        
        # Step 3: Use trained tar_extractor to classify labels
        print('[ADDA][Leave %d] Test begin!' % idx)
        path = join(args.save_path, 'model_leave%d.bin' % idx)
        model.load_model(path)
        model.tar_extractor.eval()
        model.label_classifier.eval()
        y_true, y_pred = np.empty(0, dtype=int), np.empty(0, dtype=int)
        for i, batch in enumerate(test_loader):
            batch_x, batch_y = batch
            batch_x = batch_x.to(device)
            batch_y = batch_y.data.numpy() - 1

            tar_features = model.tar_extractor(batch_x)
            label_logits = model.label_classifier(tar_features)
            label_logits = label_logits.data.cpu().numpy()
            pred = np.argmax(label_logits, axis=1) - 1
            y_true = np.append(y_true, batch_y)
            y_pred = np.append(y_pred, pred)

        acc = accuracy_score(y_true, y_pred)
        print('[ADDA][Leave %d] Test done! On dataset #%d: Acc %.4f' % (idx, idx, acc))
        total_y_true = np.append(total_y_true, y_true)
        total_y_pred = np.append(total_y_pred, y_pred)

    acc = accuracy_score(total_y_true, total_y_pred)
    print('[ADDA] All tests done! Total acc: %.4f' % acc)
    disp = ConfusionMatrixDisplay.from_predictions(
        total_y_true,
        total_y_pred,
        normalize='all',
        values_format='.3f'
    )
    plt.savefig('figures/ADDA.png')
    plt.show()
