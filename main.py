import datetime
import argparse
import os
import random
import time
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from torch import nn
from torch.utils.data import DataLoader
from pathlib import Path
import torch
from tensorboardX import SummaryWriter
from models import misc
from models.RMT import RMT_S_pretrain, RMT_L_pretrain
from models.models_list import vit_s, swin_s
from models.overlock import OverLock_B_pretrain
from models.starnet import starnet_s4_pretrain
from dataset import FLDataset


def get_args_parser():
    parser = argparse.ArgumentParser('Classification', add_help=False)
    parser.add_argument('--lr', default=0.0001, type=float)
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--weight_decay', default=1e-5, type=float)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--lr_drop', default=80, type=int)
    parser.add_argument('--size', default=224, type=int)
    parser.add_argument('--output_dir', default='output/exp_overlockb_e4/', help='path where to save')
    parser.add_argument('--dataset_name', default='汇总数据-统一为6张', help='')
    parser.add_argument('--label_num', default=2, help='2', type=int)
    parser.add_argument('--device', default='cuda', type=str, help='cuda or cpu')
    parser.add_argument('--GPU_ids', type=str, default='0', help='Ids of GPUs')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--num_workers', default=0, type=int)
    return parser

def main(args):
    writer = SummaryWriter(log_dir=args.output_dir + '/summary')
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.allow_tf32 = True
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    device = torch.device(args.device)

    model = OverLock_B_pretrain(args.label_num)

    output_dir = Path(args.output_dir)
    model.to(device)
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)
    print('Building training dataset...')
    dataset_train = FLDataset(args.dataset_name, 'train')
    print('Number of training images: {}'.format(len(dataset_train)))
    print('Building validation dataset...')
    dataset_val = FLDataset(args.dataset_name, 'test')
    print('Number of validation images: {}'.format(len(dataset_val)))
    dataloader_train = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    dataloader_val = DataLoader(dataset_val, batch_size=1, shuffle=False, num_workers=args.num_workers)
    # -------------------------------------------------------
    print("Start training")
    start_time = time.time()
    best_acc = None
    print_freq = 50
    for epoch in range(0, args.epochs):
        print('-' * 40)
        print('Epoch: [{}] '.format(epoch + 1))
        print('Training...')
        # ------------------------------------------------------------
        # train
        # ------------------------------------------------------------
        model.train()
        train_start_time = time.time()
        step = 0
        pred_list = []
        label_list = []
        criterion_cls = nn.CrossEntropyLoss()
        for img, label, _ in dataloader_train:
            # ------------------------------------------------------------
            img = img.to(device)
            label = label.to(device)
            # ------------------------------------------------------------
            output = model(img)
            output_main, output_aux = output['main'], output['aux']
            # ------------------------------------------------------------
            loss_cls = criterion_cls(output_main, label)
            loss_aux = criterion_cls(output_aux, label)
            loss = loss_cls + 0.4 * loss_aux
            # ------------------------------------------------------------
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # ------------------------------------------------------------
            _, pred = torch.max(output_main, 1)
            pred_list.append(pred.cpu().detach().numpy().tolist())
            label_list.append(label.cpu().detach().numpy().tolist())
            if step % print_freq == 0:
                print('    lr: {:.6f}'.format(optimizer.param_groups[0]["lr"]))
                print('    loss_cls: {:.4f}'. format(loss_cls.item()))
            step = step + 1
        # ------------------------------------------------------------
        train_total_time = time.time() - train_start_time
        total_time_str = str(datetime.timedelta(seconds=int(train_total_time)))
        pred_list = [b for a in pred_list for b in a]
        label_list = [b for a in label_list for b in a]
        cm = confusion_matrix(pred_list, label_list)
        print(cm)
        print('Training time: {} ({:.4f} r / it )'.format(total_time_str, train_total_time / len(dataloader_train)))
        writer.add_scalar('loss', loss.item(), epoch)
        lr_scheduler.step()
        # ------------------------------------------------------------
        # evaluate
        # ------------------------------------------------------------
        if (epoch + 1) % 1 == 0:
            model.eval()
            pred_list = []
            label_list = []
            print('Val...')
            val_start_time = time.time()
            for img, label, _ in dataloader_val:
                # ------------------------------------------------------------
                img = img.to(device)
                label = label.to(device)
                # ------------------------------------------------------------
                output = model(img)
                loss = criterion_cls(output, label)
                # ------------------------------------------------------------
                _, pred = torch.max(output, 1)
                pred_list.append(pred.cpu().detach().numpy().tolist())
                label_list.append(label.cpu().detach().numpy().tolist())
            # ------------------------------------------------------------
            val_total_time = time.time() - val_start_time
            total_time_str = str(datetime.timedelta(seconds=int(val_total_time)))
            print('Val time: {} ({:.4f} r / it )'.format(total_time_str, val_total_time / len(dataloader_val)))
            pred_list = [b for a in pred_list for b in a]
            label_list = [b for a in label_list for b in a]
            acc_score = accuracy_score(pred_list, label_list)
            print('Val Acc: {:.4f}  Val loss: {:.4f}'.format(acc_score, loss.item()))
            cm = confusion_matrix(pred_list, label_list)
            print(cm)
            writer.add_scalar('val_loss', loss.item(), epoch)
            writer.add_scalar('val_acc', acc_score, epoch)
            # ------------------------------------------------------------
            # save checkpoint for high dice score
            # ------------------------------------------------------------
            if args.output_dir:
                checkpoint_paths = [output_dir / 'checkpoint.pth']
                if best_acc is None or acc_score > best_acc:
                    best_acc = acc_score
                    print("Update best model!")
                    checkpoint_paths.append(output_dir / 'best_checkpoint.pth')
                # You can change the threshold
                if acc_score > 0.75:
                    print("Update high dice score model!")
                    file_name = str(acc_score)[0:6] + '_' + str(epoch + 1) + '_checkpoint.pth'
                    checkpoint_paths.append(output_dir / file_name)
                # extra checkpoint before LR drop and every 100 epochs
                if (epoch + 1) % args.lr_drop == 0 or (epoch + 1) % 100 == 0:
                    checkpoint_paths.append(output_dir / f'checkpoint{epoch:04}.pth')
                for checkpoint_path in checkpoint_paths:
                    misc.save_on_master({
                        'model': model.state_dict(),
                    }, checkpoint_path)
        print()
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Classification training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(args.GPU_ids)
    main(args)