import sys
import os
import logging

import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from resnet_classifier import TranResnet34
from wsi.bin.image_producer import ImageDataset
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../')

torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

def train_epoch(summary,summary_writer,model,loss_fn,optimizer,dataloader_tumor,dataloader_normal,sample_interval=40):
    model.train()
    steps1 = len(dataloader_tumor)
    steps2 = len(dataloader_normal)
    steps = min(steps1,steps2)-1
    batch_size = dataloader_tumor.batch_size

    dataiter_tumor = iter(dataloader_tumor)
    dataiter_normal = iter(dataloader_normal)

    time_now = time.time()

    for step in range(steps):
        # image data and labels
        data_tumor, target_tumor, _ = next(dataiter_tumor)
        data_tumor = Variable(data_tumor.cuda())
        target_tumor = Variable(target_tumor.cuda())
        data_normal, target_normal, _ = next(dataiter_normal)
        data_normal = Variable(data_normal.cuda())
        target_normal = Variable(target_normal.cuda())

        idx_rand = Variable(torch.randperm(batch_size * 2).cuda())

        data = torch.cat([data_tumor, data_normal])[idx_rand]
        target = torch.cat([target_tumor, target_normal])[idx_rand]

        # Train model
        _ , result = model(data)
        loss = loss_fn(result, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_data = loss.item()
        _, predicted = result.max(1)
        acc_data = 100.*predicted.eq(target).type(torch.cuda.FloatTensor).sum().item() / target.size(0)

        time_spent = time.time() - time_now

        summary['step'] += 1
        if summary['step'] % sample_interval == 0:
            summary_writer.add_scalar('train/loss', loss_data, summary['step'])
            summary_writer.add_scalar('train/acc', acc_data, summary['step'])
            logging.info(
                '{}, Epoch : {}, Step : {}, Training Loss : {:.5f}, '
                'Training Acc : {:.3f}, Run Time : {:.4f}'
                .format(
                    time.strftime("%Y-%m-%d %H:%M:%S"), summary['epoch'] + 1,
                    summary['step'], loss_data, acc_data, time_spent))


    summary['epoch'] += 1

    return summary


def valid_epoch(summary, model, loss_fn, optimizer, dataloader_tumor, dataloader_normal):
    model.eval()
    steps1 = len(dataloader_tumor)
    steps2 = len(dataloader_normal)
    steps = min(steps1, steps2)-1 #considering list.txt
    batch_size = dataloader_tumor.batch_size

    dataiter_tumor = iter(dataloader_tumor)
    dataiter_normal = iter(dataloader_normal)

    loss_sum = 0
    acc_sum = 0
    for step in range(steps):
        # image data and labels
        data_tumor, target_tumor, _ = next(dataiter_tumor)
        data_tumor = Variable(data_tumor.cuda())
        target_tumor = Variable(target_tumor.cuda())
        data_normal, target_normal, _ = next(dataiter_normal)
        data_normal = Variable(data_normal.cuda())
        target_normal = Variable(target_normal.cuda())

        idx_rand = Variable(torch.randperm(batch_size * 2))

        data = torch.cat([data_tumor, data_normal])[idx_rand]
        target = torch.cat([target_tumor, target_normal])[idx_rand]

        # Validation result

        _, result = model(data)
        loss = loss_fn(result, target)

        loss_data = loss.item()
        _, predicted = result.max(1)
        acc_data = 100.*predicted.eq(target).type(torch.cuda.FloatTensor).sum().item() / target.size(0)

        loss_sum += loss_data
        acc_sum += acc_data

    summary['loss'] = loss_sum / steps
    summary['acc'] = acc_sum / steps

    return summary

def run(R, epochs = 50, batch_size = 4, learning_rate = 0.0002, sample_interval = 40, save_path = './save_models/'):

    loss_fn = nn.CrossEntropyLoss().cuda()
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, R.parameters()), lr=learning_rate, momentum=0.9, weight_decay=1e-5)

    dataset_tumor_train = ImageDataset('../wsi/patches/tumor_train',
                                       '../wsi/jsons/train',
                                       normalize=True)
    dataloader_tumor_train = DataLoader(dataset_tumor_train, batch_size=batch_size, num_workers=3)

    dataset_normal_train = ImageDataset('../wsi/patches/normal_train',
                                        '../wsi/jsons/train',
                                        normalize=True)
    dataloader_normal_train = DataLoader(dataset_normal_train, batch_size=batch_size, num_workers=3)

    dataset_tumor_valid = ImageDataset('../wsi/patches/tumor_valid',
                                       '../wsi/jsons/valid',
                                       normalize=True)
    dataloader_tumor_valid = DataLoader(dataset_tumor_valid, batch_size=batch_size, num_workers=3)

    dataset_normal_valid = ImageDataset('../wsi/patches/normal_valid',
                                        '../wsi/jsons/valid',
                                        normalize=True)
    dataloader_normal_valid = DataLoader(dataset_normal_valid, batch_size=batch_size, num_workers=3)

    summary_train = {'epoch': 0, 'step': 0}
    summary_valid = {'loss': float('inf'), 'acc': 0}
    summary_writer = SummaryWriter(save_path)
    loss_valid_best = float('inf')

    for epoch in range(epochs):
        summary_train = train_epoch(summary_train, summary_writer, R,
                                    loss_fn, optimizer,
                                    dataloader_tumor_train,
                                    dataloader_normal_train,
                                    sample_interval=sample_interval)
        torch.save({'epoch': summary_train['epoch'],
                    'step': summary_train['step'],
                    'state_dict': R.state_dict()},
                   os.path.join(save_path, 'train.ckpt'))

        if (epoch+1) % 3 == 0:
            torch.save(R.state_dict(), './save_models/TranResnet34_params_%d.pkl' % (epoch + 1))

        time_now = time.time()
        summary_valid = valid_epoch(summary_valid, R, loss_fn, optimizer,
                                    dataloader_tumor_valid, dataloader_normal_valid)
        time_spent = time.time() - time_now

        logging.info(
            '{}, Epoch : {}, Step : {}, Validation Loss : {:.5f}, '
            'Validation Acc : {:.3f}, Run Time : {:.4f}'
            .format(
                time.strftime("%Y-%m-%d %H:%M:%S"), summary_train['epoch'],
                summary_train['step'], summary_valid['loss'],
                summary_valid['acc'], time_spent))

        summary_writer.add_scalar(
            'valid/loss', summary_valid['loss'], summary_train['step'])
        summary_writer.add_scalar(
            'valid/acc', summary_valid['acc'], summary_train['step'])

        if summary_valid['loss'] < loss_valid_best:
            print("Update the best with %d epoch"%(epoch+1))
            loss_valid_best = summary_valid['loss']
            torch.save({'epoch': summary_train['epoch'],
                        'step': summary_train['step'],
                        'state_dict': R.state_dict()},
                       os.path.join(save_path, 'best.ckpt'))

    summary_writer.close()

def main():
    logging.basicConfig(level=logging.INFO)
    R = TranResnet34()
    # To start from formal trained model
    # R.load_state_dict(torch.load('./save_models/TranResnet34_params_formal.pkl'))
    #R.freeze(6)
    R.cuda()
    run(R)

if __name__ == '__main__':
    main()