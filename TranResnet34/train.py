import sys
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from resnet_classifier import TranResnet34
from wsi.bin.image_producer import ImageDataset
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../')

torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

def train(epochs = 50, batch_size = 3, learning_rate = 0.0002, sample_interval = 1):

    time_now = time.time()
    R = TranResnet34()

    ## Change paths below to where you store your data
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, R.parameters()), lr=learning_rate, momentum=0.9, weight_decay=1e-5)

    for epoch in range(epochs):
        dataset_tumor = ImageDataset('../wsi/patches/tumor_train','../wsi/jsons/train',normalize=True)
        dataloader_tumor = DataLoader(dataset_tumor, batch_size=batch_size, num_workers=3)
        dataset_normal = ImageDataset('../wsi/patches/normal_train','../wsi/jsons/train',normalize=True)
        dataloader_normal = DataLoader(dataset_normal, batch_size=batch_size, num_workers=3)

        steps1 = len(dataloader_tumor)-1 #considering list.txt
        steps2 = len(dataloader_normal)-1
        steps = min(steps1,steps2)
        batch_size = dataloader_tumor.batch_size

        dataiter_tumor = iter(dataloader_tumor)
        dataiter_normal = iter(dataloader_normal)

        train_loss = 0
        correct = 0
        total = 0

        for step in range(steps):

            # image data and labels
            data_tumor, target_tumor, _ = next(dataiter_tumor)
            data_tumor = Variable(data_tumor)
            target_tumor = Variable(target_tumor)
            data_normal, target_normal, _ = next(dataiter_normal)
            data_normal = Variable(data_normal)
            target_normal = Variable(target_normal)

            idx_rand = Variable(torch.randperm(batch_size * 2))

            data = torch.cat([data_tumor, data_normal])[idx_rand]
            target = torch.cat([target_tumor, target_normal])[idx_rand]

            # Train Tranresnet34
            _  , result = R(data)
            loss = criterion(result, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = result.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

            time_spent = time.time() - time_now
            if (step+1) % 20 == 0:
                print("[Epoch %d/%d], [Step %d/%d], [loss: %.4f], [Accu:%3d%%], [RunTime:%.4f]"
                      % (epoch+1, epochs, step+1, steps, loss.item(), 100.*correct/total, time_spent))
        if (epoch+1) % sample_interval == 0:
            torch.save(R.state_dict(), './save_models/TranResnet34_params_%d.pkl'%(epoch+1))
    torch.save(R.state_dict(), './save_models/TranResnet34_params.pkl')
    print("FINAL:[Epoch %d/%d], [Step %d/%d], [loss: %.4f], [Accu:%3d%%], [RunTime:%.4f]"
          % (epoch+1, epochs, step+1, steps, loss.item(), 100.*correct/total, time_spent))
    print("Model saved.")


if __name__ == '__main__':
    train()
