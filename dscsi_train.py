import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch import optim
import os.path

from wsi.bin.image_producer import ImageDataset

import pytorch_ssim
import pytorch_dscsi
from unet_generator import UNet
from dcgan_discriminator import Discriminator
from resnet_classifier import TranResnet34

torch.manual_seed(0)
torch.cuda.manual_seed_all(0)


def freeze_model(model):
    model.eval()
    for params in model.parameters():
        params.requires_grad = False

def train(epochs=8, batch_size=2, learning_rate=0.001, sample_interval=1):
    time_now = time.time()

    G = UNet(3, 3).cuda()
    D = Discriminator(3, [32, 64, 128, 256, 512, 1024], 1).cuda()
    R = TranResnet34().cuda()
    R.load_state_dict(torch.load('./TranResnet34/save_models/TranResnet34_params.pkl'))

    freeze_model(R)
    criterion_fp = nn.KLDivLoss().cuda()
    criterion_gan = nn.BCELoss().cuda()
    ssim_loss = pytorch_ssim.SSIM()
    dscsi_loss = pytorch_dscsi.COLOR_DSCSI(7,1)
    criterion_reco = nn.MSELoss().cuda()

    betas = (0.5, 0.999)
    # Optimizers
    G_optimizer = optim.Adam(G.parameters(), lr=learning_rate, betas=betas)
    D_optimizer = optim.Adam(D.parameters(), lr=learning_rate, betas=betas)
    valid = Variable(torch.ones(batch_size * 2).cuda())
    fake = Variable(torch.zeros(batch_size * 2).cuda())
    # Change paths below to where you store your data
    LOSS_DIR = "./losses"
    LOSS_SSIM_FILE = os.path.join(LOSS_DIR, "dscsi_loss.txt")
    with open(LOSS_SSIM_FILE, "w") as lossSSIM:
        for epoch in range(epochs):
            dataset_tumor_train = ImageDataset('./wsi/patches/tumor_train', './wsi/jsons/train', normalize=True)
            dataloader_tumor = DataLoader(dataset_tumor_train, batch_size=batch_size, num_workers=3)
            dataset_normal_train = ImageDataset('./wsi/patches/normal_train', './wsi/jsons/train', normalize=True)
            dataloader_normal = DataLoader(dataset_normal_train, batch_size=batch_size, num_workers=3)

            steps1 = len(dataloader_tumor) - 1  # considering list.txt
            steps2 = len(dataloader_normal) - 1
            steps = min(steps1, steps2)
            batch_size = dataloader_tumor.batch_size

            dataiter_tumor = iter(dataloader_tumor)
            dataiter_normal = iter(dataloader_normal)
            D_losses = []
            G_losses = []

            correct = 0
            total = 0
            for step in range(steps):
                # image data and labels
                data_tumor, target_tumor, _ = next(dataiter_tumor)
                data_tumor = Variable(data_tumor.cuda(), requires_grad=False)
                target_tumor = Variable(target_tumor.cuda(), requires_grad=False)
                # data_tumor_gray = Variable(data_tumor_gray.cuda(),requires_grad=False)
                data_normal, target_normal, _ = next(dataiter_normal)
                data_normal = Variable(data_normal.cuda(), requires_grad=False)
                target_normal = Variable(target_normal.cuda(), requires_grad=False)
                # data_normal_gray = Variable(data_normal_gray.cuda(),requires_grad=False)
                idx_rand = Variable(torch.randperm(batch_size * 2).cuda())
                data = torch.cat([data_tumor, data_normal])[idx_rand]
                # data_gray = torch.cat([data_tumor_gray, data_normal_gray])[idx_rand]
                target = torch.cat([target_tumor, target_normal])[idx_rand]

                # Train discriminator with real data
                D_valid_decision = D(data).squeeze()
                D_valid_loss = criterion_gan(D_valid_decision, valid)

                # Train discriminator with fake data
                data_gene = G(data)
                D_fake_decision = D(data_gene).squeeze()
                D_fake_loss = criterion_gan(D_fake_decision, fake)

                # Back propagation
                D_loss = D_valid_loss + D_fake_loss

                lossSSIM.write("%5d,%5d,%10lf," % (epoch + 1, (epoch + 1) * steps + step, D_loss.data))

                D_optimizer.zero_grad()
                D_loss.backward()
                D_optimizer.step()

                # Train generator
                data_gene = G(data)
                D_fake_decision = D(data_gene).squeeze()

                # Computer feature preservation loss
                valid_features, _ = R(data)
                valid_features = Variable(valid_features).cuda()  # , requires_grad=False)
                fake_features, result = R(data_gene)

                fake_features_lsm = F.log_softmax(fake_features, 1)
                valid_features_sm = F.softmax(valid_features, 1)
                fp_loss = criterion_fp(fake_features_lsm, valid_features_sm)

                # Compute image reconstruction loss
                mse_loss = criterion_reco(data_gene, data)
                ssim = ssim_loss(data,data_gene)
                dscsi = dscsi_loss(data, data_gene)

                reco_loss = torch.ones(1).cuda().float() - dscsi

                # Back propagation # you can also give weight to three kinds of loss below
                G_loss = 0.2 * criterion_gan(D_fake_decision, valid) + 0.4 * fp_loss + 0.4 * reco_loss

                lossSSIM.write("%10lf,%10lf,%10lf,%10lf," % (
                G_loss.data, reco_loss.data, 1-ssim.data, mse_loss.data))

                D_optimizer.zero_grad()
                G_optimizer.zero_grad()
                G_loss.backward()
                G_optimizer.step()

                D_losses.append(D_loss.item())
                G_losses.append(G_loss.item())

                # Test the result
                _, predicted = result.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()

                lossSSIM.write("%10lf\n" % (correct / total))

                time_spent = time.time() - time_now
                if (step + 1) % 20 == 0:
                    print("[Epoch %d/%d], [Step %d/%d], [D_loss: %.4f], [G_loss: %.4f],[FP_loss: %.4f], [Reco_loss: %.4f], [Accu:%3d%%], [RunTime:%.4f]"
                          % (epoch + 1, epochs, step + 1, steps, D_loss.item(), G_loss.item(),fp_loss.item(), reco_loss.item(), 100. * correct / total,
                             time_spent))

            D_avg_loss = torch.mean(torch.FloatTensor(D_losses))
            G_avg_loss = torch.mean(torch.FloatTensor(G_losses))
            print("[Epoch %d/%d], [Step %d/%d], [D_avg_loss: %.4f], [G_avg_loss: %.4f], [Accu:%3d%%], [RunTime:%.4f]"
                  % (epoch + 1, epochs, step + 1, steps, D_avg_loss, G_avg_loss, 100. * correct / total, time_spent))
            if (epoch + 1) % sample_interval == 0:
                torch.save(D.state_dict(), './save_models/D_params_dscsi_%d.pkl' % (epoch + 1))
                torch.save(G.state_dict(), './save_models/G_params_dscsi_%d.pkl' % (epoch + 1))

    print("FINAL:[Epoch %d/%d], [Step %d/%d], [D_avg_loss: %.4f], [G_avg_loss: %.4f], [Accu:%3d%%], [RunTime:%.4f]"
          % (epoch + 1, epochs, step + 1, steps, D_avg_loss, G_avg_loss, 100. * correct / total, time_spent))
    print("Model saved.")


if __name__ == '__main__':
    train()
