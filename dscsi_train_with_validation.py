import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch import optim
import numpy as np
from wsi.bin.image_producer import ImageDataset

import pytorch_ssim
from unet_generator import UNet
from dcgan_discriminator import Discriminator
from resnet_classifier import TranResnet34

torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

def train(G,D, R,epochs, batch_size = 1, learning_rate = 0.0002, sample_interval = 1):

    time_now = time.time()

    criterion_fp = nn.KLDivLoss().cuda()
    criterion_gan = nn.BCELoss().cuda()
    ssim_loss = pytorch_ssim.SSIM()
    criterion_reco = nn.MSELoss().cuda()

    betas = (0.5, 0.999)
    # Optimizers
    G_optimizer = optim.Adam(G.parameters(), lr=learning_rate, betas=betas)
    D_optimizer = optim.Adam(D.parameters(), lr=learning_rate, betas=betas)
    valid = Variable(torch.ones(batch_size * 2).cuda())
    fake = Variable(torch.zeros(batch_size * 2).cuda())
    # Change paths below to where you store your data
    # LOSS_DIR = "./losses"
    # LOSS_SSIM_FILE = os.path.join(LOSS_DIR,"new_ssim_loss_4.txt")
    # with open(LOSS_SSIM_FILE,"w") as lossSSIM:
    for epoch in range(1):
        dataset_tumor_train = ImageDataset('./wsi/patches/tumor_train','./wsi/jsons/train',normalize=True)
        dataloader_tumor = DataLoader(dataset_tumor_train, batch_size=batch_size, num_workers=4)
        dataset_normal_train = ImageDataset('./wsi/patches/normal_train','./wsi/jsons/train',normalize=True)
        dataloader_normal = DataLoader(dataset_normal_train, batch_size=batch_size, num_workers=4)

        steps1 = len(dataloader_tumor)-1 # considering list.txt
        steps2 = len(dataloader_normal)-1
        steps = min(steps1,steps2)
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
            data_tumor = Variable(data_tumor.cuda(),requires_grad=False)
            target_tumor = Variable(target_tumor.cuda(),requires_grad=False)
            # data_tumor_gray = Variable(data_tumor_gray.cuda(),requires_grad=False)

            data_normal, target_normal, _ = next(dataiter_normal)
            data_normal = Variable(data_normal.cuda(),requires_grad=False)
            target_normal = Variable(target_normal.cuda(),requires_grad=False)
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

            # lossSSIM.write("%5d,%5d,%10lf," % (epoch+4, (epoch+4)*steps+step,D_loss.data))

            D_optimizer.zero_grad()
            D_loss.backward()
            D_optimizer.step()

            # Train generator
            data_gene = G(data)
            D_fake_decision = D(data_gene).squeeze()
            # Computer feature preservation loss
            valid_features, _ = R(data)
            valid_features = Variable(valid_features.cuda())#, requires_grad=False)
            fake_features, result = R(data_gene)
            fake_features_lsm = F.log_softmax(fake_features,1)
            valid_features_sm = F.softmax(valid_features,1)
            fp_loss = criterion_fp(fake_features_lsm, valid_features_sm)
            # Compute image reconstruction loss
            # reco_loss = criterion_reco(data_gene, data)
            ssim = ssim_loss(data,data_gene)
            reco_loss = torch.ones(1).cuda().float() - ssim

            # Back propagation # you can also give weight to three kinds of loss below
            G_loss = 0.2 * criterion_gan(D_fake_decision, valid) + 0.4 * fp_loss + 0.4 * reco_loss

            # lossSSIM.write("%10lf,%10lf,%10lf,%10lf," % (G_loss.data, reco_loss.data, ssim.data, criterion_reco(data_gene, data).data))

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

            # lossSSIM.write("%10lf\n" % (correct/total))

            time_spent = time.time() - time_now
            if (step+1) % 20 == 0:
                print("[Epoch %d], [Step %d/%d], [D_loss: %.4f], [G_loss: %.4f], [Accu:%3d%%], [RunTime:%.4f]"
                    % (epochs+1, step+1, steps, D_loss.item(), G_loss.item(), 100.*correct/total, time_spent))

            D_avg_loss = torch.mean(torch.FloatTensor(D_losses))
            G_avg_loss = torch.mean(torch.FloatTensor(G_losses))
            # print("[Epoch %d/%d], [Step %d/%d], [D_avg_loss: %.4f], [G_avg_loss: %.4f], [Accu:%3d%%], [RunTime:%.4f]"
            #     % (epoch + 1, epochs, step + 1, steps, D_avg_loss, G_avg_loss, 100.*correct/total, time_spent))
            # if (epoch+1) % sample_interval == 0:
            #     torch.save(D.state_dict(), './save_models/D_params_new_%d.pkl'%(epoch+5))
            #     torch.save(G.state_dict(), './save_models/G_params_new_%d.pkl'%(epoch+5))
    torch.save(D.state_dict(), './save_models/D_params_%d.pkl'%(epochs+1))
    torch.save(G.state_dict(), './save_models/G_params_%d.pkl'%(epochs+1))
    print("FINAL:[Epoch %d], [Step %d/%d], [D_avg_loss: %.4f], [G_avg_loss: %.4f], [Accu:%3d%%], [RunTime:%.4f]"
          % (epochs+1, step + 1, steps, D_avg_loss, G_avg_loss, 100.*correct/total, time_spent))
    print("Model saved.")
    return G,D,R

def valid(G,D,R, epochs, batch_size = 1):
    time_now = time.time()
    TOTAL = 0
    CORRECT = 0
 
    for epoch in range(1):
        dataset_tumor_test = ImageDataset('./wsi/patches/tumor_test','./wsi/jsons/test',normalize=True)
        dataloader_tumor = DataLoader(dataset_tumor_test, batch_size=batch_size, num_workers=2)
        dataset_normal_test = ImageDataset('./wsi/patches/normal_test','./wsi/jsons/test',normalize=True)
        dataloader_normal = DataLoader(dataset_normal_test, batch_size=batch_size, num_workers=2)

        steps1 = len(dataloader_tumor)-1 #considering list.txt
        steps2 = len(dataloader_normal)-1
        steps = min(steps1,steps2)
        batch_size = dataloader_tumor.batch_size
        dataiter_tumor = iter(dataloader_tumor)
        dataiter_normal = iter(dataloader_normal)

        correct = 0
        total = 0
        y_score = np.array([])
        y_true = np.array([])
        y_pred = np.array([])
        for step in range(steps):
            # image data and labels
            data_tumor, target_tumor, _ = next(dataiter_tumor)
            data_tumor = Variable(data_tumor.cuda(), requires_grad=False)
            target_tumor = Variable(target_tumor.cuda(), requires_grad=False)
            # data_tumor_gray = Variable(data_tumor_gray.cuda())

            data_normal, target_normal, _ = next(dataiter_normal)
            data_normal = Variable(data_normal.cuda(), requires_grad=False)
            target_normal = Variable(target_normal.cuda(), requires_grad=False)
            # data_normal_gray = Variable(data_normal_gray.cuda())

            idx_rand = Variable(torch.randperm(batch_size * 2).cuda(), requires_grad=False)
            # data_gray = torch.cat([data_tumor_gray, data_normal_gray])[idx_rand]
            data = torch.cat([data_tumor, data_normal])[idx_rand]
            target = torch.cat([target_tumor, target_normal])[idx_rand]

            data_gene = G(data)
            _, result = R(data_gene)

            _, predicted = result.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

            if step == 0:
                y_score = result.detach().cpu().numpy()
            else:
                y_score = np.vstack((y_score,result.detach().cpu().numpy()))
            y_true = np.append(y_true, target.detach().cpu().numpy())
            y_pred = np.append(y_pred,predicted.cpu().numpy())
            # If you want to sample generated images, uncomment the below two lines
            # if (step+1) % 30 == 0:
            #     sample_images(data,data_gene,epoch,step)
            time_spent = time.time() - time_now
            if (step+1) % 20 == 0:
                print("[Epoch %d], [Step %d/%d], [Accu:%3d%%], [RunTime:%.4f]"
                      % (epochs+1, step+1, steps, 100.*correct/total, time_spent))
        TOTAL += total
        CORRECT += correct

        # np.savetxt("y_score.txt",y_score,fmt="%.5f",delimiter=",")
        # np.savetxt("y_true.txt",y_true,fmt="%d",delimiter=",")
        # np.savetxt("y_pred.txt",y_pred,fmt="%d",delimiter=",")

        print("Average_precision_score : " + str(metrics.average_precision_score(y_true, y_score[:,1])))
        print("Roc_auc_score : " + str(metrics.roc_auc_score(y_true, y_score[:,1])))
        print("Recall : " + str(metrics.recall_score(y_true, y_pred)))
        print("Accuracy : " + str(metrics.accuracy_score(y_true, y_pred)))

        # print("[Epoch %d/%d], [Step %d/%d], [Accu:%3d%%], [RunTime:%.4f]"
        #       % (epoch + 1, epochs, step + 1, steps, 100.*correct/total, time_spent))
    print("FINAL:[Epoch %d], [Step %d/%d], [Average accu:%3d%%], [RunTime:%.4f]"
          % (epochs, step + 1, steps, 100.*CORRECT/TOTAL, time_spent))
    print("Finish Test.")

def freeze_model(model):
    model.eval()
    for params in model.parameters():
        params.requires_grad = False

if __name__ == '__main__':
    G = UNet(3,3).cuda()
    D = Discriminator(3, [32,64,128, 256, 512, 1024], 1).cuda()
    R = TranResnet34().cuda()
    R.load_state_dict(torch.load('./TranResnet34/save_models/best.ckpt')['state_dict'])
    freeze_model(R)
    # valid(G,D,R,0)
    for epoch in range(20):
        G,D,R = train(G,D,R,epoch)
        # valid(G,D,R,epoch+1)
