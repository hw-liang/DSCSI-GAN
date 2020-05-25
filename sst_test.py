import time
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics

from wsi.bin.image_producer import ImageDataset
from fusionnet_generator import FusionGenerator
from resnet_classifier import TranResnet34

torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

def test(epochs = 1, batch_size = 1):

    time_now = time.time()

    G = FusionGenerator(1, 3, 16).cuda()
    R = TranResnet34().cuda()
    R.load_state_dict(torch.load('./TranResnet34/save_models/TranResnet34_params.pkl'))
    # OR: R.load_state_dict(torch.load('./TranResnet34/save_models/best.ckpt')['state_dict'])
    G.load_state_dict(torch.load('./save_models/G_params_sst.pkl'))

    TOTAL = 0
    CORRECT = 0

    for epoch in range(epochs):
        dataset_tumor_test = ImageDataset('./wsi/patches/tumor_test','./wsi/jsons/test',normalize=True)
        dataloader_tumor = DataLoader(dataset_tumor_test, batch_size=batch_size, num_workers=2)
        dataset_normal_test = ImageDataset('./wsi/patches/normal_test','./wsi/jsons/test',normalize=True)
        dataloader_normal = DataLoader(dataset_normal_test, batch_size=batch_size, num_workers=2)

        steps1 = len(dataloader_tumor)-1  # consider list.txt
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
            data_tumor, target_tumor, data_tumor_gray = next(dataiter_tumor)
            target_tumor = Variable(target_tumor.cuda())
            data_tumor_gray = Variable(data_tumor_gray.cuda())

            data_normal, target_normal, data_normal_gray = next(dataiter_normal)
            target_normal = Variable(target_normal.cuda())
            data_normal_gray = Variable(data_normal_gray.cuda())

            idx_rand = Variable(torch.randperm(batch_size * 2).cuda())
            data_gray = torch.cat([data_tumor_gray, data_normal_gray])[idx_rand]
            data = torch.cat([data_tumor, data_normal])[idx_rand]
            target = torch.cat([target_tumor, target_normal])[idx_rand]

            data_gene = G(data_gray)
            _, result = R(data_gene)

            _, predicted = result.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

            if step == 0:
                y_score = result.detach().cpu().numpy()
            else:
                y_score = np.vstack((y_score,result.detach().cpu().numpy()))

            y_true = np.append(y_true, target.detach().cpu().numpy())
            y_pred = np.append(y_pred, predicted.detach().cpu().numpy())
            # If you want to sample generated images, uncomment the below two lines
            # if (step+1) % 30 == 0:
            #     sample_images(data,data_gene,epoch,step)
            time_spent = time.time() - time_now
            if (step+1) % 20 == 0:
                print("[Epoch %d/%d], [Step %d/%d], [Accu:%3d%%], [RunTime:%.4f]"
                      % (epoch+1, epochs, step+1, steps, 100.*correct/total, time_spent))
        TOTAL += total
        CORRECT += correct

        # np.savetxt("y_score.txt",y_score,fmt="%.5f",delimiter=",")
        # np.savetxt("y_true.txt",y_true,fmt="%d",delimiter=",")
        # np.savetxt("y_pred.txt",y_pred,fmt="%d",delimiter=",")

        print("Average_precision_score : " + str(metrics.average_precision_score(y_true, y_score[:,1])))
        print("Roc_auc_score : " + str(metrics.roc_auc_score(y_true, y_score[:,1])))
        print("Recall : " + str(metrics.recall_score(y_true, y_pred)))
        print("Accuracy : " + str(metrics.accuracy_score(y_true, y_pred)))

        print("[Epoch %d/%d], [Step %d/%d], [Accu:%3d%%], [RunTime:%.4f]"
              % (epoch + 1, epochs, step + 1, steps, 100.*correct/total, time_spent))
    print("FINAL:[Epoch %d/%d], [Step %d/%d], [Average accu:%3d%%], [RunTime:%.4f]"
          % (epoch + 1, epochs, step + 1, steps, 100.*CORRECT/TOTAL, time_spent))
    print("Finish Test.")

def sample_images(original,gene,epoch,step):
    batchs = original.shape[0]
    data_show = np.transpose(np.vstack((original.detach().cpu().numpy(),gene.detach().cpu().numpy())),(0,2,3,1))
    data_show = data_show * 0.5 + 0.5

    titles = ['Original', 'Generated']
    r=2
    c=batchs
    fig, axs = plt.subplots(r, c)
    cnt = 0
    for i in range(r):
        for j in range(c):
            axs[i,j].imshow(data_show[cnt])
            axs[i,j].set_title(titles[i])
            axs[i,j].axis('off')
            cnt += 1
    fig.savefig("./demo/ssim/%d_%d.png" % (epoch+1, step+1))
    plt.close()

if __name__ == '__main__':
    test()