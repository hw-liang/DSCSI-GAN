import time
import torch
from torch.autograd import Variable
import numpy as np

import os
from PIL import Image
import scipy.misc
from unet_generator import UNet
 
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

def new_generator(source_path='./wsi/patches/normal_test', save_path='./demo/dscsi/', model_path = './save_models/G_params_dscsi.pkl'):
    G = UNet(3,3).cuda()
    # To load new dscsi loss based model
    G.load_state_dict(torch.load(model_path))

    try:
        os.makedirs(save_path)
    except OSError:
        pass

    for fn in os.listdir(source_path):
        path = os.path.join(source_path, fn)
        if path.endswith("txt"):
            continue
        img = Image.open(path)
        img = np.array(img, dtype=np.float32).transpose((2, 0, 1))
        img = (img-128.0)/128.0
        img = torch.from_numpy(img)
        img = Variable(img.cuda(), requires_grad=False).unsqueeze(0)

        img_gene = G(img)
        img_save = np.transpose(img_gene.detach().cpu().numpy(),(0,2,3,1))[0] * 0.5 + 0.5
        # fn = fn[:-4] + "_3" + fn[-4:]

        new_path = os.path.join(save_path,fn)
        scipy.misc.imsave(new_path,img_save)
      
    print("Finish Generating.")


if __name__ == '__main__':
    new_generator()
