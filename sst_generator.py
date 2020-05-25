import torch
import numpy as np
import os
from fusionnet_generator import FusionGenerator
from PIL import Image

torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

def generate(source_path='./wsi/patches/tumor_test', save_path='./demo/sst/', model_path = './save_models/G_params_sst.pkl'):
    G = FusionGenerator(1,3,16).cuda()
    G.load_state_dict(torch.load(model_path))
    try:
        os.makedirs(save_path)
    except OSError:
        pass
    for img_name in os.listdir(source_path):
        if img_name.endswith('.txt'):
            continue
        img_path = os.path.join(source_path, img_name)
        img_gray = Image.open(img_path).convert('L')

        img_gray = np.expand_dims(img_gray, axis=2)
        img_gray = np.array(img_gray, dtype=np.float32).transpose((2, 0, 1))
        img_gray = (img_gray - 128.0) / 128.0
        img_gray = torch.from_numpy(img_gray).float().unsqueeze(0).cuda()

        data_gene = G(img_gray)
        img_show = np.transpose(data_gene.detach().cpu().numpy(), (0, 2, 3, 1)) * 0.5 + 0.5
        img_show = np.squeeze(img_show)
        new_path = os.path.join(save_path, img_name)
        result = Image.fromarray((img_show * 255).astype(np.uint8))
        result.save(new_path)

    print("Finish Generating.")

if __name__ == '__main__':
    generate()
