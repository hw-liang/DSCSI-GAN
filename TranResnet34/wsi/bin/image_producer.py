import os
import numpy as np
from torch.utils.data import Dataset
from PIL import Image

np.random.seed(0)

from torchvision import transforms  # noqa

from wsi.bin.annotation import Annotation  # noqa

class ImageDataset(Dataset):
    """
    Data producer that generate patch of image and its
    corresponding label from pre-sampled images.
    """

    def __init__(self, data_path, json_path, normalize=True):
        """
        Initialize the data producer.

        Arguments:
            data_path: string, path to pre-sampled images using patch_gen.py
            json_path: string, path to the annotations in json format

        """
        self._data_path = data_path
        self._json_path = json_path
        # self._color_jitter = transforms.ColorJitter(64.0 / 255, 0.75, 0.25, 0.04)
        self._preprocess()
        self._normalize = normalize

    def _preprocess(self):

        self._pids = list(map(lambda x: x.rstrip('.json'),
                              os.listdir(self._json_path)))

        self._annotations = {}
        for pid in self._pids:
            pid_json_path = os.path.join(self._json_path, pid + '.json')
            anno = Annotation()
            anno.from_json(pid_json_path)
            self._annotations[pid] = anno

        self._coords = []
        f = open(os.path.join(self._data_path, 'list.txt'))
        for line in f:
            pid, x_center, y_center = line.strip('\n').split(',')[0:3]
            x_center, y_center = int(x_center), int(y_center)
            self._coords.append((pid, x_center, y_center))
        f.close()
        self._num_all_image = len(self._coords)

    def __len__(self):
        return len(os.listdir(self._data_path)) - 1

    def __getitem__(self, idx):
        # extracted images from WSI is transposed with respect to
        # the original WSI (x, y)
        path = os.path.join(self._data_path, os.listdir(self._data_path)[idx])
        if path.endswith("txt"):
            idx += 1
            if idx == len(os.listdir(self._data_path)):
                idx = 0
            path = os.path.join(self._data_path, os.listdir(self._data_path)[idx])

        img = Image.open(path)
        img_gray = img.convert("L")
        
        i = int((os.listdir(self._data_path)[idx]).rstrip('.png'))
        pid, x_center, y_center = self._coords[i]

        if self._annotations[pid].inside_polygons((x_center, y_center), True):
            label = 1
        else:
            label = 0

        # color jitter
        # img = self._color_jitter(img)

        # use left_right flip
        if np.random.rand() > 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            img_gray = img_gray.transpose(Image.FLIP_LEFT_RIGHT)

        # use rotate
        num_rotate = np.random.randint(0, 4)
        img = img.rotate(90 * num_rotate)
        img_gray = img_gray.rotate(90 * num_rotate)

        # PIL image:   H x W x C
        # torch image: C X H X W
        img = np.array(img, dtype=np.float32).transpose((2, 0, 1))
        
        img_gray = np.expand_dims(img_gray, axis=2)
        img_gray = np.array(img_gray, dtype=np.float32).transpose((2, 0, 1))
        # img_gray = np.concatenate((img_gray, img_gray, img_gray), axis = 0)

        if self._normalize:
            img = (img-128.0)/128.0
            img_gray = (img_gray-128.0) / 128.0

        return img, label, img_gray

