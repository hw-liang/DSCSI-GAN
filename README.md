# Stain-Style-Transfer-of-Histopathology-Images-Using-Structural-Similarity-Based-Generative-Learning
Implementation of DSCSI-GAN on histopathology images

- [Prerequisites](#prerequisites)
- [Data](#data)
    - [Whole slide images](#whole-slide-images)
    - [Annotations](#annotations)
    - [Patch images](#patch-images)
- [Training-DSCSI-GAN](#training-dscsi-gan)
    - [Resnet34](#resnet34-dscsi-gan)
    - [GAN](#gan-dscsi-gan)
- [Testing-DSCSI-GAN](#testing-dscsi-gan)
- [Training-SST](#training-sst)
    - [Resnet34](#resnet34-sst)
    - [GAN](#gan-sst)
- [Testing-SST](#testing-sst)

# Prerequisites
* Python (3.6)

* Numpy (1.14.5)

* Scipy (1.0.0)

* PIL (5.0.0)

* scikit-image (0.13.1)

* matplotlib (2.1.2)

* [PyTorch (0.4.1)/CUDA 9.0](https://pytorch.org/)

* torchvision (0.2.1)

* [openslide (1.1.1)](https://github.com/openslide/openslide-python)

* [tensorboardX](https://github.com/lanpa/tensorboard-pytorch) Standard along tensorboard that also works for PyTorch. This is mostly used in monitoring the training curves.

Most of the dependencies can be installed through pip install with version number, e.g. 
```
pip install 'numpy==1.14.5'
```
Or just simply
```
pip install numpy
```
A [requirements.txt](requirements.txt) file is also provided, so that you can install most of the dependencies at once:
```
pip install -r requirements.txt -i https://pypi.python.org/simple/
```
For PyTorch please consider downloading the specific wheel binary and use
```
pip install torch-0.4.1-cp36-cp36m-linux_x86_64.whl
```

# Data
## Whole slide images
The dataset are the whole slide images(WSI) in `*.tif` format from the [Camelyon17](https://camelyon17.grand-challenge.org/) challenge. There are dataset of Camelyon17 and Camelyon16, and we used Camelyon16. You can download from either Google Drive, or Baidu Pan. Note that, one slide is usually ~100Kx100K pixels at level 0 and 1GB+ on disk.There are 400 slides in total from two independent datasets collected in Radboud University Medical Center (Nijmegen, the Netherlands), and the University Medical Center Utrecht (Utrecht, the Netherlands), together about 700GB+, so make sure you have enough disk space. The tumor slides for training are named as `tumor_XXX.tif`, where XXX ranges from 001 to 110. The normal slides for training are named as `normal_XXX.tif`, where XXX ranges from 001 to 160. The slides for testing are named as `test_XXX.tif` where XXX ranges from 001 to 130.

Once you download all the slides, please put all the tumor slides and normal slides for training under one same directory, e.g. named `/wsi/raw_data/train`.

## Annotations
The Camelyon16 organizers also provides annotations of tumor regions for each tumor slide in xml format, located under [/wsi/lesion_annotations](/wsi/lesion_annotations/). We've converted them into some what simpler json format, located under [/wsi/all_jsons](/wsi/all_jsons/). Each annotation is a list of polygons, where each polygon is represented by its vertices. Particularly, positive polygons mean tumor region and negative polygons mean normal regions. You can also use the following command to convert the xml format into the json format.
```
python ./wsi/bin/camelyon16xml2json.py tumor_001.xml tumor_001.json
```

## Patch images
Although the original 400 WSI files contain all the necessary information, they are not directly applicable to train a deep GAN. Therefore, we have to sample much smaller image patches, e.g. 256x256, that a typical deep CNN can handle. Efficiently sampling informative and representative patches is one of the most critical parts to achieve good tumor detection performance. To ease this process, I have included the coordinates of pre-sampled patches used in the paper for training within this repo. They are located at [/wsi/coords](/wsi/coords/). Each one is a csv file, where each line within the file is in the format like `tumor_024, 25417, 127565` that the last two numbers are (x, y) coordinates of the center of each patch at level 0.`tumor_train.txt` and `normal_train.txt` contains 200,000 coordinates respectively, and `tumor_valid.txt` and `normal_valid.txt` contains 20,000 coordinates respectively. Note that, coordinates of hard negative patches, typically around tissue boundary regions, are also included within `normal_train.txt` and `normal_valid.txt`. With the original WSI and pre-sampled coordinates, we can now generate image patches for training deep CNN models. Run the six commands below to generate the corresponding patches:
```
python ./wsi/bin/patch_gen.py ./wsi/raw_data/train/ ./wsi/coords/tumor_train.txt ./wsi/patches/tumor_train/
python ./wsi/bin/patch_gen.py ./wsi/raw_data/train/ ./wsi/coords/normal_train.txt ./wsi/patches/normal_train/
python ./wsi/bin/patch_gen.py ./wsi/raw_data/valid/ ./wsi/coords/tumor_train.txt ./wsi/patches/tumor_valid/
python ./wsi/bin/patch_gen.py ./wsi/raw_data/valid/ ./wsi/coords/normal_train.txt ./wsi/patches/normal_valid/
python ./wsi/bin/patch_gen.py ./wsi/raw_data/test/ ./wsi/coords/tumor_valid.txt ./wsi/patches/tumor_test/
python ./wsi/bin/patch_gen.py ./wsi/raw_data/test/ ./wsi/coords/normal_valid.txt ./wsi/patches/normal_test/
```
where `/wsi/raw_data/train/` is the path to the directory where you put all the WSI files for training as mentioned above, and `/wsi/patches/tumor_train/` is the path to the directory to store generated tumor patches for training. Same naming applies to `/wsi/patches/normal_train/`, `/wsi/patches/normal_valid/`, `/wsi/patches/tumor_valid/`, `/wsi/patches/normal_test/`, `/wsi/patches/tumor_test/`. By default, each command is going to generate patches of size 256x256 at level 0 using 5 processes, where the center of each patch corresponds to the coordinates.

# Training-DSCSI-GAN
Training has two procedures:
1. Create training and validation dataset, and train modified ResNet34 for binary classification.
2. Based on the trained ResNet34 and training dataset:
   Use current generator to get generated image (x̂) for input (x);
   Compute the reconstruction loss;
   Feed x and x̂ to ResNet34 to get feature vector and compute feature preservation loss;
   Feed x and x̂ to current discriminator to obtain GAN-loss;
   Based on three losses, update the generator part of GAN;
   Use GAN-loss to update the discriminator part of GAN.
```
## Resnet34
With the generated patch images, we can now train the model by the following command
```
python ./TranResNet34/train_with_validation.py
```
Please modify `./wsi/patches/tumor_train`(normal_train,tumor_valid,normal_valid) respectively to your own path of generated patch images. Please also modify `./wsi/jsons/train` (valid) with respect to the full path to the repo on your machine. Typically, train_with_validation.py will generate a `train.ckpt`, which is the most recently saved model, and a `best.ckpt`, which is the model with the best validation accuracy and the model will be saved in file [/TranResnet34/save_models](/TranResnet34/save_models/).

By default, `train.py` use 1 GPU (GPU_0) to train model, 2 processes for load tumor patch images, and 2 processes to load normal patch images. On one GTX 1050, it took about 1.5 hours to train 12000 images.

## GAN
With the generated patch images and trained TranResNet34 model, we can now train the GAN by following command:
```
python ./dscsi_train.py
```
Please modify “./TranResnet34/save_models/TranResnet34_params.pkl” to where you save your pretrained classifier. Please also modify “./wsi/patches/tumor_train”   (normal_train) respectively to your own path of generated patch images. Please modify './wsi/jsons/train' (valid) with respect to the path of training (validation) repo on your machine. Typically, sst_train.py will generate a D_params.pkl and G_params.pkl, which is the most recently saved model and the model will be saved in file “/save_models/dscsi/”.

# Testing-DSCSI-GAN
The main testing result from a trained model for WSI analysis is the classification result that represents whether the model judges the image is tumor or normal. The testing image is from a second distinct dataset-- Utrecht University, which can prove the generalization capacity of the model.
By using the following command:
```
python ./dscsi_test.py
```
Please modify “./TranResnet34/save_models/TranResnet34_params.pkl” to your own path of trained ResNet34 model and “./save_models/dscsi/G_params.pkl” to your path of saving DSCSI-GAN model. Please also modify “./wsi/patches/tumor_test” (normal_test) respectively to your own path of generated patch images. Please also modify “./wsi/jsons/test” with respect to the full path to the repo on your machine. Typically, dscsi_test.py will generate the result of classification task, including Accuracy, Precision, Recall and AUC, and save the generated images along with their original counterpart into file “/demo/dscsi/”.
In addition, we can also sample the generated images of the model. Uncomment the `sample_images` function implementation in dscsi_test.py, and you could get the original and genereated images. Please modify `./demo/test_result/%d_%d.png` in this function to your own path of generated test images.

# Training-DSCSI-GAN

Training has two procedures:
```
1. Create training and validation dataset, and train modified ResNet34 for binary classification.
2. Based on the trained ResNet34 and training dataset:
   Use current generator to get generated image (x̂) for input (x);
   Compute the reconstruction loss;
   Feed x and x̂ to ResNet34 to get feature vector and compute feature preservation loss;
   Feed x and x̂ to current discriminator to obtain GAN-loss;
   Based on three losses, update the generator part of GAN;
   Use GAN-loss to update the discriminator part of GAN.
```
## Resnet34
With the generated patch images, we can now train the model by the following command
```
python ./TranResNet34/train_with_validation.py
```
Please modify `./wsi/patches/tumor_train`(normal_train,tumor_valid,normal_valid) respectively to your own path of generated patch images. Please also modify `./wsi/jsons/train` (valid) with respect to the full path to the repo on your machine. Typically, train_with_validation.py will generate a `train.ckpt`, which is the most recently saved model, and a `best.ckpt`, which is the model with the best validation accuracy and the model will be saved in file [/TranResnet34/save_models](/TranResnet34/save_models/).

By default, `train.py` use 1 GPU (GPU_0) to train model, 2 processes for load tumor patch images, and 2 processes to load normal patch images. On one GTX 1050, it took about 1.5 hours to train 12000 images.

## GAN
With the generated patch images and trained TranResNet34 model, we can now train the GAN by following command:
```
python ./sst_train.py
```
Please modify `./wsi/patches/tumor_train`(normal_train) respectively to your own path of generated patch images. Please also modify `./wsi/jsons/train` (valid) with respect to the path to the repo on your machine. Typically, sst_train_cuda.py will generate a D_params.pkl and G_params.pkl, which is the most recently saved model and the model will be saved in file [/save_models/](/save_models/).

# Testing-SST
The main testing result from a trained model for WSI analysis is the classification result that represents whether the model judges the image is tumor or normal. The testing image is from a second distinct dataset-- Utrecht University, which can prove the generalization capacity of the model.
By using the following command:
```
python ./sst_test.py
```
Please modify `./TranResnet34/save_models/train.ckpt` to your own path of storing trained ResNet34 model. Please also modify `/wsi/patches/tumor_test`
(normal_test) respectively to your own path of generated patch images for testing. Please also modify `./wsi/jsons/test` with respect to the full path
to the repo on your machine. Typically, sst_test.py will generate the result of classification task, including Accuracy, Precision, Recall and AUC.

In addition, we can also sample the generated images of the model. Uncomment the `sample_images` function implementation in sst_test.py, and you could get the original and genereated images. Please modify `./demo/test_result/%d_%d.png` in this function to your own path of generated test images.

