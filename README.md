# Rethinking EfficientNet and EfficientNetV2 - Make a Powerful Shallow Network
Qimeng Tao(qt2139), Aditya Kulkarni(ak4725)

##  Description of the project

##  description of the repository


##  Introduction
Tan et. Al's EfficientNetV2 (https://arxiv.org/abs/2104.00298) family of convolutional networks are well known in the industry for (i) their high accuracies but more importantly (ii) the drastically faster training speed and better parameter efficiency than the previous EfficientNet (https://arxiv.org/abs/1905.11946). Tan et. Al's paper also provides us a great understanding of the architecture of these deep networks as well as validation results on well-known datasets such as ImageNet and CIFAR. 
In this project we intend to provide further justification to demonstrate the advancements and improvements in the latest iterations of EfficientNet (version 2) using our own 5-class flower dataset (ADD LINK/ INFO). This is done by comparing various iterations of EfficientNet, EfficientNetV2, and pre-trained EfficientNetV2 by comparing them based on size, parameters, training time, loss, etc. Further, we also demonstrate our understanding of the architecture by creating our own modified EfficientNetV2 and observe the results.

##  Dataset
We use the ”flowers” dataset from TensorFlow, a dataset comprised of 3,670 images of 5 types of flowers (daisies, dandelions, roses, sunflowers, tulips)

![sample_pics](https://user-images.githubusercontent.com/90971979/167304558-e1f37b8c-b191-473c-acd3-0835f84df6c0.png)

##  Example commands to execute the code
1. First please download the dataset, https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz
2. In train.py, please set --data-path to the absolute path of the decompressed flower_photos folder.
3. Download pretrained weights
4. If you want to train from scratch, set 'freeze-layers' to False in the fourth-to-last line in train.py, and set 'freeze-layers' to True if you want to use pretrained weights.
5. In train.py, please set the --weights parameter to the downloaded pretrained weights path.
6. After setting the path of the dataset -- data-path and the path of pre-training weights -- weights, you can use train.py to start training (the class_indices.json file will be automatically generated during the training process)
7. In predict.py, please import the same model as in the training script, and set model_weight_path to the weight path of the trained model (saved in the weights folder by default)
8. In predict.py, please set img_path to the absolute path of the image you need to predict.
9. After setting the weight path model_weight_path and the predicted image path img_path, you can use predict.py to make predictions.
10. If you want to use your own dataset, please arrange it according to the file structure of the flower classification dataset (that is, one category corresponds to one folder), and set num_classes in the training and prediction scripts to the number of categories of your own data.
11. In your terminal, go to the path of your train.py. Because we have created python scripts for different models, when you run our function, please change the python script name according to your needs, for example, if you want to use the EfficientNet_Large model trained from scratch, please refer to the code below.
```bash
python train_efficientv2_l.py
```
Running train.py will also get four pictures, which are training loss, training accuracy, validation loss, and validation accuracy. These four pictures will be saved under the same path.
If you want to use the EfficientNet_Large model with pre-trained weights, please refer to the following code.
```bash
python train_pretrain_efficientv2_l.py
```
11. When you want to use predict.py, enter this line of code to get the result, and we save the prediction result in the same path.
```bash
python predict.py
```

##  Results
We use EfficientNet and EfficientV2 and the modified EfficientV2_Large model, the modified model makes the shallow network more powerful and can extract more effective information.
First, we get 8 sub-models using 8 different scaling factors and train them on the flower dataset. The architecture of the network is shown in Fig.

![image](https://user-images.githubusercontent.com/90971979/167306416-1373a361-71c7-4167-acba-cdbd5aaf4bfa.png)

We set epoch = 30, batch size = 16, Optimizer: SGD(lr=0.01, momentum=0.9, weight_decay=1E-4), and got the following results.

| Architecture | Epoch#  | Loss | Accuracy |
| -------------| ------- | ------|-------- |
| EfficientB0  | Epoch29 | 0.071 | 0.96    |
| EfficientB1  | Epoch29 | 0.07  | 0.964   |

Next, we get Small, Medium and Large models using 3 scaling factors. The architecture of the network is shown in Fig.

![image](https://user-images.githubusercontent.com/90971979/167307212-18fab78c-0ece-432f-b60a-90adda3a55e0.png)


We train from scratch， we set epochs = 30, batch size = 8, Optimizer: SGD(lr=0.01, momentum=0.9, weight_decay=1E-4), and got the the following results.

Then we use pretrained weights (pretrained on ImageNet) and get the following results.
##  Environment
Python 3.8  
Pytorch-gpu with Cuda 11.3
Google Cloud Platform (NVIDIA Tesla T4)
