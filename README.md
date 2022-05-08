# Rethinking EfficientNet and EfficientNetV2 - Making shallow networks more powerful
Qimeng Tao(qt2139), Aditya Kulkarni(ak4725)

##  Description of the project
This is my final project for COMS 6998 at Columbia University and my advisor is Prof. Parijat Dube. In this final project we discuss the impact of strengthening shallow networks on accuracy. 

When we use a convolutional neural network, the shallow layer will extract Edges, Texture, and Colors in the image, and the deep convolution will contain some local features. After summarizing all the information, we can get the final classification. 


According to Visualizing and Understanding Convolutional Networks (https://arxiv.org/abs/1311.2901), in order to obtain higher accuracy, make the shallow network more powerful, and let the shallow network extract more information is crucial.

We use EfficientNet and EfficientNetV2 for comparison, because EfficientNet not only has fast training speed and high accuracy. Our modified model is also based on EfficientNetV2_Large. The experimental results show that when the shallow neural network has more powerful performance, the accuracy increases.


##  Description of the repository
This repository is our code and it contains two folders, EfficientNet and EfficientNetV2. Our goal is to perform image classification faster and more accurately. Now it contains 12 models, namely EfficientNetB0-B7, EfficientNetV2 Small, EfficientNetV2 Medium, EfficientNetV2 Large and Modified models.

Our core idea is to make shallow networks more powerful. To achieve our purpose, we use two methods. For the first method, we refer to Deep Networks with Stochastic Depth (https://arxiv.org/abs/1603.09382) . This method is to randomly discard the main branch of each block in the network. As can be seen from the figure, if the current Block is in the inactive state, then the output of Block t-1 is the input of Block t+1. We set a linear decay function, based on the larger weights of the shallow network, so that each block of the shallow network has a greater probability of being saved during training.

![image](https://user-images.githubusercontent.com/90971979/167310056-7202f2aa-7af8-45c4-8aba-6324e82423e1.png)


The second method is to replace MBConv in shallow network with Fused-MBConv. Because DepthWise convolution is used in MBConv, although it has fewer training parameters than ordinary convolution, it also makes the shallow layer of the network lose the ability to learn.
With these two methods, our modified model has higher accuracy.

##  Introduction
Tan et. Al's EfficientNetV2 (https://arxiv.org/abs/2104.00298) family of convolutional networks are well known in the industry for (i) their high accuracies but more importantly (ii) the drastically faster training speed and better parameter efficiency than the previous EfficientNet (https://arxiv.org/abs/1905.11946). Tan et. Al's paper also provides us a great understanding of the architecture of these deep networks as well as validation results on well-known datasets such as ImageNet and CIFAR. 
In this project we intend to provide further justification to demonstrate the advancements and improvements in the latest iterations of EfficientNet (version 2) using our own 5-class flower dataset (ADD LINK/ INFO). This is done by comparing various iterations of EfficientNet, EfficientNetV2, and pre-trained EfficientNetV2 by comparing them based on size, parameters, training time, loss, etc. Further, we also demonstrate our understanding of the architecture by creating our own modified EfficientNetV2 and observe the results.

##  Dataset
We use the ”flowers” dataset from TensorFlow, a dataset comprised of 3,670 images of 5 types of flowers (daisies, dandelions, roses, sunflowers, tulips)

![sample_pics](https://user-images.githubusercontent.com/90971979/167304558-e1f37b8c-b191-473c-acd3-0835f84df6c0.png)</a>

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
Google Cloud Platform (NVIDIA Tesla T4)  
Python 3.8  
Pytorch-gpu with Cuda 11.3
