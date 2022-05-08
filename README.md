# Rethinking EfficientNet and EfficientNetV2 - Make a Powerful Shallow Network
Qimeng Tao(qt2139), Aditya Kulkarni(ak4725)

##  Introduction
Tan et. Al's EfficientNetV2 (https://arxiv.org/abs/2104.00298) family of convolutional networks are well known in the industry for (i) their high accuracies but more importantly (ii) the drastically faster training speed and better parameter efficiency than the previous EfficientNet (https://arxiv.org/abs/1905.11946). Tan et. Al's paper also provides us a great understanding of the architecture of these deep networks as well as validation results on well-known datasets such as ImageNet and CIFAR. 
In this project we intend to provide further justification to demonstrate the advancements and improvements in the latest iterations of EfficientNet (version 2) using our own 5-class flower dataset (ADD LINK/ INFO). This is done by comparing various iterations of EfficientNet, EfficientNetV2, and pre-trained EfficientNetV2 by comparing them based on size, parameters, training time, loss, etc. Further, we also demonstrate our understanding of the architecture by creating our own modified EfficientNetV2 and observe the results.

##  Dataset
We use the ”flowers” dataset from TensorFlow, a dataset comprised of 3,670 images of 5 types of flowers (daisies, dandelions, roses, sunflowers, tulips)

![sample_pics](https://user-images.githubusercontent.com/90971979/167304558-e1f37b8c-b191-473c-acd3-0835f84df6c0.png)


##  Results
We use EfficientNet and EfficientV2 and the modified EfficientV2_Large model, the modified model makes the shallow network more powerful and can extract more effective information.
First, we get 8 sub-models using 8 different scaling factors and train them on the flower dataset. The architecture of the network is shown in Fig.

![image](https://user-images.githubusercontent.com/90971979/167306416-1373a361-71c7-4167-acba-cdbd5aaf4bfa.png)

We set epoch = 30, batch size = 16, Optimizer: SGD(lr=0.01, momentum=0.9, weight_decay=1E-4), and got the following results.

| Architecture | Epoch#  | Loss | Accuracy |
| -------------| ------- | --------------- |
| EfficientB0  | Epoch29 | 0.071 | 0.96    |
| EfficientB1  | Epoch29 | 0.07  | 0.964   |

| Network Structure | Error | Reference Error | Traning Times |
| ----------------- | ----- | --------------- | ------------- |
| With Shortcut     | 6.71% | 5.52% | 107s / epochs(bath size = 64) |
| Without Shortcut   | 6.68% | 5.52% | 98s  / epochs(bath size = 64) |

##  Environment
Python 3.8  
Pytorch-gpu with Cuda 11.3
Google Cloud Platform (NVIDIA Tesla T4)
