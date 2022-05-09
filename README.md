# Rethinking EfficientNet and EfficientNetV2 - Making Shallow Networks More Powerful
Collaborators: Qimeng Tao (qt2139), Aditya Kulkarni (ak4725)

##  Description of the project
This is our final project for COMS 6998 at Columbia University and our advisor is Prof. Parijat Dube. 

In this final project we discuss the impact of strengthening shallow networks on accuracy. 

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

Efficient proposed Compound Scaling, in which we scale either the depth, using a constant ratio in three dimensions- width, depth, and resolution. Traditional CNN's were scaled randomly based on depth. However, this causes the accury to saturate due to the additional layers- leading to the degradation problem. Compound Scaling, on the other hand, scales the three dimensions using a constant ratio thus balancing the network across them. Further, EfficientNetV2 brings in the concept of progressive learning- increasing the image sizes progressively as training continues- which drastically reduces training time. 

<p align="center">
  <img width="1175" height="484" src="https://github.com/qt2139/Rethinking-EfficientNet-and-EfficientNetV2/blob/main/EfficientNetV2/utils/model%20scaling.png">
</p>

In this project we intend to provide further justification to demonstrate the advancements and improvements in the latest iterations of EfficientNet (version 2) using our own 5-class flower dataset (described below). This is done by comparing various iterations of EfficientNet, EfficientNetV2, and pre-trained EfficientNetV2 by comparing them based on size, parameters, training time, loss, etc. Further, we also demonstrate our understanding of the architecture by creating our own modified EfficientNetV2 and observe the results.





##  Dataset
We use the ”flowers” dataset from TensorFlow, a dataset comprised of 3,670 images of 5 types of flowers (daisies, dandelions, roses, sunflowers, tulips)

<p align="center">
  <img width="568" height="384" src="https://github.com/qt2139/Rethinking-EfficientNet-and-EfficientNetV2/blob/main/EfficientNetV2/utils/dataset_sample.png">
</p>

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
Running train.py will also get four images, which are training loss, training accuracy, validation loss, and validation accuracy. These four pictures will be saved under the same path.
If you want to use the EfficientNet_Large model with pre-trained weights, please refer to the following code.
```bash
python train_pretrain_efficientv2_l.py
```
12. When you want to use predict.py, enter this line of code to get the result, and we save the prediction result in the same path.
```bash
python predict.py
```

For EfficientNet (version 1):
- EfficientNet can be run in similar fashion as above. 
- Pretrained models are stored in torch_efficientnet folder
- Data is same data- ```flower_data``` folder
- To train the model, add image paths in utils.py, choose model through ```num_model``` parameter in train.py, and finally run the following command: ``` python train.py ```
- This should create a model. Call ```python predict.py``` to check results. Ensure you change ```img_path``` parameter in predict.py.


##  Results
This is our Solution architecture. We will experiment according to this flowchart.

<p align="center">
  <img width="1255" height="721" src="https://github.com/qt2139/Rethinking-EfficientNet-and-EfficientNetV2/blob/main/EfficientNetV2/utils/Rethinking%20EfficientNet%20and%20EfficientNetV2.png">
</p>

We use EfficientNet and EfficientV2 and the modified EfficientV2_Large model. The modified model makes the shallow network more powerful and can extract more effective information.  
First, we get 8 sub - EfficientNet models by using 8 different scaling factors and train them on the flower dataset. The architecture of the EfficientNet is shown in table.

<p align="center">
  <img width="932" height="448" src="https://github.com/qt2139/Rethinking-EfficientNet-and-EfficientNetV2/blob/main/EfficientNetV2/utils/table1.png">
</p>

We set epoch = 30, batch size = 16, Optimizer: SGD(lr=0.01, momentum=0.9, weight_decay=1E-4), and got the following results.


  
| **Model Architecture** | **Epoch#** | **Loss** | **Accuracy** |
|:--------------------------:|:-------------------:|:---------------:|:-------------------:|
| EfficientNet-B0          | Epoch 29          | 0.071         | 0.96              |
| EfficientNet-B1          | Epoch 29          | 0.07          | 0.964             |
| EfficientNet-B2          | Epoch 29          | 0.066         | 0.967             |
| EfficientNet-B3          | Epoch 29          | 0.063         | 0.972             |
| EfficientNet-B4          | Epoch 29          | 0.6           | 0.974             |
| EfficientNet-B5          | Epoch 29          | 0.58          | 0.977             |
| EfficientNet-B6          | Epoch 29          | 0.57          | 0.979             |
| EfficientNet-B7          | Epoch 29          | 0.56          | 0.981             |

1. From the experimental results, it can be seen that when EfficientNet increases the image width, depth, and resolution, the accuracy will be improved, but the corresponding cost is increased training time.
2. It can be seen from the training time and training parameters that compared with other mainstream networks, such as ResNet, EfficientNet greatly reduces the training parameters and training time due to the use of Depthwise convolution, but still has a high accuracy.



Next, we get Small, Medium and Large models using 3 scaling factors. The architecture of the network is shown in table.

<p align="center">
  <img width="884" height="375" src="https://github.com/qt2139/Rethinking-EfficientNet-and-EfficientNetV2/blob/main/EfficientNetV2/utils/table2.png">
</p>

| **Model Architecture**        | **Dataset, Epoch Info** | **Loss** | **Accuracy** | **Time Taken**                    |
|:-------------------------------:|:----------------------------:|:-------------:|:-----------------:|:--------------------------------------:|
| EfficientNetV2 (Small)          | Train, Epoch 29              | 0.074         | 0.978             | [01:15<00:00,  4.88it/s] |
| EfficientNetV2 (Small)          | Valid, Epoch 29              | 0.075         | 0.982             | [00:07<00:00, 11.65it/s] |
| EfficientNetV2 (Medium)         | Train, Epoch 29              | 0.052         | 0.985             | [02:04<00:00,  2.95it/s] |
| EfficientNetV2 (Medium)         | Valid, Epoch 29              | 0.100         | 0.973             | [00:11<00:00,  8.07it/s] |
| EfficientNetV2 (Large)          | Train, Epoch 29              | 0.053         | 0.982             | [04:00<00:00,  1.53it/s] |
| EfficientNetV2 (Large)          | Valid, Epoch 29              | 0.074         | 0.983             | [00:21<00:00,  4.15it/s] |
| Modified EfficientNetV2 (Large) | Train, Epoch 29              | 0.049         | 0.984             | [04:20<00:00,  1.47it/s] |
| Modified EfficientNetV2 (Large) | Valid, Epoch 29              | 0.071         | 0.984             | [00:23<00:00,  3.95it/s] |

We train from scratch, we set epochs = 30, batch size = 8, Optimizer: SGD(lr=0.01, momentum=0.9, weight_decay=1E-4), and got the the following results.This image is the validation accuracy of EfficientNetV2_Large trained from scratch.

<p align="center">
  <img width="1000" height="800" src="https://github.com/qt2139/Rethinking-EfficientNet-and-EfficientNetV2/blob/main/EfficientNetV2/results/mypretrain_efficientnetv2_l_val_accuracy.png">
</p>

By training EfficientNetV2, we can find that EfficientNetV2 has better performance than EfficientNet, thanks to the improvement of shallow network. NVIDIA's convolution technology also enables the network to have a faster training speed even if Depthwise convolution is not used, and the performance of EfficientNetV2 exceeds that of EfficientNet.

Then we use pretrained weights (pretrained on ImageNet) and get the following results.

| **Model Architecture**        | **Dataset, Epoch Info** | **Loss** | **Accuracy** | **Time Taken**                    |
|:------------------------------------:|:------------------------------:|:---------------:|:-------------------:|:----------------------------------------:|
| Pretrained EfficientNetV2 (Small)  | Train, Epoch 29              | 0.239         | 0.920             | [00:24<00:00, 14.76it/s] |
| Pretrained  EfficientNetV2 (Small) | Valid, Epoch 29              | 0.137         | 0.960             | [00:07<00:00, 11.87it/s] |
| Pretrained EfficientNetV2 (Medium) | Train, Epoch 29              | 0.237         | 0.923             | [00:36<00:00, 10.03it/s] |
| Pretrained EfficientNetV2 (Medium) | Valid, Epoch 29              | 0.131         | 0.964             | [00:11<00:00,  7.73it/s] |
| Pretrained EfficientNetV2 (Large)  | Train, Epoch 29              | 0.224         | 0.928             | [01:13<00:00,  5.02it/s] |
| Pretrained EfficientNetV2 (Large)  | Valid, Epoch 29              | 0.118         | 0.966             | [00:23<00:00,  3.98it/s] |

This image is the validation accuracy of EfficientNetV2_Large with pretrained weights.
<p align="center">
  <img width="1000" height="800" src="https://github.com/qt2139/Rethinking-EfficientNet-and-EfficientNetV2/blob/main/EfficientNetV2/results/my_efficientnetv2_l_val_accuracy.png">
</p>
By training the model with pretrained weights, we can see that the training time with pretrained weights is reduced by about 3-4 times compared to training from scratch. At the same time, the accuracy did not decrease significantly. This is really important in industry.

This is our modified model architecture.
<p align="center">
  <img width="953" height="306" src="https://github.com/qt2139/Rethinking-EfficientNet-and-EfficientNetV2/blob/main/EfficientNetV2/utils/table3.png">
</p>

Finally, as shown in the table, through our two methods, observing the experimental results of our modified model, we find that the accuracy of the network does improve when we enhance the performance of the shallow network. This proves our point.

This image is the validation accuracy of our modified model.
<p align="center">
  <img width="1000" height="800" src="https://github.com/qt2139/Rethinking-EfficientNet-and-EfficientNetV2/blob/main/EfficientNetV2/results/modified_efficientnetv2_l_val_accuracy.png">
</p>

We also used the EfficientNetV2_Large weights trained from scratch to make predictions, and the following road pictures are the prediction results.
<p align="center">
  <img width="640" height="480" src="https://github.com/qt2139/Rethinking-EfficientNet-and-EfficientNetV2/blob/main/EfficientNetV2/results/predict1.png">
</p>
<p align="center">
  <img width="640" height="480" src="https://github.com/qt2139/Rethinking-EfficientNet-and-EfficientNetV2/blob/main/EfficientNetV2/results/predict2.png">
</p>
As can be seen from the prediction results, our model has very good performance.  
Finally, we experimentally demonstrate that accuracy can be improved by enhancing the capabilities of shallow networks.


##  Environment
- Google Cloud Platform (NVIDIA Tesla T4)  
- Python 3.8  
- Pytorch-gpu with Cuda 11.3
