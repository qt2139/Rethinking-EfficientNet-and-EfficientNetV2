import os
import math
import argparse

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import torch.optim.lr_scheduler as lr_scheduler

from model import efficientnetv2_s as create_model
from my_dataset import MyDataSet
from utils import read_split_data, train_one_epoch, evaluate
import matplotlib.pyplot as plt
# 75s / epoch
# [train epoch 29] loss: 0.074, acc: 0.978: 100%|███████████████████████████████| 368/368 [01:15<00:00,  4.88it/s]
# [valid epoch 29] loss: 0.075, acc: 0.982: 100%|█████████████████████████████████| 92/92 [00:07<00:00, 11.65it/s]
def main(args):
    epo = []
    list1 = []
    list2 = []
    list3 = []
    list4 = []
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print(args)
    print('Start Tensorboard with "tensorboard --logdir=runs", view at http://localhost:6006/')
    tb_writer = SummaryWriter()
    if os.path.exists("./weights") is False:
        os.makedirs("./weights")

    train_images_path, train_images_label, val_images_path, val_images_label = read_split_data(args.data_path)

    img_size = {"s": [300, 384],  # train_size, val_size
                "m": [384, 480],
                "l": [384, 480]}
    num_model = "s"

    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(img_size[num_model][0]),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]),
        "val": transforms.Compose([transforms.Resize(img_size[num_model][1]),
                                   transforms.CenterCrop(img_size[num_model][1]),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])}

    # 实例化训练数据集
    train_dataset = MyDataSet(images_path=train_images_path,
                              images_class=train_images_label,
                              transform=data_transform["train"])

    # 实例化验证数据集
    val_dataset = MyDataSet(images_path=val_images_path,
                            images_class=val_images_label,
                            transform=data_transform["val"])

    batch_size = args.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               pin_memory=True,
                                               num_workers=nw,
                                               collate_fn=train_dataset.collate_fn)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=nw,
                                             collate_fn=val_dataset.collate_fn)

    # 如果存在预训练权重则载入
    model = create_model(num_classes=args.num_classes).to(device)
    if args.weights != "":
        if os.path.exists(args.weights):
            weights_dict = torch.load(args.weights, map_location=device)
            load_weights_dict = {k: v for k, v in weights_dict.items()
                                 if model.state_dict()[k].numel() == v.numel()}
            print(model.load_state_dict(load_weights_dict, strict=False))
        else:
            raise FileNotFoundError("not found weights file: {}".format(args.weights))

    # 是否冻结权重
    if args.freeze_layers:
        for name, para in model.named_parameters():
            # 除head外，其他权重全部冻结
            if "head" not in name:
                para.requires_grad_(False)
            else:
                print("training {}".format(name))

    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(pg, lr=args.lr, momentum=0.9, weight_decay=1E-4)
    # Scheduler https://arxiv.org/pdf/1812.01187.pdf
    lf = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - args.lrf) + args.lrf  # cosine
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    for epoch in range(args.epochs):
        # train
        train_loss, train_acc = train_one_epoch(model=model,
                                                optimizer=optimizer,
                                                data_loader=train_loader,
                                                device=device,
                                                epoch=epoch)
        list1.append(train_loss)
        list2.append(train_acc)
        scheduler.step()

        # validate
        val_loss, val_acc = evaluate(model=model,
                                     data_loader=val_loader,
                                     device=device,
                                     epoch=epoch)
        list3.append(val_loss)
        list4.append(val_acc)
        tags = ["train_loss", "train_acc", "val_loss", "val_acc", "learning_rate"]
        tb_writer.add_scalar(tags[0], train_loss, epoch)
        tb_writer.add_scalar(tags[1], train_acc, epoch)
        tb_writer.add_scalar(tags[2], val_loss, epoch)
        tb_writer.add_scalar(tags[3], val_acc, epoch)
        tb_writer.add_scalar(tags[4], optimizer.param_groups[0]["lr"], epoch)

        torch.save(model.state_dict(), "./weights/model-{}.pth".format(epoch))
        
        
    for epoch in range(args.epochs):
        epo.append(epoch)
    fig = plt.figure(figsize=(10,8))
    plt.plot(epo, list1, label = 'train loss')
    plt.title('Efficientnetv2_s Train Loss of Training from Scratch')
    plt.xlabel('epoch')
    plt.ylabel('train loss')
    plt.legend()
    plt.grid()
    plt.show()
    plt.savefig("my_efficientnetv2_s_train_loss.png")
    
    fig = plt.figure(figsize=(10,8))
    plt.plot(epo, list2, label = 'train accuracy')
    plt.title('Efficientnetv2_s Train Accuracy of Training from Scratch')
    plt.xlabel('epoch')
    plt.ylabel('train accuracy')
    plt.legend()
    plt.grid()
    plt.show()
    plt.savefig("my_efficientnetv2_s_train_accuracy.png")
    
    fig = plt.figure(figsize=(10,8))
    plt.plot(epo, list3, label = 'val loss')
    plt.title('Efficientnetv2_s Val Loss of Training from Scratch')
    plt.xlabel('epoch')
    plt.ylabel('val loss')
    plt.legend()
    plt.grid()
    plt.show()
    plt.savefig("my_efficientnetv2_s_val_loss.png")
    
    fig = plt.figure(figsize=(10,8))
    plt.plot(epo, list4, label = 'val accuracy')
    plt.title('Efficientnetv2_s Val Accuracy of Training from Scratch')
    plt.xlabel('epoch')
    plt.ylabel('val accuracy')
    plt.legend()
    plt.grid()
    plt.show()
    plt.savefig("my_efficientnetv2_s_val_accuracy.png")
    print('Finished Training')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--lrf', type=float, default=0.01)

    # https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz
    parser.add_argument('--data-path', type=str,
                        default='/home/ecbm4040/deep-learning-for-image-processing/data_set/flower_data/flower_photos')
    
    parser.add_argument('--weights', type=str, default='./pre_efficientnetv2-s.pth',
                        help='initial weights path')
    parser.add_argument('--freeze-layers', type=bool, default=False)
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')
    
    opt = parser.parse_args()

    main(opt)