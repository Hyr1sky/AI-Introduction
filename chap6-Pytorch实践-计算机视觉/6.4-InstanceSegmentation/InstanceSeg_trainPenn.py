import os
import numpy as np
import torch
from PIL import Image
import os
import numpy as np
import torch
from PIL import Image
from showDataset import getMasks
import sys

sys.path.append('detection')

class PennFudanDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms):
        self.root = root
        self.transforms = transforms  # 对图像做变换
        # 加载所有图像文件名和掩码文件名
        self.imgs = list(sorted(os.listdir(os.path.join(root, "PNGImages"))))
        self.masks = list(sorted(os.listdir(os.path.join(root, "PedMasks"))))

    def __getitem__(self, idx):
        # 得到第idx个图像文件名和掩码文件名
        img_path = os.path.join(self.root, "PNGImages", self.imgs[idx])
        mask_path = os.path.join(self.root, "PedMasks", self.masks[idx])
        img = Image.open(img_path).convert("RGB")  # 打开原始图像，并转换为RGB格式
        boxes, masks = getMasks(mask_path)  # 得到每个实例的标注信息
        # 转换为张量
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.ones((len(boxes),), dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)
        image_id = torch.tensor([idx])
        # 得到面积
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # 假设所有实例都不是拥挤人群
        iscrowd = torch.zeros((len(boxes),), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img = self.transforms(img)

        return img, target

    def __len__(self):
        return len(self.imgs)


import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import torchvision.transforms as T


def get_model_instance_segmentation(num_classes):
    # 加载预训练模型
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
    # 得到模型中分类器的输入特征数
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # 替换到预训练头（the pre-trained head）
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    # 得到模型中掩码分类器的输入特征数
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # 替换掉掩码预测器（the mask predictor）
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)
    return model  # 返回新模型


from detection import engine  # 从辅助文件中导入训练和评估函数
from detection import utils  # 导入辅助文件


def get_transform(train):  # 定义转换函数，训练和测试阶段转换过程有差别
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_classes = 2  # 数据集只有2类：背景和行人
    # 使用提前定义好的PennFudanDataset类和转换函数
    dataset = PennFudanDataset('PennFudanPed', get_transform(train=True))
    dataset_test = PennFudanDataset('PennFudanPed', get_transform(train=False))
    # 将数据集分割为训练集和验证集
    indices = torch.randperm(len(dataset)).tolist()
    dataset = torch.utils.data.Subset(dataset, indices[:-50])
    dataset_test = torch.utils.data.Subset(dataset_test, indices[-50:])
    # 定义训练数据集加载器和验证数据集加载器
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=2, shuffle=True, num_workers=4,
        collate_fn=utils.collate_fn)
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, shuffle=False, num_workers=4,
        collate_fn=utils.collate_fn)
    # 加载前面定义好的模型
    model = get_model_instance_segmentation(num_classes)
    model.to(device)
    # 定义优化器
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005,
                                momentum=0.9, weight_decay=0.0005)
    # 定义学习率调整策略
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.1)
    num_epochs = 10  # 训练过程迭代10次
    for epoch in range(num_epochs):
        # 1次迭代训练，每10个循环打印训练信息
        engine.train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
        lr_scheduler.step()  # 更新学习率
        engine.evaluate(model, data_loader_test, device=device)  # 在验证集上验证


    torch.save(model, "model.pt")  # 保存模型到文件
    print("Done, model.pt saved!")

if __name__ == "__main__":
    main()
