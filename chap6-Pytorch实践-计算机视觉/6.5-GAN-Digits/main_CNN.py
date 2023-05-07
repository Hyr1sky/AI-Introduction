import torch
from torch import nn
from torchvision import transforms, datasets
from PIL import Image
import numpy as np
import os

class Generator(nn.Module):
    def __init__(self, input_size=10):
        super(Generator, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, 1024),
            nn.ReLU(True),
            nn.BatchNorm1d(1024),
            nn.Linear(1024, 7 * 7 * 128),
            nn.ReLU(True),
            nn.BatchNorm1d(7 * 7 * 128)
        )

        self.conv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(64),
            nn.ConvTranspose2d(64, 1, 4, 2, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.fc(x)
        x = x.view(x.shape[0], 128, 7, 7)  # reshape 通道是 128，大小是 7x7
        x = self.conv(x)
        return x

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 5, 1),
            nn.LeakyReLU(0.01),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 5, 1),
            nn.LeakyReLU(0.01),
            nn.MaxPool2d(2, 2)
        )
        self.fc = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.LeakyReLU(0.01),
            nn.Linear(1024, 1)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x

def gloss(scores_fake):
    loss = 0.5 * ((scores_fake - 1) ** 2).mean()
    return loss

def dloss(scores_real, scores_fake):
    loss = 0.5 * ((scores_real - 1) ** 2).mean() + 0.5 * (scores_fake ** 2).mean()
    return loss

def setKeyObj(generator, discriminator):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    loss = nn.BCELoss().to(device)  # 将损失函数置入合适的计算环境
    optim_G = torch.optim.Adam(generator.parameters(), lr=3e-4, betas=(0.5, 0.999))
    optim_D = torch.optim.Adam(discriminator.parameters(),lr=3e-4,betas=(0.5, 0.999))
    return device, loss, optim_G, optim_D


def getDataLoader(batch_size):
    Transform = transforms.Compose([transforms.Resize(28),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.5], [0.5])])
    data_loader = torch.utils.data.DataLoader(
            datasets.MNIST("./data", train=True, download=True, transform=Transform),
            batch_size=batch_size,
            shuffle=True
        )
    return data_loader

def saveImage(images, filename):  # 按5行5列保存25个图像到一个png文件
    bigImg = np.ones((150, 150))*255  # 先生成宽高均为150的全白大图数组
    for i in range(len(images)):
        row = int(i / 5) * 30  # 计算每个子图在大图中的左上角位置
        col = i % 5 * 30
        img = images[i]
        bigImg[col:col + 28, row:row + 28] = (1-img)*255/2  # 将子图放入大图中
    f = Image.fromarray(bigImg).convert('L')  # 将数组转换为8位灰度图
    f.save(filename, 'png')  # 保存文件




torch.manual_seed(0)  # 设置随机数种子
if __name__ == '__main__':
    batch_size = 4096  # 设置批处理大小
    EPOCH = 200  # 设置迭代次数
    input_size = 100  # 设置生成器输入随机向量维度
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    generator = Generator(input_size=input_size).to(device)  # 实例化生成器
    discriminator = Discriminator().to(device)  # 实例化鉴别器

    os.makedirs('imgs', exist_ok=True)  # 建立生成图像的存储目录

    device, loss, optim_G, optim_D = setKeyObj(generator, discriminator)
    data_loader = getDataLoader(batch_size)

    for epoch in range(EPOCH):
        for i, (real_images, _) in enumerate(data_loader):
            img_num = len(real_images)  # 得到一次处理的图像数量，一般为batch_size
            in_random = torch.rand((img_num, input_size)).to(device)  # 输入向量

            optim_G.zero_grad()  # 生成网络优化器梯度清零
            # 得到生成图像及其标签，针对生成器，标签为1表示让鉴别器认为是真实图像
            nfake_images = generator(in_random).to(device)
            nfake_labels = torch.ones(img_num).view((img_num, 1)).to(device)
            y3 = discriminator(nfake_images)  # 通过鉴别器得到判别结果

            G_loss = gloss(y3)  # 计算loss
            G_loss.backward()
            optim_G.step()

            optim_D.zero_grad()  # 鉴别网络优化器梯度清零
            # 设置真实图像和标签，针对鉴别器，标签为1表示是真实图像
            r_imgs = real_images.to(device)
            r_labels = torch.ones(img_num).view((img_num, 1)).to(device)
            # 设置生成图像和标签，针对鉴别器，标签为0表示是生成图像
            in_random = torch.rand((img_num, input_size)).to(device)
            f_imgs = generator(in_random).to(device)
            f_labels = torch.zeros(img_num).view((img_num, 1)).to(device)
            y1 = discriminator(r_imgs)  # 针对真实图像得到鉴别器输出
            r_loss = dloss(y1, r_labels)  # 计算loss
            y2 = discriminator(f_imgs)  # 针对生成图像得到鉴别器输出
            f_loss = dloss(y2, f_labels)  # 计算loss
            D_loss = (r_loss + f_loss) / 2  # 得到鉴别器总的loss
            D_loss.backward()
            optim_D.step()

            if i % 10 == 0:  # 每10个batch打印一下信息
                print("epoch: %d/%d, batch: %d/%d, D_loss: %f, G_loss: %f"
                      % (epoch, EPOCH, i, len(data_loader), D_loss.item(), G_loss.item())
                      )
        if epoch % 5 == 0:  # 每5个epoch保存一下生成图像
            saveImage(f_imgs[:25].detach().cpu().numpy(), "imgs/%d.png" % epoch)