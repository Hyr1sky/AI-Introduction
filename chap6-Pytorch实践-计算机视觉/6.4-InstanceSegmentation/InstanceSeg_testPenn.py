import torchvision
import torch
import numpy as np
from PIL import Image, ImageDraw

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device ='cpu'
model=torch.load("./model.pt")
model.to(device)
model.eval()
transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

def drawLabel(origin_img, boxes, masks, indexes):  # 在原始图像的实例上画掩码和边框
    COLORS = [
        '#e6194b', '#3cb44b', '#ffe119', '#0082c8', '#f58231',
        '#911eb4', '#46f0f0', '#f032e6', '#d2f53c', '#fabebe',]
    draw = ImageDraw.Draw(origin_img)
    for i in indexes:
        mask = masks[i].astype('float')
        mask[mask >= 0.5] = 128  # 将掩码放大
        bitmap = Image.fromarray(mask.astype('int8'), mode='L')  # 转化为bitmap
        draw.bitmap((0, 0), bitmap=bitmap, fill=COLORS[i % 10])  # 将掩码附加到原图上

        box_loc = boxes[i]
        draw.rectangle(xy=box_loc, outline=COLORS[i % 10])  # 画框
    del draw
    return origin_img

def detection(img_path):
    # 读取输入图像，并转化为tensor
    origin_img = Image.open(img_path, mode='r').convert('RGB')
    img = transform(origin_img)
    img = img.to(device)

    # 将图像输入神经网络模型中，得到输出
    output = model(img.unsqueeze(0))

    scores = output[0]['scores'].cpu().detach().numpy()  # 预测每一个obj的得分
    bboxes = output[0]['boxes'].cpu().detach().numpy()  # 预测每一个obj的边框
    masks = output[0]['masks'].cpu().detach().numpy()

    # 这个我们只选取得分大于0.8的
    obj_index = np.argwhere(scores > 0.8).squeeze(axis=1).tolist()

    # 使用ImageDraw将检测到的边框和类别打印在图片中，得到最终的输出
    masks= np.squeeze(masks, axis=(1,))
    new_img = drawLabel(origin_img, bboxes, masks, obj_index)
    new_img.save("result-"+img_path)


if __name__ == '__main__':
    detection("test2.jpg")
