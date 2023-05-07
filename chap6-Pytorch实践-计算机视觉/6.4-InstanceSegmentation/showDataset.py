import numpy as np
from PIL import Image, ImageDraw

def getMasks(mask_path):  # 获得标注信息
    mask = Image.open(mask_path)  # 打开标注图像
    mask = np.array(mask)  # 转换为Numpy数组，形状为(W,H)
    obj_ids = np.unique(mask)  # 得到编码数目，每个实例所有像素对应一个唯一编码
    obj_ids = obj_ids[1:]  # 第一个编码为背景，不代表任何实例，删除
    N = len(obj_ids)  # 得到实例数目N
    # 得到每个实例对应整个图像的掩码数组，masks形状为(N,W,H)
    masks = mask == obj_ids[:, None, None]  
    boxes = []  # 用于存储每个实例的边框
    for i in range(N):
        pos = np.where(masks[i])
        xmin = np.min(pos[1])
        xmax = np.max(pos[1])
        ymin = np.min(pos[0])
        ymax = np.max(pos[0])
        boxes.append([xmin, ymin, xmax, ymax])
    return boxes, masks

def drawLabel(origin_img, boxes, masks, indexes):  # 在原始图像的实例上画掩码和边框
    COLORS = [
        '#e6194b', '#3cb44b', '#ffe119', '#0082c8', '#f58231',
        '#911eb4', '#46f0f0', '#f032e6', '#d2f53c', '#fabebe',]
    draw = ImageDraw.Draw(origin_img)
    for i in indexes:
        mask = masks[i].astype('float')
        mask[mask >= 0.5] = 128  # 将掩码放大
        bitmap = Image.fromarray(mask.astype('int8'), mode='L')  # 转化为bitmap
        draw.bitmap((0, 0), bitmap=bitmap, fill=COLORS[i % 10])  # 原图上掩码画
        box_loc = boxes[i]
        draw.rectangle(xy=box_loc, outline=COLORS[i % 10])  # 画框
    del draw
    return origin_img

def showPic():
    img_path = "PennFudanPed\\PNGImages\\FudanPed00016.png"
    mask_path = "PennFudanPed\\PedMasks\\FudanPed00016_mask.png"
    origin_img = Image.open(img_path, mode='r').convert('RGB')
    boxes, masks = getMasks(mask_path)
    new_img = drawLabel(origin_img, boxes, masks, range(len(boxes)))
    new_img.save("result.png")
    new_img.show()

if __name__=='__main__':
    showPic()