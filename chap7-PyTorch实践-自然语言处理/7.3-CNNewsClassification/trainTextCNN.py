import os

from torch import nn
from torch.nn import functional as F
import numpy as np
import torch
def setSeed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

setSeed(seed=1)  # 设置随机种子，保证每次结果一样


class TextCNNModel(nn.Module):  # 定义模型
    def __init__(self, embedding_file='embedding_SougouNews.npz'):
        super(TextCNNModel, self).__init__()
        # 加载词向量文件
        embedding_pretrained = torch.tensor(
            np.load(embedding_file)["embeddings"].astype('float32'))
        # 定义词嵌入层
        self.embedding = nn.Embedding.from_pretrained(embedding_pretrained, freeze=False)
        # 定义三个卷积
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, 256, (k, 300)) for k in [2, 3, 4]])
        # 定义dropout层
        self.dropout = nn.Dropout(0.5)
        # 定义全连接层
        self.fc = nn.Linear(256 * 3, 10)

    def conv_and_pool(self, x, conv):  # 定义卷积+激活函数+池化层构成的一个操作块
        x = conv(x)  # N,1,32,300 -> N,256,31/30/29,1
        x = F.relu(x).squeeze(3)  # x -> N,256,31/30/29
        x = F.max_pool1d(x, x.size(2)).squeeze(2)  # x -> N,256,1 -> N,256
        return x

    def forward(self, x):  # 前向传播
        out = self.embedding(x)  # N,32 -> N,32,300
        out = out.unsqueeze(1)  # out -> N,1,32,300
        out = torch.cat([self.conv_and_pool(out, conv) for conv in self.convs], 1) # out ->N,768
        out = self.dropout(out)
        out = self.fc(out)  # N,768 -> N,10
        return out




from torch.utils.data import Dataset

labels = ['体育','娱乐','家居','教育','时政','游戏','社会','科技','股票','财经']
LABEL2ID = { x:i for (x,i) in zip(labels,range(len(labels)))}

class MyData(Dataset):  # 继承Dataset
    def __init__(self, tokenize_fun, filename):
        self.filename = filename  # 要加载的数据文件名
        self.tokenize_function = tokenize_fun  # 实例化时需传入分词器函数
        print("Loading dataset "+ self.filename +" ...")
        self.data, self.labels = self.load_data()  # 得到分词后的id序列和标签
    #读取文件，得到分词后的id序列和标签，返回的都是tensor类型的数据
    def load_data(self):
        labels = []
        data = []
        with open(self.filename, mode='r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                fields  = line.strip().split('\t')
                if len(fields)!=2 :
                    continue
                labels.append(LABEL2ID[fields[0]])  #标签转换为序号
                data.append(self.tokenize_function(fields[1]))  # 样本为词id序列
        f.close()
        return torch.tensor(data), torch.tensor(labels)
    def __len__(self):  # 返回整个数据集的大小
        return len(self.data)
    def __getitem__(self, index):  # 根据索引index返回dataset[index]
        return self.data[index], self.labels[index]


import pickle as pkl
vocab_file = "vocab.pkl"  # 之前生成的存储词及其id的文件
word_to_id = pkl.load( open(vocab_file, 'rb'))  #加载词典

def tokenize_textCNN(s):  # 输入句子s
    max_size=32  # 句子分词最大长度
    ts = [w for i, w in enumerate(s) if i < max_size]  # 得到字符列表，最多32个
    ids = [word_to_id[w] if w in word_to_id.keys() else word_to_id['[UNK]'] for w in ts]  # 根据词典，将字符列表转换为id列表
    ids += [0 for _ in range(max_size-len(ts))]  # 若id列表达不到最大长度，则补0
    return ids


from torch.utils.data import DataLoader
def getDataLoader(train_dataset, dev_dataset):
    batch_size = 128
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,  # 从数据集合中每次抽出batch_size个样本
        shuffle=True,  # 加载数据时打乱样本顺序
    )
    dev_dataloader = DataLoader(
        dataset=dev_dataset,
        batch_size=batch_size,
        shuffle=False,  # 按原始数据集样本顺序加载
    )
    return train_dataloader, dev_dataloader


from sklearn import metrics  # 从sklearn工具包中导入指标评价模块


def evaluate(model, device, dataload, postProc=None):
    model.eval()  # 设置模型为评估模式，该模式下不会更新模型参数
    loss_total = 0  # 总损失
    predict_all = np.array([], dtype=int)  # 存储所有样本的总预测类别
    labels_all = np.array([], dtype=int)  # 存储所有样本的总标注类别
    with torch.no_grad():
        for texts, labels in dataload:  # 循环加载每批验证数据
            outputs = model(texts.to(device))  # 将数据输入模型，得到输出结果
            if postProc:  # 可以对输出结果自定义后处理，得到实际的每样本各类别logits
                outputs =postProc(outputs)
            loss = F.cross_entropy(outputs, labels.to(device))  # 计算本批样本损失
            loss_total += loss  #  累加到总损失上
            labels = labels.data.cpu().numpy()  # 得到标注类别
            predic = torch.max(outputs.data, 1)[1].cpu().numpy()  # 得到预测类别
            labels_all = np.append(labels_all, labels)  # 添加到总标注类别
            predict_all = np.append(predict_all, predic)  # 添加到总预测类别
    # 得到在验证集上的精确率（accuracy）
    acc = metrics.accuracy_score(labels_all, predict_all)
    model.train()  # 恢复为训练模式
    # 返回精确率和平均loss值
    return acc, loss_total / len(dataload), (labels_all, predict_all)


from tensorboardX import SummaryWriter
import time
from datetime import timedelta

def get_time_dif(start_time):  # 获取已使用时间
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))

def train(model, device, model_name, lr, train_dataloader, dev_dataloader, postProc=None):
    start_time = time.time()  # 记录起始时间
    model.train()  # 设置model为训练模式
    optimizer = torch.optim.Adam(model.parameters(), lr=lr )  # 定义优化器
    total_batch = 0  # 记录进行到多少batch
    dev_best_loss = float('inf')  # 记录验证集上的最好损失
    last_improve = 0  # 记录上次验证集loss下降的batch数
    flag = False  # 记录是否很久没有效果提升
    for s in ['log', 'checkpoint']:
        os.makedirs(s, exist_ok=True)
    writer = SummaryWriter(log_dir='./log/%s.'%(model_name) + time.strftime('%m-%d_%H.%M', time.localtime()))  # 实例化SummaryWriter
    num_epochs = 3  # 设置训练次数
    for epoch in range(num_epochs):
        print('Epoch [{}/{}]'.format(epoch + 1, num_epochs))
        for i, (trains, labels) in enumerate(train_dataloader):
            outputs = model(trains.to(device))  # 将数据输入模型，得到输出结果
            if postProc:
                outputs =postProc(outputs) # 对输出结果后处理，得到每样本各类别logits
            model.zero_grad()  # 模型梯度清零
            loss = F.cross_entropy(outputs, labels.to(device))  # 计算交叉熵损失
            loss.backward()  # 梯度回传
            optimizer.step()  # 更新参数
            if total_batch % 100  == 0:
                # 每100轮输出在训练集和验证集上的效果
                true = labels.data.cpu()
                predic = torch.max(outputs.data, 1)[1].cpu()
                train_acc = metrics.accuracy_score(true, predic)  # 训练集精确度
                # 调用函数得到测试集精确度
                dev_acc, dev_loss, _ = evaluate(model, device, dev_dataloader, postProc)
                # 记录验证集当前的最优损失，并保存模型参数
                if dev_loss < dev_best_loss:
                    dev_best_loss = dev_loss
                    torch.save(model, 'checkpoint/%s_model.%.2f'%(model_name, dev_loss))
                    improve = '*'  # 设置标记
                    last_improve = total_batch
                else:
                    improve = ''
                time_dif = get_time_dif(start_time)  # 得到当前运行时间
                msg = 'Iter:{0:>4}, Tr-Loss:{1:>4.2}, Tr-Acc:{2:>6.2%}, Va-Loss:{3:>4.2}, Va-Acc:{4:>6.2%}, Time:{5}{6}'
                # 打印训练过程信息
                print(msg.format(total_batch, loss.item(), train_acc, dev_loss, dev_acc, time_dif, improve))
                # 写入tensorboardX可视化用的日志信息
                writer.add_scalar("loss/train", loss.item(), total_batch)
                writer.add_scalar("loss/dev", dev_loss, total_batch)
                writer.add_scalar("acc/train", train_acc, total_batch)
                writer.add_scalar("acc/dev", dev_acc, total_batch)
            total_batch += 1
            if total_batch - last_improve > 1000:
                # 验证集loss超过1000batch没下降，结束训练
                print("No optimization for a long time, auto-stopping...")
                flag = True
                break
        if flag:
            break
    writer.close()  # 关闭writer对象

def trainTextCNN():
    train_dataset = MyData(tokenize_fun=tokenize_textCNN, filename='data/train.txt')
    dev_dataset = MyData(tokenize_fun=tokenize_textCNN, filename='data/dev.txt')
    train_dataset, dev_dataset = getDataLoader(train_dataset, dev_dataset)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 设备
    model = TextCNNModel().to(device)
    print(model)
    lr = 1e-3  # Adam优化器学习率
    train(model, device, 'textcnn', lr, train_dataset, dev_dataset)  # 开始训练



if __name__ == '__main__':
    trainTextCNN()
    