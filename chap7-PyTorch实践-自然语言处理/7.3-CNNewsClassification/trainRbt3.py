from trainTextCNN import MyData, getDataLoader,  train, labels, setSeed
import torch
from transformers import AutoTokenizer, RobertaForSequenceClassification

setSeed(seed=1)


def trainRbt3():
    MODEL_NAME = 'hfl/rbt3'
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)  # 实例化分词器
    # 定义分词函数
    def tokenize_BERT(s):
        return tokenizer.encode(s, max_length=32, truncation=True, padding="max_length")

    # 得到数据集
    train_dataset = MyData(tokenize_fun=tokenize_BERT, filename='data/train.txt')
    dev_dataset = MyData(tokenize_fun=tokenize_BERT, filename='data/dev.txt')
    # 得到数据加载器
    train_dataset, dev_dataset = getDataLoader(train_dataset, dev_dataset)
    # 定义模型
    bertModel = RobertaForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=10, return_dict=False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 设备
    model = bertModel.to(device)
    # 定义后处理函数，因为预训练模型返回的是triple对象
    postProc = lambda x: x[0]
    lr = 1e-4  # 设置Adam优化器学习率
    train(model, device, 'rbt3', lr, train_dataset, dev_dataset, postProc=postProc)  # 开始训练

if __name__ == '__main__':
    trainRbt3()
    