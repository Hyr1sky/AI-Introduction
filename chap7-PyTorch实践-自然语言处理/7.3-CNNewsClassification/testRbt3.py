import torch
from torch.utils.data import DataLoader
from trainTextCNN import MyData, evaluate, labels
from sklearn import metrics  # 从sklearn工具包中导入指标评价模块

from transformers import AutoTokenizer, RobertaForSequenceClassification


def testModel(model_file, test_file):
    MODEL_NAME = 'hfl/rbt3'
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)  # 实例化分词器
    # 定义分词函数
    def tokenize_BERT(s):
        return tokenizer.encode(s, max_length=32, truncation=True, padding="max_length")

    # 得到数据集
    test_dataset = MyData(tokenize_fun=tokenize_BERT, filename=test_file)
    batch_size = 128
    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,  # 从数据集合中每次抽出batch_size个样本
        shuffle=False,  # 加载数据时不打乱样本顺序
    )
    model = torch.load(model_file, map_location=lambda s,l:s)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 设备
    model.to(device).eval()
    postProc = lambda x: x[0]
    acc, loss, allInfo = evaluate(model, device, test_dataloader, postProc)
    report = metrics.classification_report(allInfo[0], allInfo[1], target_names=labels, digits=4)
    confusion = metrics.confusion_matrix(allInfo[0], allInfo[1])
    msg = 'Test Loss: {0:>5.2},  Test Acc: {1:>6.2%}'
    print(msg.format(loss, acc))
    print("Precision, Recall and F1-Score...")
    print(report)
    print("Confusion Matrix...")
    print(confusion)

if __name__ == '__main__':
    testModel(model_file='checkpoint/rbt3_model.0.22', test_file='data/test.txt')
