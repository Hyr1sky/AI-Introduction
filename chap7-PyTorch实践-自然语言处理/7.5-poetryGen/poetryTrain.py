import torch
import torch.nn as nn
import numpy as np
import tqdm


def setSeed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True  # 保证每次结果一样

setSeed(seed=1)


class PoetryModel(nn.Module):
    """
    诗词模型，核心层为双层单向LSTM，前面是嵌入层，后面是全连接层
    """

    def __init__(self, voc_size, emb_dim, hid_dim):
        """
        模型初始化：
        @voc_size: 此表大小
        @emb_dim: 词嵌入长度
        @hid_dim: RNN隐藏数量
        """
        super(PoetryModel, self).__init__()
        self.hid_dim = hid_dim  # 隐层数量
        self.embeddings = nn.Embedding(voc_size, emb_dim)  # 嵌入层
        self.rnn = nn.LSTM(emb_dim, self.hid_dim, num_layers=2)  # RNN层
        self.linear = nn.Linear(self.hid_dim, voc_size)  # 全连接层

    def forward(self, input, hidden=None):
        seq_len, batch_size = input.size()  # 得到句子长度L和每批次处理的句子数N
        if hidden is None:  # 初始化2层LSTM短时记忆隐状态和长时记忆隐状态, [2, N, H]
            h_0 = input.data.new(2, batch_size, self.hid_dim).fill_(0).float()
            c_0 = input.data.new(2, batch_size, self.hid_dim).fill_(0).float()
        else:
            h_0, c_0 = hidden
        # 经过嵌入层, [L, N]->[L, N, E], E:emb_dim
        embeds = self.embeddings(input)  #
        # 经过RNN层, [L, N, E]->[L, N, H], H:hid_dim
        output, hidden = self.rnn(embeds, (h_0, c_0))
        # 经过全连接层, [L*N, H]->[L*N, V], V:voc_size
        output = self.linear(output.view(seq_len * batch_size, -1))
        return output, hidden


def getData(pklfile):
    data = np.load(pklfile, allow_pickle=True)
    data, word2id, id2word = data['data'], data['word2id'].item(), data['id2word'].item()
    return data, word2id, id2word


def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 获取数据
    data, word2id, id2word = getData('poetry.npz')
    print(f"加载数据完成，共有{len(data)}首诗，含有{len(word2id)}种汉字")
    data = torch.from_numpy(data)
    epochs = 20  # 迭代此处
    lr = 1e-3  # 学习率
    batch_size = 256  # 批处理大小
    dataloader = torch.utils.data.DataLoader(data,
                                             batch_size=batch_size,
                                             shuffle=True,
                                             num_workers=1)

    # 模型实例化
    model = PoetryModel(voc_size=len(word2id), emb_dim=128, hid_dim=256)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    model.to(device)
    for epoch in range(epochs):
        loss_avg = 0
        for ii, data in tqdm.tqdm(enumerate(dataloader)):
            # 训练
            data = data.long().transpose(1, 0).contiguous()
            data = data.to(device)
            optimizer.zero_grad()
            input, target = data[:-1, :], data[1:, :].contiguous()
            output, _ = model(input)
            loss = criterion(output, target.view(-1).long())
            loss_avg += loss.item()
            loss.backward()
            optimizer.step()
        print("epoch:{}, loss:{:.2f}".format(epoch, loss_avg / len(data)))
        torch.save(model, 'poetry_model.pth')


if __name__ == '__main__':
    train()
