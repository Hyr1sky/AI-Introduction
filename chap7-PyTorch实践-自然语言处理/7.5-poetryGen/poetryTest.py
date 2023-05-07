import torch as t
import torch.nn as nn
import numpy as np


class Config(object):
    max_gen_len = 50  # 生成诗歌最长长度
    prefix_words = '白日依山尽，黄河入海流。'  # 不是诗歌的组成部分，用来控制生成诗歌的意境
    start_words = '深度学习'  # 诗歌开始


opt = Config


def getData(pklfile):
    data = np.load(pklfile, allow_pickle=True)
    data, word2id, id2word = data['data'], data['word2id'].item(), data['id2word'].item()
    return data, word2id, id2word



def gen_acrostic(model, start_words, ix2word, word2ix, prefix_words=None):
    """
    生成藏头诗
    start_words : u'深度学习'
    生成：
    深山通海月，度水入江流。学道无人识，习家无事心。
    """
    results = []
    start_word_len = len(start_words)
    input = (t.Tensor([word2ix['[START]']]).view(1, 1).long())
    input = input.to(opt.device)
    hidden = None

    index = 0  # 用来指示已经生成了多少句藏头诗
    # 上一个词
    pre_word = '[START]'

    if prefix_words:
        for word in prefix_words:
            output, hidden = model(input, hidden)
            input = (input.data.new([word2ix[word]])).view(1, 1)

    for i in range(opt.max_gen_len):
        output, hidden = model(input, hidden)
        top_index = output.data[0].topk(1)[1][0].item()
        w = ix2word[top_index]

        if (pre_word in {u'，', u'。', u'！', '[START]'}):
            # 如果遇到句号，藏头的词送进去生成

            if index == start_word_len:
                # 如果生成的诗歌已经包含全部藏头的词，则结束
                break
            else:
                # 把藏头的词作为输入送入模型
                w = start_words[index]
                index += 1
                input = (input.data.new([word2ix[w]])).view(1, 1)
        else:
            # 否则的话，把上一次预测是词作为下一个词输入
            input = (input.data.new([word2ix[w]])).view(1, 1)
        results.append(w)
        pre_word = w
    return results


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


def processWord(words, defword):
    if words.isprintable():
        new_words = words if words else defword
    else:
        new_words = words.encode('ascii', 'surrogateescape')\
            .decode('utf8') if words else defword
    return new_words.replace(',', u'，') \
        .replace('.', u'。') \
        .replace('?', u'？')


def gen(**kwargs):
    """
    提供命令行接口，用以生成相应的诗
    如：
    """

    for k, v in kwargs.items():
        setattr(opt, k, v)

    _, word2id, id2word = getData("poetry.npz")

    model = t.load("poetry_model.pth", map_location=lambda s, l: s)

    opt.device = t.device('cuda' if t.cuda.is_available() else 'cpu')
    model.to(opt.device)
    start_words = processWord( opt.start_words, defword='我')
    prefix_words = processWord( opt.prefix_words, defword=None)

    result = gen_acrostic(model, start_words, id2word, word2id, prefix_words)

    print(''.join(result))


if __name__ == '__main__':
    import fire
    fire.Fire()

