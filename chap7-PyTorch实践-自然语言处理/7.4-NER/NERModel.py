
from copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from sklearn import metrics

# 双向BiLSTM模型
class BiLSTM(nn.Module):
    def __init__(self, vocab_size, emb_len, hidden_size, output_size):
        # 参数说明：vocab_size:字典的大小，emb_size:词向量的维数，hidden_size：隐向量的维数，out_size:标注的种类
        super(BiLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_len)  # 定义嵌入层
        self.bilstm = nn.LSTM(emb_len, hidden_size,  # 定义LSTM层,因为是双向LSTM，所以输出是2*hidden_size
                              batch_first=True,
                              bidirectional=True)
        self.bilstm.flatten_parameters()  # 为提高内存的利用率和效率
        self.linear = nn.Linear(2*hidden_size, output_size)  # 定义线性层

    def forward(self, sents_tensor, lengths):
        # 如何经过嵌入层，N表示句子数量，senMLen表示最大句子长度
        emb = self.embedding(sents_tensor)  # [N, senMLen] -> [N, senMLen, emb_size]
        # 根据lengths去掉每个句子的PAD，packed为PackedSequence类型
        # 其中主要的两个成员作用如下：
        # data成员为[所有句子词总数, senMLen]
        # batch_sizes成员存储了每批次进入bilstm的词嵌入向量数，共有senLen个批次
        packed = pack_padded_sequence(emb, lengths, batch_first=True)
        rnn_out, _ = self.bilstm(packed)
        # 将结果恢复为[N, senMLen, hidden_size*2]
        rnn_out, _ = pad_packed_sequence(rnn_out, batch_first=True)
        logits = self.linear(rnn_out)  # [N, senMLen, output_size]
        return logits







# 定义NER识别类，该类含有前面定义的双向BiLSTM模型实例
class NER_Model(object):
    def __init__(self, vocab_size, out_size):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        # 加载模型参数
        self.emb_size = 128
        self.hidden_size = 128
        self.model = BiLSTM(vocab_size, self.emb_size,
                                self.hidden_size, out_size).to(self.device)

        # 加载训练参数：
        self.epoches = 20
        self.print_step = 30
        self.lr = 1e-3
        self.batch_size = 64

        # 初始化优化器
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        # 初始化其他指标
        self.step = 0
        self._best_val_loss = 1e18
        self.best_model = None

    #定义损失函数
    def cal_loss_func(self, logits, targets, tag2id):
        # 参数logits: [N, senMLen, output_size]
        # 参数targets: [N, senMLen]
        # 参数tag2id: tag对应id的词典
        PAD = tag2id.get('[PAD]')
        mask = (targets != PAD)  # 得到掩码[N, senMLen], 句子含有词的位置为1，含有PAD的位置为0
        targets = targets[mask]  # 得到N个句子对应的实际tag列表（一维），设长度为lenAll
        out_size = logits.size(2)  # 得到类别数
        logits = logits.masked_select(
            mask.unsqueeze(2).expand(-1, -1, out_size)
        ).contiguous().view(-1, out_size)  # [lenAll, out_size]
        loss = F.cross_entropy(logits, targets) # 计算交叉熵损失
        return loss

    # 训练函数
    def train(self, word_lists, tag_lists,
              dev_word_lists, dev_tag_lists,
              word2id, tag2id):
        # 对数据集按照长度进行排序
        word_lists, tag_lists, _ = self.sort_by_lengths(word_lists, tag_lists)
        dev_word_lists, dev_tag_lists, _ = self.sort_by_lengths(
            dev_word_lists, dev_tag_lists)

        N = self.batch_size
        for e in range(1, self.epoches+1):
            self.step = 0
            losses = 0.
            for ind in range(0, len(word_lists), N):
                batch_sents = word_lists[ind:ind+N]
                batch_tags = tag_lists[ind:ind+N]
                # 训练一批次数据
                losses += self.train_step(batch_sents,
                                          batch_tags, word2id, tag2id)
                # 打印训练过程中的信息
                if self.step % self.print_step == 0:
                    total_step = (len(word_lists) // N + 1)
                    print("Epoch {}, step/total_step: {}/{} {:.2f}% Loss:{:.4f}".format(
                        e, self.step, total_step,
                        100. * self.step / total_step,
                        losses / self.print_step
                    ))
                    losses = 0.

            # 每轮训练结束后，测试在验证集上的性能，保存最好的模型
            val_loss, pred_tag_id_lists, tag_id_lists = self.validate(
                dev_word_lists, dev_tag_lists, word2id, tag2id)
            tag = ''
            if val_loss < self._best_val_loss:
                tag = '*'
                self.best_model = deepcopy(self.model)
                self._best_val_loss = val_loss
            # 计算精确度指标acc
            acc = metrics.accuracy_score(pred_tag_id_lists, tag_id_lists)
            print("Epoch {}, Val Loss:{:.4f} Val acc:{:.4f} {}".format(e, val_loss, acc, tag))

    # 将N个按长度倒排序后句子的词列表转换为相同长度的tensor
    def toTensor(self, batch, maps):
        PAD = maps.get('[PAD]')
        UNK = maps.get('[UNK]')
        senMLen = len(batch[0])  # 得到最大句子长度，第一个句子最长
        batch_size = len(batch)  # 得到句子数量
        # 初始化tensor：[N, senMLen]
        batch_tensor = torch.ones(batch_size, senMLen).long() * PAD
        # 每个词转换为id
        for i, l in enumerate(batch):
            for j, e in enumerate(l):
                batch_tensor[i][j] = maps.get(e, UNK)
        # 存储每个句子原始长度,lengths的长度为N
        lengths = [len(l) for l in batch]
        return batch_tensor, lengths

    # 将所有句子按长度倒排序，tag也应句子顺序对应
    def sort_by_lengths(self, word_lists, tag_lists):
        pairs = list(zip(word_lists, tag_lists))
        # 按句子长度由大到小对句子原始序号排序，indices存储倒排后句子的原始序号
        indices = sorted(range(len(pairs)),
                         key=lambda k: len(pairs[k][0]),
                         reverse=True)
        # 得到倒排序后的pairs
        pairs = [pairs[i] for i in indices]
        # 得到倒排序后的所有句子词列表和tag列表
        word_lists, tag_lists = list(zip(*pairs))
        return word_lists, tag_lists, indices

    # 对一个批次的数据进行训练
    def train_step(self, batch_sents, batch_tags, word2id, tag2id):
        self.model.train()
        self.step += 1
        # N个句子转为长度相同的tensor
        tensorized_sents, lengths = self.toTensor(batch_sents, word2id)
        tensorized_sents = tensorized_sents.to(self.device)
        # N个句子对应的tag转为长度相同的tensor
        targets, lengths = self.toTensor(batch_tags, tag2id)
        targets = targets.to(self.device)

        # forward
        logits = self.model(tensorized_sents, lengths)
        # 计算损失
        loss = self.cal_loss_func(logits, targets, tag2id).to(self.device)

        #更新参数
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    # 对验证集或测试集进行验证
    def validate(self, dev_word_lists, dev_tag_lists, word2id, tag2id):
        self.model.eval()
        self.model.bilstm.flatten_parameters()  # 为提高内存的利用率和效率
        with torch.no_grad():
            val_losses = 0.
            val_step = 0
            tag_id_lists = []
            pred_tag_id_lists = []
            #循环按每批次处理
            for ind in range(0, len(dev_word_lists), self.batch_size):
                val_step += 1
                # 准备当前批次N个句子数据
                batch_sents = dev_word_lists[ind:ind+self.batch_size]
                batch_tags = dev_tag_lists[ind:ind+self.batch_size]
                # N个句子转为长度相同的tensor
                tensorized_sents, lengths = self.toTensor(
                    batch_sents, word2id)
                tensorized_sents = tensorized_sents.to(self.device)
                # N个句子对应的tag转为长度相同的tensor
                targets, lengths = self.toTensor(batch_tags, tag2id)
                targets = targets.to(self.device)

                # forward
                logits = self.model(tensorized_sents, lengths)

                # 计算损失
                loss = self.cal_loss_func(
                    logits, targets, tag2id).to(self.device)
                val_losses += loss.item()

                # 得到预测的tag[N, senMLen], 带最后的PAD
                _, batch_tagids = torch.max(logits, dim=2)

                # 都去掉最后的PAD，转换为实际长度，变为一个list，用于计算metrics指标
                tag_id_list = self.batch2ListNoPad(targets, lengths)
                pred_tag_id_list = self.batch2ListNoPad(batch_tagids, lengths)

                tag_id_lists += tag_id_list
                pred_tag_id_lists += pred_tag_id_list

            val_loss = val_losses / val_step
        self.model.train()
        return val_loss, pred_tag_id_lists, tag_id_lists


    # 对于N个句子的预测结果，去掉最后的PAD，转换为实际长度，变为一个list，用于计算metrics指标
    def batch2ListNoPad(self, tagids, lengths):
        return [i.item() for j, sentence in enumerate(tagids) for k, i in enumerate(sentence)
                                     if k < lengths[j]]

    # 针对测试集进行模型测试
    def test(self, word_lists, tag_lists, word2id, tag2id):
        # 准备数据
        word_lists, tag_lists, _ = self.sort_by_lengths(word_lists, tag_lists)
        _, pred_tag_id_lists, tag_id_lists = self.validate(word_lists, tag_lists, word2id, tag2id)
        # 返回预测tag列表和真实tag列表，以便于计算评价指标
        return pred_tag_id_lists, tag_id_lists



def build_corpus(filename, make_vocab=True):
    # 得到lists每个元素和其id的对应关系词典
    def build_map(lists):
        maps = {}
        for list_ in lists:
            for e in list_:
                if e not in maps:
                    maps[e] = len(maps)
        return maps

    # 如果是训练集，make_vocab应为True
    word_lists = []
    tag_lists = []
    # 读取数据
    with open(filename, 'r', encoding='utf-8') as f:
        word_list = []
        tag_list = []
        for line in f:
            if line != '\n':
                word, tag = line.strip('\n').split()
                word_list.append(word)
                tag_list.append(tag)
            else:
                word_lists.append(word_list)
                tag_lists.append(tag_list)
                word_list = []
                tag_list = []

    # 如果是训练集，除了词列表和tag列表，还需要返回word2id和tag2id
    if make_vocab:
        word2id = build_map(word_lists)  # 得到词和其id的对应关系词典
        tag2id = build_map(tag_lists)  # 得到tag和其id的对应关系词典
        return word_lists, tag_lists, word2id, tag2id
    else:
        return word_lists, tag_lists

