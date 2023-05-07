import os
import json
import numpy as np
from gb2312_80 import gb2312_80
def getPeotryData(pklfile, path, maxlen):
    """
    读取path下的各个json文件
    将每首诗转换成长度为maxlen的字id序列
    存储到pklfile中
    """
    def _parseRawData(path, category):
        """
        读取path下的以category开头的各个json文件
        将每首诗转换成长度为maxlen的字id序列
        存储到列表后返回
        """
        data = []
        for filename in os.listdir(path):
            if filename.startswith(category):  # 文件开头必须为指定串
                data.extend(_handleJson(os.path.join(path, filename)))
        return data

    def _handleJson(file):
        """
        读取文件名为file的json文件
        将每首诗的正文存储到列表后返回
        """
        rst = []
        data = json.loads(open(file, 'r', encoding='utf-8').read())
        for poetry in data:
            pdata = poetry.get("paragraphs")  # 读取指定字段
            if pdata != "":
                rst.append(pdata)
        return rst

    def _padSequences(sequences, maxlen, value):
        """
        使用value填充到sequences中的每首诗（即字id序列）的前部
        使其长度均为maxlen
        """
        num_samples = len(sequences)
        x = (np.ones((num_samples, maxlen)) * value).astype('int32')
        for id, s in enumerate(sequences):
            trunc = np.asarray(s[:maxlen]).astype('int32')  # 取前maxlen个字符
            x[id, -len(trunc):] = trunc  # 将诗置于尾部
        return x

    # 得到文件名以'poet.tang'开头的json文件中的所有唐朝诗词内容
    data = _parseRawData(path=path, category='poet.tang')
    # 得到诗词中包含的左右的字符
    words = {_word for _sentence in data for _word in _sentence}
    # 得到每个字符和其编号(id)的对应字典
    word2id = {_word: _id for _id, _word in enumerate(words)}
    # 增加古诗中未出现的常用字，增加鲁棒性
    for s in gb2312_80:
        if s not in word2id.keys():
            word2id[s] = len(word2id)
    # 添加特殊字符
    word2id['[START]'] = len(word2id)  # 起始标识符
    word2id['[END]'] = len(word2id)  # 终止标识符
    word2id['[PAD]'] = len(word2id)  # 填充标识符
    # 得到每个字符编号(id)和字符的对应字典
    id2word = {_id: _word for _word, _id in list(word2id.items())}

    # 为每首诗歌加上起始符和终止符
    data = [["[START]"] + list(p) + ["[END]"] for p in data]

    # 将每首诗歌保存的内容由字符变成id
    # 形如[春,江,花,月,夜]变成[1,2,3,4,5]
    new_data = [[word2id[_word] for _word in _sentence]
                for _sentence in data]

    # 诗词长度不够maxlen的在前面补填充符，超过maxlen的诗词只保留前面字符。
    pad_data = _padSequences(new_data, maxlen=maxlen, value=word2id['[PAD]'])

    # 保存成二进制文件
    np.savez_compressed(pklfile,
                        data=pad_data,
                        word2id=word2id,
                        id2word=id2word)

# 执行
getPeotryData('poetry.npz', path='chinese-poetry/simplified', maxlen=125)