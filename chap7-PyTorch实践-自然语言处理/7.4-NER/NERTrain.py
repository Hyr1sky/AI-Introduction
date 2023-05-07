import os
import pickle
import time
import numpy as np
import torch

from NERModel import NER_Model, build_corpus

def setSeed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True  # 保证每次结果一样
setSeed(1)


# LSTM模型训练的时候需要在word2id和tag2id加入PAD和UNK
def extend_maps(word2id, tag2id):
    word2id['[UNK]'] = len(word2id)
    word2id['[PAD]'] = len(word2id)
    tag2id['[UNK]'] = len(tag2id)
    tag2id['[PAD]'] = len(tag2id)
    return word2id, tag2id

# 持久化对象
def save_file(model, filename):
    pickle.dump(model, open(filename, "wb"))


# 训练和验证
def bilstm_train_and_eval(train_data, dev_data, word2id, tag2id):
    train_word_lists, train_tag_lists = train_data
    dev_word_lists, dev_tag_lists = dev_data

    start = time.time()
    vocab_size = len(word2id)
    out_size = len(tag2id)
    bilstm_model = NER_Model(vocab_size, out_size)
    bilstm_model.train(train_word_lists, train_tag_lists,
                       dev_word_lists, dev_tag_lists, word2id, tag2id)

    save_file(bilstm_model, "./ckpts/ner.pkl")

    print("训练完毕,共用时{}秒.".format(int(time.time()-start)))



def main():
    # 读取数据
    print("读取数据...")
    train_word_lists, train_tag_lists, word2id, tag2id = \
        build_corpus("./ResumeNER/train.char.bmes")
    dev_word_lists, dev_tag_lists = build_corpus("./ResumeNER/dev.char.bmes", make_vocab=False)
    # LSTM模型训练的时候需要在word2id和tag2id加入PAD和UNK
    extend_maps(word2id, tag2id)
    # 保存文件，用于部署测试
    os.makedirs("./ckpts", exist_ok=True)  # 若目录不存在则先创建
    save_file([word2id, tag2id],"./ckpts/WordTag2id.pkl")

    # 训练评估BI-LSTM模型
    print("正在训练评估NER模型...")
    bilstm_train_and_eval(
        (train_word_lists, train_tag_lists),
        (dev_word_lists, dev_tag_lists),
        word2id, tag2id,
    )



if __name__ == "__main__":
    main()


