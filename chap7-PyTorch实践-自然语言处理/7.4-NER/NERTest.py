
import pickle


from sklearn import metrics
from NERModel import BiLSTM, NER_Model, build_corpus

# LSTM模型训练的时候需要在word2id和tag2id加入PAD和UNK
def extend_maps(word2id, tag2id):
    word2id['[UNK]'] = len(word2id)
    word2id['[PAD]'] = len(word2id)
    tag2id['[UNK]'] = len(tag2id)
    tag2id['[PAD]'] = len(tag2id)
    return word2id, tag2id

# 从文件加载对象
def load_file(filename):
    return pickle.load(open(filename, "rb"))


# 对测试集进行测试
def bilstm_test(test_word_lists, test_tag_lists):
    # 加载model和两个词典
    ner_model = load_file("./ckpts/ner.pkl")
    bilstm_word2id, bilstm_tag2id = load_file("./ckpts/WordTag2id.pkl")

    # 得到预测结果和实际结果列表，以计算评价指标
    pred_tagids, real_tagids = ner_model.test(
        test_word_lists, test_tag_lists, bilstm_word2id, bilstm_tag2id)

    # 得到预测结果和实际结果中包含的所有实体标注类别名称
    names = [word for word, id in bilstm_tag2id.items() if id in real_tagids+pred_tagids]
    # 得到各类别性能指标
    report = metrics.classification_report(real_tagids, pred_tagids, target_names=names, digits=4, zero_division=0)
    print(report)

    # 得到实际结果中包含的所有实体标注类别
    test_names = [word for word, id in bilstm_tag2id.items() if id in real_tagids]
    # 打印测试集中未包含的实体标注类别
    print("tags not in dataset:", [w for w in bilstm_tag2id.keys() if w not in test_names and w not in ['[UNK]','[PAD]']])


def main():

    print("测试模型中...")
    # 加载测试数据
    test_word_lists, test_tag_lists = build_corpus("./ResumeNER/test.char.bmes", make_vocab=False)
    # 测试
    bilstm_test(test_word_lists, test_tag_lists)

if __name__ == "__main__":
    main()


