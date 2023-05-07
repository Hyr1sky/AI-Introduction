import re
import os
import json
from zh_wiki import zh2Hans


def convertJsonFiles(src, dest, category):
    """
    从src目录中以category开头的所有的json文件中，
    提取诗作关键信息，并转换为简体字，在dest目录中生成对应的新的json文件。

    关键信息包括：
    诗作唯一哈希编号：id，
    作者：author
    诗作题目：title
    诗作内容：paragraphs
    """

    def sentenceParse(para):
        # para 形如 "-181-村橋路不端，數里就迴湍。積壤連涇脉，高林上笋竿。早嘗甘蔗淡，
        # 生摘琵琶酸。（「琵琶」，嚴壽澄校《張祜詩集》云：疑「枇杷」之誤。）
        # 好是去塵俗，煙花長一欄。"
        result, number = re.subn(u"（.*）", "", para)
        result, number = re.subn(u"{.*}", "", result)
        result, number = re.subn(u"《.*》", "", result)
        result, number = re.subn(u"《.*》", "", result)
        result, number = re.subn(u"[\]\[]", "", result)
        r = ""
        for s in result:
            if s not in set('0123456789-'):
                r += zh2Hans.get(s, s)
        r, number = re.subn(u"。。", u"。", r)
        return r

    def handleJson(file):
        # print file
        rst = []
        data = json.loads(open(file, 'r', encoding='utf-8').read())
        for poetry in data:
            pdata = ""
            p = poetry.get("paragraphs")
            for sentence in p:
                pdata += sentence
            pdata = sentenceParse(pdata)
            if pdata != "":
                rst.append(pdata)
        return rst

    def convert(file, file2):
        # print file
        rst = []
        data = json.loads(open(file, 'r', encoding='utf-8').read())
        for poetry in data:
            pdata = ""
            p = poetry.get("paragraphs")
            for sentence in p:
                pdata += sentence
            pdata = sentenceParse(pdata)
            id = poetry.get("id")
            title = poetry.get("title")
            title = sentenceParse(title)
            author = poetry.get("author")
            author = sentenceParse(author)
            if pdata != "":
                rst.append({"id":id, "title":title, "paragraphs":pdata, "author":author})
        json.dump(rst, open(file2,'w', encoding='utf-8'),ensure_ascii=False,  indent=2)  # 生成新的json文件

    data = []
    for filename in os.listdir(src):
        if filename.startswith(category):
            data.extend(handleJson(src + filename))
            convert(src + filename, dest + filename)
    return data

convertJsonFiles(src='./json/', dest='./simplified/', category="poet.")