# encoding:utf-8

import pandas as pd
import jieba
from collections import Counter
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.linear_model import LogisticRegression


pos_corpus="/home/Snake/PycharmProjects/chi_weibo_classify/data/weibo-remark/train/pos.txt"#正向情感训练语料库
neg_corpus="/home/Snake/PycharmProjects/chi_weibo_classify/data/weibo-remark/train/neg.txt"#负向情感训练语料库
validate_corpus="/home/Snake/PycharmProjects/chi_weibo_classify/data/weibo-remark/test/corpus_validation.txt"#验证集语料库
validate_result="validate_result.txt"#分类结果保存位置
label_file="/home/Snake/PycharmProjects/chi_weibo_classify/data/weibo-remark/label/二分类标签.txt"#标签文件的存放位置
prediction_file="validate_result.txt"#预测结果文件的存放位置

# 获取数据和标记
def load_data():
    data = pd.read_table(neg_corpus, header=None, sep='\n')
    data2 = pd.read_table(pos_corpus, header=None, sep='\n')
    posting_list = []
    class_list = [] # 方便计算转换为1,2,3

    for i in range(len(data)):
        # posting_list.append((data.iloc[i, 1]))
        posting_list.append((str(data.iloc[i])).strip())
        class_list.append(str(0))
    for i in range(len(data2)):
        posting_list.append((str(data2.iloc[i])).strip())
        class_list.append(str(1))

    print (class_list)
    return posting_list, class_list


def jieba_tokenizer(x): return jieba.cut(x, cut_all=True)


def get_classify():
    X_train, Y_train = load_data()

    # 定义分类器
    classifier = Pipeline([
        ('counter', CountVectorizer(tokenizer=jieba_tokenizer)),  # 标记和计数，提取特征用 向量化
        ('tfidf', TfidfTransformer()),                            # IF-IDF 权重
        ('clf', OneVsRestClassifier(LinearSVC())),  # 1-rest 多分类(多标签)
    ])
    mlb = MultiLabelBinarizer()
    Y_train = mlb.fit_transform(Y_train)                          # 分类号数值化

    classifier.fit(X_train, Y_train)

    corpus_list=[]
    for line in open(validate_corpus,'r'):
        if line.strip():
            corpus_list.append(line)

    prediction = classifier.predict(corpus_list)
    result = mlb.inverse_transform(prediction)

    f = open(validate_result, 'w')
    print (len(corpus_list))
    print(len(result))
    for i in range(len(corpus_list)):
        f.write(str(corpus_list[i].replace("\n","")))
        print (result[i])
        f.write(str(result[i][0]) + '\n')

    print (result, len(result))
    num_dict = Counter(result)
    print (len(num_dict))
    # print ((num_dict[('1',)] + num_dict[('2',)] + num_dict[('3',)]) / float(len(result)))  # 整数除整数为0，应把其中一个改为浮点数。
    print ((num_dict[('0',)] + num_dict[('1',)]) / float(len(result)))  # 整数除整数为0，应把其中一个改为浮点数。


def calculate_validation_corpus():
    y_true=[]
    y_pred=[]
    for line in open(label_file, 'r'):
        if line.strip():
            value=float(line.strip())
            value=int(value)
            y_true.append(value)

    for line in open(prediction_file,'r'):
        if line.strip():
            value=line[-2:-1]
            value=int(value)
            y_pred.append(value)

    print (y_true)
    print (y_pred)
    accuracy=accuracy_score(y_true, y_pred)
    recall=recall_score(y_true, y_pred, average='binary')
    F_value=2*accuracy*recall/(accuracy+recall)
    print("准确率:"+str(accuracy))
    print("召回率:"+str(recall))
    print("F值:"+str(F_value))
if __name__ == '__main__':
    get_classify()
    calculate_validation_corpus()
