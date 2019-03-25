#coding=utf-8
import os
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score

label_file="./data/weibo-remark/label/二分类标签.txt"#标签文件的存放位置
prediction_file="runs/1552742944/prediction.csv"#预测结果文件的存放位置
# prediction_file="machine_learning/validate_result.txt"#预测结果文件的存放位置

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
            list=line.split(',')
            value=int(float(list[-1].replace('\r\n','')))
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
    calculate_validation_corpus()