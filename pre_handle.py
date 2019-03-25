#coding=utf-8
from bs4 import BeautifulSoup as BF
import  sys
import re
import numpy as np
import glob
import xlwt

#定义一些参数
add_ori_coprpus="./data/weibo-remark/train/tagging_corpus"  #原语料库文件,未经处理的xml格式
shuffled_corpus="./data/weibo-remark/train/shuffled_corpus"  #经过随机打乱顺序后的语料库
add_after_pos_corpus="./data/weibo-remark/train/pos"  #处理后转换为txt格式的积极语料库文件
add_after_neg_corpus="./data/weibo-remark/train/neg"  #处理后转换为txt格式的消极语料库文件
corpus_post="./data/weibo-remark/test/corpus"#处理后的语料库,只包含有情感的微博评论数据
binary_label="./data/weibo-remark/label/二分类标签"#二分类标签文件
validate_corpus="./data/weibo-remark/test/corpus_validation"#验证集,用于模型准确性的比对和标签文件的生成
def get_soup(corpus):#获取爬去xml文件的Beautifulsoup
    original_corpus = open('%s.xml' % corpus, "r")
    soup = BF(original_corpus, 'lxml')
    return soup


def load_xml_corpus(corpus):#读取xml文件中的内容到内存中
    soup=get_soup(corpus)
    sentence_list = soup.find_all("sentence")
    print(len(sentence_list))

    return sentence_list


def write_to_txt_corpus(content_train):#将内存中的数据写入txt文件,测试集中的部分数据
    pos=open("%s.txt"%add_after_pos_corpus,"w+")
    neg=open("%s.txt"%add_after_neg_corpus,"w+")
    corpus=open("%s.txt"%corpus_post,"w+")
    for i in range(len(content_train)):
        print (content_train[i])
        text = content_train[i].text
        if (content_train[i]['opinionated'] is 'Y'):
            # 将愤怒,厌恶,恐惧,悲伤定义为消极情绪
            if(content_train[i]['emotion-1-type']=='愤怒' or content_train[i]['emotion-1-type']=='厌恶'
            or content_train[i]['emotion-1-type']=='恐惧' or content_train[i]['emotion-1-type']=='悲伤'
            or content_train[i]['emotion-1-type']=='anger' or content_train[i]['emotion-1-type']=='disgust'
            or content_train[i]['emotion-1-type'] == 'fear' or content_train[i]['emotion-1-type']=='sadness'):
                neg.write(text+'\n')
                corpus.write(text+'\n')
            # 将高兴,喜好,惊讶定义为积极情绪
            elif(content_train[i]['emotion-1-type']=='高兴' or content_train[i]['emotion-1-type']=='喜好'
            or content_train[i]['emotion-1-type']=='惊讶' or content_train[i]['emotion-1-type']=='happiness'
            or content_train[i]['emotion-1-type'] == 'like' or content_train[i]['emotion-1-type']=='surprise'):
                pos.write(text+'\n')
                corpus.write(text+'\n')

def write_to_txt_corpus2(content_dev):#获取标签文件
    corpus = open("%s.txt" % binary_label, "w+")
    validate = open("%s.txt" % validate_corpus, "w+")
    for i in range(len(content_dev)):
        print (content_dev[i])
        text = content_dev[i].text
        if (content_dev[i]['opinionated'] is 'Y'):
            # 将愤怒,厌恶,恐惧,悲伤定义为消极情绪
            if (content_dev[i]['emotion-1-type'] == '愤怒' or content_dev[i]['emotion-1-type'] == '厌恶'
            or content_dev[i]['emotion-1-type'] == '恐惧' or content_dev[i]['emotion-1-type'] == '悲伤'
            or content_dev[i]['emotion-1-type'] == 'anger' or content_dev[i]['emotion-1-type'] == 'disgust'
            or content_dev[i]['emotion-1-type'] == 'fear' or content_dev[i]['emotion-1-type'] == 'sadness'):
                validate.write(text + '\n')
                corpus.write('0.0\n')

            # 将高兴,喜好,惊讶定义为积极情绪
            elif (content_dev[i]['emotion-1-type'] == '高兴' or content_dev[i]['emotion-1-type'] == '喜好'
            or content_dev[i]['emotion-1-type'] == '惊讶' or content_dev[i]['emotion-1-type'] == 'happiness'
            or content_dev[i]['emotion-1-type'] == 'like' or content_dev[i]['emotion-1-type'] == 'surprise'):
                validate.write(text + '\n')
                corpus.write('1.0\n')


def split_data():
    # Shuffle data randomly
    content = load_xml_corpus(add_ori_coprpus)
    array=[]
    for i in range(len(content)):
        array.append(content[i])
    content_shuffled = np.random.permutation(array)
    dev_sample_index = -1 * int(0.1 * float(len(content_shuffled)))
    content_train, content_dev = content_shuffled[:dev_sample_index], content_shuffled[dev_sample_index:]
    print("Train/Dev split: {:d}/{:d}".format(len(content_train), len(content_dev)))
    write_to_txt_corpus(content_train)
    write_to_txt_corpus2(content_dev)

reload(sys)
sys.setdefaultencoding('utf-8')
# write_to_txt_corpus()
# write_to_txt_corpus2()
# write_to_txt_corpus3()

split_data()





