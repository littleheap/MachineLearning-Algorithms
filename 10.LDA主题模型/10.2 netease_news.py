# -*- coding:utf-8 -*-

import numpy as np
from gensim import corpora, models, similarities
from pprint import pprint
import time


# import logging
# logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


# 停止词加载函数
def load_stopword():
    f_stop = open('stopword.txt')
    sw = [line.strip() for line in f_stop]
    f_stop.close()
    return sw


if __name__ == '__main__':
    print('初始化停止词列表 --')

    # 记录开始时间
    t_start = time.time()

    # 获取停止词
    stop_words = load_stopword()

    print('开始读入语料数据 -- ')
    f = open('news.dat', encoding='utf-8')  # LDA_test.txt

    # 使用停止词分割
    texts = [[word for word in line.strip().lower().split() if word not in stop_words] for line in f]
    # texts = [line.strip().split() for line in f]
    print('读入语料数据完成，用时%.3f秒' % (time.time() - t_start))

    # 关闭文件流
    f.close()

    # 获取文本数量
    M = len(texts)
    print('文本数目：%d个' % M)
    # pprint(texts)

    print('正在建立词典 --')
    dictionary = corpora.Dictionary(texts)

    # 获取词典长度
    V = len(dictionary)
    print('词典中词的个数：', V)

    # 计算文本向量
    print('正在计算文本向量 --')
    corpus = [dictionary.doc2bow(text) for text in texts]

    # 计算TF-IDF
    print('正在计算文档TF-IDF --')
    t_start = time.time()
    corpus_tfidf = models.TfidfModel(corpus)[corpus]  # 喂养数据
    print('建立文档TF-IDF完成，用时%.3f秒' % (time.time() - t_start))

    print('LDA模型拟合推断 --')
    # 设置主题数目
    num_topics = 10

    t_start = time.time()

    # 构建LDA模型
    lda = models.LdaModel(corpus_tfidf, num_topics=num_topics, id2word=dictionary,
                          alpha=0.01, eta=0.01, minimum_probability=0.001,
                          update_every=1, chunksize=100, passes=1)
    print('LDA模型完成，训练时间为\t%.3f秒' % (time.time() - t_start))

    # 所有文档的主题分布
    # doc_topic = [a for a in lda[corpus_tfidf]]
    # print('Document-Topic:\n')
    # pprint(doc_topic)

    # 随机打印某10个文档的主题
    num_show_topic = 10  # 每个文档显示前几个主题
    print('10个文档的主题分布：')
    # 所有文档的主题分布
    doc_topics = lda.get_document_topics(corpus_tfidf)
    # 0~(M-1)的数组
    idx = np.arange(M)
    # 乱序
    np.random.shuffle(idx)
    # 取前十个不重复数字
    idx = idx[:10]
    for i in idx:
        # i号文档的主题分布
        topic = np.array(doc_topics[i])
        # 只获取当前主题分布的分布概率数据，即第二列，第一列为0~9序号忽略
        topic_distribute = np.array(topic[:, 1])
        # print(topic_distribute)
        # 主题分布排序
        topic_idx = topic_distribute.argsort()[:-num_show_topic - 1:-1]
        print(('第%d个文档的前%d个主题：' % (i, num_show_topic)), topic_idx)
        print(topic_distribute[topic_idx])

    num_show_term = 7  # 每个主题显示几个词
    print('每个主题的词分布：')
    for topic_id in range(num_topics):
        print('主题#%d：\t' % topic_id)
        # LDA第id号主题对应的词分布
        term_distribute_all = lda.get_topic_terms(topicid=topic_id)
        # 取前7个词
        term_distribute = term_distribute_all[:num_show_term]
        # 转换array形式
        term_distribute = np.array(term_distribute)
        term_id = term_distribute[:, 0].astype(np.int)
        print('词：\t', )
        for t in term_id:
            # 从词典中取出对应的词显示
            print(dictionary.id2token[t], )
        # print('\n概率：\t', term_distribute[:, 1])
