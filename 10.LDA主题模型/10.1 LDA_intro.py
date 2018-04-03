# -*- coding:utf-8 -*-

from gensim import corpora, models, similarities
from pprint import pprint

# 配置输出结果
# import logging
# logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


if __name__ == '__main__':

    # 9个英文文档数据，每一行即一个文档
    f = open('LDA_test.txt')
    # 创造停止词
    stop_list = set('for a of the and to in'.split())

    # 读取每一行文本数据，去掉两边空格，不考虑停止词，输出原文本分词
    # texts = [line.strip().split() for line in f]
    # pprint(texts)

    # 读取每一行文本数据，去掉两边空格，转小写，分开，考虑停止词，输出文本分词（除去停止词）
    texts = [[word for word in line.strip().lower().split() if word not in stop_list] for line in f]
    print('Text = ')
    pprint(texts)

    # 生成词典
    dictionary = corpora.Dictionary(texts)
    # 获取词典中词的个数
    V = len(dictionary)
    # 实际分词数据转换成词典向量语料库模型
    corpus = [dictionary.doc2bow(text) for text in texts]
    # 建立TF-IDF预料模型，此模型对原始词典向量语料加权处理了
    corpus_tfidf = models.TfidfModel(corpus)[corpus]

    # 输出原始词典向量语料库，非稀疏矩阵
    print('Initial Vector Data:')
    for c in corpus:
        print(c)

    # 输出TF-IDF
    print('TF-IDF:')
    for c in corpus_tfidf:
        print(c)

    # LSI模型（即LSA隐语义分析模型）
    print('\n---------------LSI Model---------------\n')

    # 设置语料库为TF-IDF，主题数为2，词典为dictionary
    lsi = models.LsiModel(corpus_tfidf, num_topics=2, id2word=dictionary)

    # 遍历获取所有文档语料的主题
    topic_result = [a for a in lsi[corpus_tfidf]]

    print('LSI Topics Result 文档的主题分布:')
    pprint(topic_result)

    print('LSI Topics Content 主题下的词分布（取前5个词相关度）:')
    pprint(lsi.print_topics(num_topics=2, num_words=5))

    # 根据主题计算文档间的相似度
    similarity = similarities.MatrixSimilarity(lsi[corpus_tfidf])
    print('LSI Similarity:')
    pprint(list(similarity))

    # LDA模型
    print('\n---------------LDA Model---------------:')

    # 指定主题个数
    num_topics = 2
    lda = models.LdaModel(corpus_tfidf, num_topics=num_topics, id2word=dictionary,
                          alpha='auto', eta='auto', minimum_probability=0.001)  # 主题小于0.001则忽略

    # 模型得到后，文档放进去，返回文档对应的主题
    doc_topic = [doc_t for doc_t in lda[corpus_tfidf]]
    print('LDA Topics Result 文档的主题分布:')
    pprint(doc_topic)

    # for doc_topic in lda.get_document_topics(corpus_tfidf):
    #     print(doc_topic)

    # 显示主题内部的词分布，即相关度
    for topic_id in range(num_topics):
        print('LDA Topics Content 主题下的词分布:', topic_id)
        # pprint(lda.get_topic_terms(topicid=topic_id))
        pprint(lda.show_topic(topic_id))

    # 根据主题计算文档间的相似度
    similarity = similarities.MatrixSimilarity(lda[corpus_tfidf])
    print('LDA Similarity:')
    pprint(list(similarity))

    # HDA模型
    print('\n---------------HDA Model---------------:')

    hda = models.HdpModel(corpus_tfidf, id2word=dictionary)

    # 获取HDA分析的每个文本的主题分布
    topic_result = [a for a in hda[corpus_tfidf]]

    print('HDA Topics Result 文档的主题分布:')
    pprint(topic_result)

    print('HDA Topics Content 主题下的词分布（取前5个词相关度）:')
    print(hda.print_topics(num_topics=2, num_words=5))
