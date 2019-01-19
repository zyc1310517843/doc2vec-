
import os

import gensim
import jieba

# path = "word2vec/doc"
# files = os.listdir(path)
# wf = open('trainData.txt', 'a+', encoding='utf-8')
# count = 0
# for file in files:
#     tempFile = open(path+'/'+file, 'r', encoding='utf-8')
#     document = tempFile.read()
#     document = document.replace('<p>', '').replace('</p>', '').replace('\n', '').replace('，', '').replace('。', '').replace('：', '').replace('？', '').replace(' ', '')
#     document_cut = jieba.cut(document, cut_all=False)
#     result = ' '.join(document_cut)
#     count = count+1
#     wf.write(result+'\n')
# wf.close()

rf = open('new.txt', 'r', encoding='utf-8')  # 自己的需要训练的文档集合，一行一个文章或者句子
wf = open('trainData.txt', 'a+', encoding='utf-8')
lines = rf.readlines()
for line in lines:
    document = line.replace('<p>', '').replace('</p>', '').replace('\n', '').replace('，', '').replace('。',
                                                                                                          '').replace(
        '：', '').replace('？', '').replace(' ', '').replace('"summarization":', '').replace('"article"', '').replace('{', '').replace('}', '').replace(':', '').replace('<Paragraph>', '')
    document_cut = jieba.cut(document, cut_all=False)
    result = ' '.join(document_cut)
    wf.write(result+'\n')


sentences = gensim.models.doc2vec.TaggedLineDocument('tempTrain.txt')  #  训练
model = gensim.models.Doc2Vec(sentences, vector_size=256, window=2)
model.train(sentences, total_examples=model.corpus_count, epochs=model.epochs)
model.save('all_model.txt')


print(len(model.docvecs))  #  所有文章的向量
sims = model.docvecs.most_similar(0)  #输出和训练集中第一个文章最相近的文章
print(sims)

sims1 = model.docvecs.most_similar(14)
print(sims1)

sims2 = model.docvecs.most_similar(3)
print(sims2)
