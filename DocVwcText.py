
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

# rf = open('word2vec/new.txt', 'r', encoding='utf-8')
# wf = open('trainData.txt', 'a+', encoding='utf-8')
# lines = rf.readlines()
# for line in lines:
#     document = line.replace('<p>', '').replace('</p>', '').replace('\n', '').replace('，', '').replace('。',
#                                                                                                           '').replace(
#         '：', '').replace('？', '').replace(' ', '').replace('"summarization":', '').replace('"article"', '').replace('{', '').replace('}', '').replace(':', '').replace('<Paragraph>', '')
#     document_cut = jieba.cut(document, cut_all=False)
#     result = ' '.join(document_cut)
#     wf.write(result+'\n')


sentences = gensim.models.doc2vec.TaggedLineDocument('tempTrain.txt')
model = gensim.models.Doc2Vec(sentences, vector_size=256, window=2)
model.train(sentences, total_examples=model.corpus_count, epochs=model.epochs)
model.save('all_model.txt')


print(len(model.docvecs))
sims = model.docvecs.most_similar(0)
print(sims)

sims1 = model.docvecs.most_similar(14)
print(sims1)

sims2 = model.docvecs.most_similar(3)
print(sims2)