## https://bab2min.github.io/tomotopy/v0.12.1/en/#tomotopy.LDAModel
""" 다음 함수 사용을 위한 준비:
import sys
sys.path.append('C:/Users/82104/Desktop/pycharm/modules')
from md_topicmodeling import get_perplexity, topic_model"""

import tomotopy as tp
# print(tp.isa) # prints 'avx2', 'avx', 'sse2' or 'none'
import pyLDAvis
import numpy as np
import pandas as pd

def get_perplexity(docs=[""], to=31):
    """docs=["단어,단어,단어","단어,단어,단어","단어,단어,단어"]로 입력. to=포함할 토픽 갯수.
    perplexity와 coherence Dataframe으로 출력"""
    docs = [doc.split(",") if type(doc) == str else [] for doc in docs]
    per_coh=[]
    for i in range(2, to):
        mdl = tp.LDAModel(k=i, alpha=0.1, eta=0.01, min_cf=10, tw=tp.TermWeight.IDF)
        for doc in docs:
            mdl.add_doc(doc)   # 한줄씩 리스트로
        mdl.train(100)
        #print(str(i), "perplexity:", str(mdl.perplexity))
        cohs=[]
        for preset in ('u_mass', 'c_uci', 'c_npmi', 'c_v'):
            coh = tp.coherence.Coherence(mdl, coherence=preset)
            average_coherence = coh.get_score()
            coherence_per_topic = [coh.get_score(topic_id=k) for k in range(mdl.k)]
            #print(str(i), 'coherence_{}:'.format(preset), str(average_coherence))
            cohs.append(average_coherence)
            #print('Average:', average_coherence, '\nPer Topic:', coherence_per_topic)
        per_coh.append([i,mdl.perplexity]+cohs)
        result=pd.DataFrame(per_coh, columns=['topic_n','perplexity','u_mass','c_uci','c_npmi','c_v'])
    return result


def topic_model(docs=[""], k=10, model_save=False, model_html=False):
    """docs=["단어,단어,단어","단어,단어,단어","단어,단어,단어"]로 입력, k=선정된 토픽 갯수 입력,
    model_save=모델 저장하려면 영어 절대경로 입력(C://path/---.bin). model_html=모델 시각화 저장하려면 영어 절대경로 입력(C://path/---.html)
    summary, topic_term, doc_topic 출력. 모델 불러오기: tp.LDAModel.load()"""
    docs = [doc.split(",") if type(doc) == str else [""] for doc in docs]
    mdl = tp.LDAModel(k=k, alpha=0.1, eta=0.01, min_cf=10, tw=tp.TermWeight.IDF) # min_cf: 코퍼스에서 최소 언급 횟수.. 보통 5..
    for doc in docs:
        try:
            mdl.add_doc(doc)  # 한줄씩 입력해 mdl 만들기
        except:
            mdl.add_doc([""])
    mdl.burn_in = 100
    mdl.train(0)
    for i in range(0, 1000, 10):    # 100회 iteration
        mdl.train(10)
    summary=mdl.summary()
    if model_save:
        mdl.save(model_save)
    else:
        pass
    topic_term=pd.DataFrame([mdl.get_topic_words(k,top_n=30) for k in range(mdl.k)], \
                            index=["topic"+str(i) for i in range(k)], \
                            columns=["keyword"+str(i) for i in range(30)]).T
    docs_topic = []
    for i, doc in enumerate(mdl.docs):
        t=list(doc.get_topic_dist())
        t.append(t.index(max(t)))
        t.append(docs[i][0])
        docs_topic.append(t)
    doc_topic=pd.DataFrame(docs_topic, columns=["topic"+str(i) for i in range(k)]+["소속토픽","문서첫단어"])
    # 모델 시각화
    topic_term_dists = np.stack([mdl.get_topic_word_dist(k) for k in range(mdl.k)])
    doc_topic_dists = np.stack([doc.get_topic_dist() for doc in mdl.docs])
    doc_topic_dists /= doc_topic_dists.sum(axis=1, keepdims=True)
    doc_lengths = np.array([len(doc.words) for doc in mdl.docs])
    vocab = list(mdl.used_vocabs)
    term_frequency = mdl.used_vocab_freq
    prepared_data = pyLDAvis.prepare(
        topic_term_dists,
        doc_topic_dists,
        doc_lengths,
        vocab,
        term_frequency,
        start_index=0,  # tomotopy starts topic ids with 0, pyLDAvis with 1
        sort_topics=False)  # IMPORTANT: otherwise the topic_ids between pyLDAvis and tomotopy are not matching!
    if model_html:
        pyLDAvis.save_html(prepared_data, model_html)
    else:
        pass
    return [summary, topic_term, doc_topic]  # csv로 저장. excel 안됨.
