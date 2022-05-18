""" 다음 함수 사용을 위한 준비:
import sys
sys.path.append('C:/Users/82104/Desktop/pycharm/modules')
from md_token_w2v_d2v import doc2sent, okt_tokenizer, w2v_create, d2v_create"""

from gensim.models import Word2Vec, KeyedVectors
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import sent_tokenize
import kss
from konlpy.tag import Okt
import re

def doc2sent(docs=None):
    '''docs =["문장1-1. 문장1-2", "문장2-1. 문장2-2", "문장3-1. 문장3-2"]
    출력: [[doc_i, sent_i, sent], [doc_i, sent_i, sent], [doc_i, sent_i, sent]]'''
    doc_sent=[]
    if re.findall(r"[가-힣]", docs[0]):
        for doc_i, doc in enumerate(docs):
            try:
                doc=[d.strip() for d in kss.split_sentences(doc) if d.endswith(".")]
                for sent_i, sent in enumerate(doc):
                    doc_sent.append([doc_i, sent_i, sent])
            except:
                doc_sent.append([doc_i, sent_i, ""])
    else:
        for doc_i, doc in enumerate(docs):
            try:
                doc = [d.strip() for d in sent_tokenize(doc) if d.endswith(".")]
                for sent_i, sent in enumerate(doc):
                    doc_sent.append([doc_i, sent_i, sent])
            except:
                doc_sent.append([doc_i, sent_i, ""])
    return doc_sent

'''stopwords = "라고,news,News,기자,논설위원,해설위원,사진,특파원,연합뉴스,단독,오늘,뉴스,데스크,앵커,종합,1보,2보,3보,상보,생방송,종합2보,뉴스검색,통합검색,검색,네이버,다음,개월,가지,기준,\
             뉴스룸,무단,전재,배포,금지,원본,연합,뉴시스,인턴,가운데,이날,이분,중요,한편,이번,지난달,뉴스1,관계자,오전,오후,인근,시간,신문,이후,이전,해당,답변,질문,인터뷰,\
             속보,현장,앵커리포트,리포트,르포,뉴스,뉴스피처,피처,인터뷰,특징주,만평,팩트,팩트체크,동안,닷컴".split(",")'''

def okt_tokenizer(text=None, nouns=True, stopwords=[]):
    ''' text=문자열, 출력: "형태소,형태소,형태소"  '''
    try:
        t1= re.sub(r"[^가-힣a-zA-Z]", " ", text)       # re.sub: 문자열 부분 교체. r은 정규표현식 사용한다는 표시. "[^가-힣a-zA-Z1-9]"는 한글 영어 숫자 이외의 문자열 의미.
        if nouns:
            t2 = Okt().nouns(t1)    # 명사 추출 리스트화
        else:
            t2 = Okt().pos(t1, norm=True, stem=True)
            pos_use = ['Noun', 'Verb', 'Adjective', 'Adverb', 'Determiner', 'Alpha', 'Foreign']
            t2=[word for word, pos in t2 if pos in pos_use]
        t3 = [ti for ti in t2 if len(ti)>1 ]          # 2음절 이상의 명사만 선택
        t4 = [ti for ti in t3 if ti not in stopwords] # stopwords에 포함된 단어 삭제
        return ",".join(t4)
    except:
        return ""

def w2v_create(docs=None, okt_nouns=True, stopwords=[], vector_size=100, window=5, min_count=1):
    ''' w2v_create(docs=["문장1-1. 문장1-2", "문장2-1. 문장2-2", "문장3-1. 문장3-2"], vector_size=100, window=5, min_count=1, model_path=None)
    출력: [w2v_model, w_dict, w_vectors]'''
    try:
        doc_sent = doc2sent(docs)
        doc_token = [okt_tokenizer(sent, nouns=okt_nouns, stopwords=stopwords).split(",") if type(sent) == str else "" for
                 doc_i, sent_i, sent in doc_sent]
        w2v_model = Word2Vec(sentences=doc_token, vector_size=vector_size, min_count=min_count, window=window, workers=4)
        w_dict = w2v_model.wv.key_to_index # 전체 word 리스트. 단어 사전.
        w_vectors = w2v_model.wv.vectors  # 전체 word의 vectors
        # w2v_model.wv['찾을 단어']   # 개별 word의 vector
        # w2v_model.wv.most_similar('찾을 단어', topn=10)  # 유사 단어
    except:
        print("출력 불가")
    return [w2v_model, w_dict, w_vectors]

def d2v_create(docs=None, okt_nouns=True, stopwords=[], vector_size=100, window=5, min_count=1):
    ''' w2v_create(docs=["문장1-1. 문장1-2", "문장2-1. 문장2-2", "문장3-1. 문장3-2"], vector_size=100, window=5, min_count=1, model_path=None)
    출력: [d2v_model, d_vectors, word_dct]'''
    try:
        doc_sent=doc2sent(docs)
        doc_token=[(doc_i, okt_tokenizer(sent, nouns=okt_nouns, stopwords=stopwords).split(",")) if type(sent)==str else "" for doc_i, sent_i, sent in doc_sent]
        tagged_docs = [TaggedDocument(words, [tag]) for tag, words in doc_token]
        d2v_model = Doc2Vec(tagged_docs, vector_size=vector_size, window=window, min_count=min_count, workers=4, dm=1) #, passes=15) #  If dm=1, ‘distributed memory’ (PV-DM) is used. Otherwise, distributed bag of words (PV-DBOW) is employed.
        d_vectors=[(tag, d2v_model.dv[tag]) for tag in range(len(docs))]
        word_dct=d2v_model.wv.key_to_index # 사용된 단어 사전
        # d2v_model.dv.similarity('tag1', 'tag2')  유사성 분석
    except:
        print("출력 불가")
    return [d2v_model, d_vectors, word_dct]
