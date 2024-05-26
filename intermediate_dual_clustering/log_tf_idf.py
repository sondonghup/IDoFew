import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

def get_log_tf_idf(X, stop_words = ['instruction', 'output']):
    '''
    X : 대상 데이터
    stop_words : instruction, output을 지시어로 주기때문에 제거
    '''

    vectorizer = CountVectorizer(stop_words = stop_words)
    dtm = vectorizer.fit_transform(X)
    tf = dtm.toarray()
    ltf = 1 + np.log(tf + 1) # log tf 
    df = tf.astype(bool).sum(axis=0)
    idf = np.log((len(X)) / (df + 1))
    ltfidf = ltf * idf

    return ltfidf