# -*-coding:utf-8-*-
# Project:  PJF
# Filename: pjf
# Date: 18-7-18
# Author: Smirk <smirk dot cao at gmail dot com>
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import pairwise_distances
from sklearn.feature_extraction.text import TfidfTransformer
import pandas as pd


class JobRecommendation(object):
    def __init__(self,
                 corpus=None,
                 data=None):
        self.corpus_ = corpus
        self.data_ = data

    def load_corpus(self,
                    path_="../Input/corpus/corpus_pjf.data"):
        print(path_)
        with open(path_, "r") as f:
            self.corpus_ = f.readlines()

    def load_data(self):
        filename_ = "../Input/features/pjf_data.pkl"
        self.data_ = pd.read_pickle(filename_)

    def topn(self,
             id_=0,
             topn_=10,
             method_="tf"):
        # print("id_ %d topn %d in topn" % (id_, topn_))
        print(id_, self.data_.iloc[id_]["jd_body"])
        if method_ == "tf":
            x_tf = self._tf()
            rst = pairwise_distances(X=x_tf[0], Y=x_tf[1:])
            for idx in rst.T.argsort(axis=0)[::-1][:topn_]:
                print(idx, self.data_.iloc[idx[0]]["jd_body"])
        elif method_ == "tfidf":
            x_tfidf = self._tfidf()
            rst = pairwise_distances(X=x_tfidf[0], Y=x_tfidf[1:])
            for idx in rst.T.argsort(axis=0)[::-1][:topn_]:
                print(idx, self.data_.iloc[idx[0]][["jd_title", "jd_company", "jd_body"]].values)

    def _tf(self):
        vectorizer = CountVectorizer(lowercase=False, analyzer='word')
        x_tf_ = vectorizer.fit_transform(self.corpus_)
        print("x_tf_.shape is ", x_tf_.shape)
        return x_tf_

    def _tfidf(self):
        x_tf_ = self._tf()
        transformer = TfidfTransformer()
        x_tfidf_ = transformer.fit_transform(x_tf_)
        return x_tfidf_


if __name__ == '__main__':
    jr = JobRecommendation()
    print("#"*100)
    jr.load_corpus()
    jr.load_data()
    jr.topn(id_=3, method_="tfidf")
