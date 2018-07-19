# -*-coding:utf-8-*-
# Project:  PJF
# Filename: feas_gen
# Date: 18-7-18
# Author: Smirk <smirk dot cao at gmail dot com>
import pandas as pd
import json
import os
import jieba


def load_data():
    filename_ = "../Input/features/pjf_data.pkl"
    if os.path.exists(filename_):
        df_ = pd.read_pickle(filename_)
    else:
        pwd = os.getcwd()
        target_dir = os.path.join(pwd, "../Input/job_desc/")
        f_lst = os.listdir(target_dir)

        df_ = pd.DataFrame()
        for f in f_lst:
            path_ = target_dir + f
            if os.path.isfile(path_):
                with open(path_) as f_:
                    raw_data_ = json.load(f_)
                    values_ = list(raw_data_.values())
                    cols_ = list(raw_data_.keys())
                    tmp_ = pd.DataFrame([values_], columns=cols_)
                    df_ = df_.append(tmp_, ignore_index=True)
        df_.to_pickle(filename_)
    return df_


def get_corpus(raw_data_):
    raw_corpus_ = list()
    # todo: 增加用户词典和停用词
    # todo: corpus 要和文件名对应上
    for data_ in raw_data_["jd_body"]:
        raw_corpus_.append(list(jieba.cut(data_, cut_all=False)))

    # 过滤单字符
    result_ = [" ".join([y for y in filter(lambda x: len(x) > 1, rst)]) for rst in raw_corpus_]
    # output
    filename_ = "../Input/corpus/corpus_pjf.data"
    with open(filename_, "w") as f:
        for jd_ in result_:
            f.write(jd_+"\n")
        f.close()
    return raw_corpus_


if __name__ == '__main__':
    df = load_data()
    corpus = get_corpus(df)
