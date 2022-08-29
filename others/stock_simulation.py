# インデックスファンドをどのように買うべきかを考えてみる
# https://turtlechan.hatenablog.com/entry/2019/10/09/221556

# 戦略
# 株価が1％上がったら、買う額を1％下げる
# 株価が1％下がったら、買う額を1％上げる

# 前提
  # 環境
    # シード値を固定して、様々なシチュエーションで比較できるようにする。
  # 株価
    # 100円をスタートとする。
    # 標準検査は17.00とする。
    # 買値は全て終値とする。
    # 1ヶ月スパンで10年 → 120ヶ月をシュミレート。


#! /usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def bm(T, n, seed=0):
    ''' 
    ブラウン運動をする関数 Brownian Motion
    parameters
    ----------
    T:    実験期間
    n:    実験期間を分割する数
    seed: シード値
    '''
    np.random.seed(seed)
    dt = T / n
    brownian = np.random.standard_normal(n) * np.sqrt(dt)
    return brownian.cumsum()


if __name__ == '__main__':
    T = 1.0  # 期間
    n = 250  # 期間を分割する数

    
    plt.show(pd.Series(bm(T, n)).plot())
