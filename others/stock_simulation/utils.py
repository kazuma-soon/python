# インデックスファンドをどのように買うべきかを考えてみる
# https://turtlechan.hatenablog.com/entry/2019/10/09/221556

# 戦略
# 5万円/月を基準とする
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
    
# 累計投資額を530000で固定する必要がある


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def bm(span, n, seed=0) -> np.ndarray:
    ''' 
    ブラウン運動をする関数 Brownian Motion
    parameters:
    -----------
    T:    実験期間(年)
    n:    実験期間を分割する数(月)
    seed: シード値
    T:n = 1:12
    '''
    np.random.seed(seed)
    dt = span / n
    brownian = np.random.standard_normal(n) * np.sqrt(dt)
    return brownian.cumsum()


def gbm(first_price=100, expected_return=0.05, sigma=0.18, span=10.0, n=121) -> np.ndarray:
    ''' 
    幾何ブラウン運動をする関数 Geometric Brownian Motion
    parameters:
    -----------
    '''
    brown_move = bm(span, n)

    data_x = np.linspace(0, span, n)
    trend = (expected_return - 0.8 * sigma**2) * data_x
    nonsense = sigma * brown_move

    gbm_ndarray = first_price * np.exp(trend + nonsense)
    return gbm_ndarray


