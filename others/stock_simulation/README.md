# インデックスの買い方を調べる


## 概要

- 前提
  - 環境
    - シード値を固定して、様々なシチュエーションで比較できるようにする。
  - 株価
    - 100円をスタートとする。
    - 標準検査は17.00とする。
    - 1ヶ月スパンで20年 → 240ヶ月をシュミレート。-> 変更可能
-------------------------------------------------------------------
  - basic_plan: 何も考えず同額を買い続ける
    - 初期投資5万
    - 継続投資4000円/月
    - 投資予算530000
  - percent_plan: 上がったら買う額減らし、下がったら増やす
    - 初期投資5万
    - 指定ロジック
    - 投資予算530000
  - down_plan: 下がった時のみ多めに買う
    - 初期投資5万
    - 指定ロジック
    - 投資予算530000

## 結果

- 下がった時に買っていく方が、パフォーマンスは高い。
- 下げ相場でも、`basic_plan`を大きく下回らないのも良し。
- 下げ相場はバーゲンなのかもしれない。

<br>

### 上昇トレンドの場合(年率5%)
---
<br>

![price](https://user-images.githubusercontent.com/88179125/187428216-68315467-efb5-4028-9b68-575c058c6cc5.png)


![simulate](https://user-images.githubusercontent.com/88179125/187428348-0f7c8791-2e84-4ac6-b29a-cb3592ee9092.png)

```
---basic_plan---
investment = 1396499.25
principal  = 950000
return     = 1.47
---percent_plan---
investment = 1851588.2
principal  = 950000.0
return     = 1.95
---down_plan---
investment = 1561902.6
principal  = 950000
return     = 1.64
```

### 下降トレンドの場合（年率-5%）
---
<br>

![simulate](https://user-images.githubusercontent.com/88179125/187429327-224515b2-0f4e-4ca2-b128-0bfbc16e83be.png)

![price](https://user-images.githubusercontent.com/88179125/187429347-8c9296aa-322b-4bf2-b6b4-384127ddec92.png)

```
---basic_plan---
investment = 447186.46
principal  = 950000
return     = 0.47
---percent_plan---
investment = 411400.41
principal  = 950000.0
return     = 0.43
---down_plan---
investment = 402789.06
principal  = 950000
return     = 0.42
```

