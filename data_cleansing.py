import numpy as np
import pandas as pd
import re
from collections import defaultdict
from collections import Counter
import csv


def pre_training():
    time_data = "2017/4/1_22:15"
    # 一度に複数の区切り文字でsplitできる
    ts = re.split("[/_:]", time_data)
    print(ts) # ['2017', '4', '1', '22', '15']

    # 配列time_list
    time_list = ["2006/11/26_2:40", "2009/1/16_23:35", "2014/5/4_14:26", "2017/8/9_7:5", "2017/4/1_22:15"]
    # 文字列から"時"を取り出す関数を作成して下さい
    get_time = lambda x: int(re.split("[/_:]", x)[3])
    # 上で作った関数を用いて各要素の"時"を取り出し、配列にして下さい
    hour_list = list(map(get_time, time_list))


    # 出力して下さい
    print(hour_list)

    # 文字列の"月"が条件を満たすときにTrueを返す関数を作成して下さい
    is_first_half = lambda x: 1 <= int(re.split("[/_:]", x)[1]) <= 6 
    # is_first_half = lambda x: int(re.split("[/_:]", x)[1]) - 7 < 0

    # 上で作った関数を用いて条件を満たす要素を抜き出し、配列にして下さい
    res = list(filter(is_first_half, time_list))
    # 出力して下さい
    print(res)

    # 時間データhour, 分データminute
    hour = [0, 2, 3, 1, 0, 1, 1]
    minute = [30, 35, 0, 14, 11, 0, 22]

    # 時, 分を引数に、分に換算する関数を作成して下さい
    to_m = lambda x, y : x * 60 + y

    # リスト内包表記を用いて所定の配列を作成してください
    res = [to_m(h, m) for h, m in zip(hour, minute)]

    # 出力して下さい
    print(res)

    a = [i for i in range(1, 10)]
    b = [i for i in range(10, 20)]
    print([[x, y] for x in a for y in b])

    # 二進数の桁
    fours_place = [0, 1]
    twos_place  = [0, 1]
    ones_place  = [0, 1]

    # リスト内包表記の多重ループを用いて0から7までの整数を計算し、配列にして下さい
    res = [x*4 + y*2 + z for x in fours_place for y in twos_place for z in ones_place]

    # 出力して下さい
    print(res)

    # defaultdict ==========================================================================
    description = \
    "Artificial intelligence (AI, also machine intelligence, MI) is " + \
    "intelligence exhibited by machines, rather than " + \
    "humans or other animals (natural intelligence, NI)."

    dic1 = defaultdict(int)
    for c in description:
        dic1[c] += 1
    print(dic1)

    # 正規表現でreplade = re.sub(r"正規表現", "置き換え文字", arg)
    des2 = re.sub(r"[\s,.()]","", description).lower()
    print(des2)
    dic2 = defaultdict(int)
    for c in des2:
        dic2[c] += 1
    print(dic2)
    print("頻度順")
    res2 = sorted(dic2.items(), key=lambda x: x[1], reverse=True)
    print(res2)

    # まとめたいデータprice...(名前, 値段)
    price = [
        ("apple", 50),
        ("banana", 120),
        ("grape", 500),
        ("apple", 70),
        ("lemon", 150),
        ("grape", 1000)
    ]
    # defaultdictを定義して下さい
    dic3 = defaultdict(list)

    # 上記の例と同様にvalueの要素に値段を追加して下さい
    for key, value in price:
        dic3[key].append(value)


    # 各valueの平均値を計算し、配列にして出力して下さい
    res3 = list(map(lambda x: sum(x) / len(x), dic3.values()))
    print("価格の平均値")
    print(res3)

    print("Counter, most_common(n)")

    cntr = Counter(description)
    print(cntr)
    print(cntr.most_common(10))
    print(list(map(lambda x: x[0], cntr.most_common(5))))


def training():
    data = pd.read_csv(
        "http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data", header=None)

    data.columns = ["sepal length", "sepal width", "petal length", "petal width", "class"]
    print(data.head(10))

    with open("csv0.csv", "w") as csvfile:
        writer = csv.writer(csvfile, lineterminator="\n")
        writer.writerow(["city", "year", "season"])
        writer.writerow(["Nagano", 1998, "winter"])
        writer.writerow(["Sydney", 2000, "summer"])
        writer.writerow(["Salt Lake City", 2002, "winter"])
        writer.writerow(["Athens", 2004, "summer"])
        writer.writerow(["Torino", 2006, "winter"])
        writer.writerow(["Beijing", 2008, "summer"])
        writer.writerow(["Vancouver", 2010, "winter"])
        writer.writerow(["London", 2012, "summer"])
        writer.writerow(["Sochi", 2014, "winter"])
        writer.writerow(["Rio de Janeiro", 2016, "summer"])

    pdcsv = {"city": ["Nagano", "Sydney", "Salt Lake"],
             "year": [1998, 2000, 2002],
             "season": ["winter", "summer", "winter"]}
    df = pd.DataFrame(pdcsv)
    # df.to_csv("csv1.csv")


    # DataFrameの復習

    # 連結: データの中身をある方向にそのままつなげる。pd.concat, DataFrame.append
    # 縦に連結:axis=0
    # 横に連結:axis=1

    # 結合: データの中身を何かのキーの値で紐付けてつなげる。pd.merge, DataFrame.join

    # 与えられた2つのDataFrameを結合し出力してください。
    # ただし結合した後のDataFrameはIDの昇順になるようにし、列番号も昇順になるようにしてください。
    attri_data1 = {"ID": ["100", "101", "102", "103", "104", "106", "108", "110", "111", "113"],
                   "city": ["Tokyo", "Osaka", "Kyoto", "Hokkaido", "Tokyo", "Tokyo", "Osaka", "Kyoto", "Hokkaido", "Tokyo"],
                   "birth_year": [1990, 1989, 1992, 1997, 1982, 1991, 1988, 1990, 1995, 1981],
                   "name": ["Hiroshi", "Akiko", "Yuki", "Satoru", "Steeve", "Mituru", "Aoi", "Tarou", "Suguru", "Mitsuo"]}
    attri_data_frame1 = pd.DataFrame(attri_data1)

    attri_data2 = {"ID": ["107", "109"],
                   "city": ["Sendai", "Nagoya"],
                   "birth_year": [1994, 1988]}
    attri_data_frame2 = pd.DataFrame(attri_data2)


    res = attri_data_frame1.append(attri_data_frame2)
    res = res.sort_values(by="ID")
    res = res.reset_index(drop=True)

    print(res)


データクレンジング
    3.1.1
    チャンネル数
    モノクロもRGBも両方１になっている
    RGBはチャンネル数が３なのでは

    3.3.1
    TORUNC => TRUNC

    3.3.3
    説明文
    GaussiannBlur => GaussianBlur



機械学習概論
    3.1.1 and 3.1.2
    混同行列と混合行列とが混在している

    3.1.4
    説明文
    100人のがん患者のうち、20％は「癌ではないだろう」と誤診をしてしまっていますし、陽性の患者が実際に癌である確率は、30％ほどだからです。
    20%?
    40%では
    また、課題の回答、ちょうど80%では
    >>> p = 1
    >>> r = 2/3
    >>> f = 2 * ((p*r)/(p+r))
    >>> f
    0.8

    3.1.5
    16行目　コメントが重複


教師なし学習
    2.3.1
    文中のコード例
    df => dbでは
    説明文中に min_sample と min_samples が混在している

    3.2.2
    カーネルトリック
    カーネル行列KはN*M行列と書かれているが、
    N*N行列ではないか

時系列分析
    3.1.4
    説明文
    定常性の需要さ => 重要さ
    3.2.1
    課題選択肢
    移動平均を取りことによって => 取ることによって

    3.2.4
    script.pyの１４行目
    co2_tsdata2.plot()となっているが、
    plt.plot(co2_tsdata2)である

    4.1.3
    説明文中のコード例
    ２行目
    p = d = q = range(0, 1)
    range(0, 2)では



def main():
    # pre_training()
    training()

if __name__ == '__main__':
    main()

