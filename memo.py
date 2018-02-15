import numpy as np
from numpy.random import randint, rand
import pandas as pd

def numpy_example():

    x = np.arange(30).reshape(3, 10)
    print(x)

    y = np.array([i for i in range(10)]).reshape(2, 5)
    print(y)

    xx = x[1:, 3:8]
    print(xx)

    res = np.dot(y, xx.transpose())

    print("res\n", res)
    print("res sort\n", np.sort(res,axis=0))
    print("res argsort\n", res.argsort(axis=0))
    print("res max\n", res.max())
    print("res argmax\n", res.argmax(axis=0))

    print("randint")
    print(randint(3, 23, 10))
    print("rand")
    print(rand(10))


    a = np.union1d(xx, y)
    b = np.intersect1d(xx, y)
    c = np.setdiff1d(xx, y)
    # print("和集合\n{}\n積集合\n{}\n差集合\n{}\n".format(a,b,c)))
    print(a)
    print(b)
    print(c)


    d = randint(1, 10, 5)
    print(d)
    print(np.unique(d))

    r = (d % 2 == 0)
    print(r)

    g = d[d % 2 == 0]
    print(g)

    print("fancy\n", d[[1,3,4]])

    print(d.mean())
    print(d.std())
    print(d.var())


    print(np.linalg.norm(d))


def pandas_example():
    # Series用のラベル作成（インデックス）
    index = ["apple", "orange", "banana", "strawberry", "kiwifruit"]
    # Series用のデータ値の代入
    data = [10, 5, 8, 12, 3]
    # Series作成
    series = pd.Series(data, index=index)

    # 辞書型を用いてDataFrame用のデータの作成
    data = {"fruits": ["apple", "orange", "banana", "strawberry", "kiwifruit"],
            "year": [2001, 2002, 2001, 2008, 2006],
            "time": [1, 4, 5, 6, 3]}

    # DataFrame作成
    df = pd.DataFrame(data)

    print("Seriesデータ")
    print(series)
    print()
    print("DataFrameデータ")
    print(df)

    print(series[1:4])
    print(series.values)
    print(series.index)
    print(series[['apple', 'banana']])

    print(df.index)
    print(df.values)

    series = series.append(pd.Series([12], index=["pineapple"]))
    print("appended series")
    print(series)
    print(series.sort_index(ascending = False))
    print(series.sort_values(ascending = False))
    # デフォルトはTrue
    series = series[series > 10]

    ind = ["first", "second", "third", "forth", "fifth"]
    d1 = [1299,3323,6483,6640,2342]
    d2 = [3434,3212,1231,1231,1634]
    d3 = [1111,2222,3333,4444,5655]
    s1 = pd.Series(d1, index=ind)
    s2 = pd.Series(d2, index=ind)
    s3 = pd.Series(d3, index=ind)

    "データフレーム"
    df = pd.DataFrame([s1, s2, s3])
    # df.columns = ["fi", "se", "th", "fo", "fi"]
    # df.index = [1, 2]
    print(df)

    "データの追加"
    # 行の追加
    # d3 = ["ffff", "ssss", "tttt", "ffff", "fooo"]
    # s3 = pd.Series(d3, index=ind)
    # df = df.append([s3], ignore_index=True)
    # 列の追加
    # df["sixth"] = [4444,7777,"sxxx"]
    # print(df)

    # 部分抽出
    # df = df.loc[range(1, 3), ["second", "third"]]
    # df = df.iloc[range(1, 2), [2, 4]]
    # print(df)

    # 列、行の削除
    # df = df.drop(range(1,3))
    # df = df.drop("third", axis=1)
    # print(df)

    # ソート
    df = df.sort_values(by=["second","forth","fifth"])
    # df = df.sort_values(by=["second","forth","third","first","fifth"])
    # df = df.sort_values(by=["second","forth","third","first","fifth"], ascending=False)
    # df.sort_index()
    print(df)


    # 条件式による抽出
    df["apple"] >= 5
    # これでブール値を持つ配列ができる
    # 下のやり方のどちらでも可能
    # df = df[df["apple"] >= 5]
    # df = df[df["kiwifruit"] >= 5]
    # df = df.loc[df["apple"] >= 5]
    # df = df.loc[df["kiwifruit"] >= 5]


def make_random_df(index, columns, seed):
    np.random.seed(seed)
    df = pd.DataFrame()
    for column in columns:
        df[column] = np.random.choice(range(1, 101), len(index))

    df.index = index

    return df



def DataFrameConcat():
    # 連結 concat ==============================================================
    columns = ["apple", "orange", "banana"]

    df_data1 = make_random_df(range(1, 5), columns, 0)
    df_data2 = make_random_df(range(1, 5), columns, 0)

    # df1 = pd.concat([df_data1, df_data2], axis=1)
    df1 = pd.concat([df_data1, df_data2], axis=0)
    print("\naxis = 0\n", df1)

    df2 = pd.concat([df_data1, df_data2], axis=1, keys=["X", "Y"])
    print("\naxis = 1\n", df2)

    partial_df2 = df2["Y", "orange"]
    print("\nY's orange\n", partial_df2)

    # 結合 merge ==============================================================    
    """内部結合と外部結合
        内部結合＝key列に共通の値のない行は破棄される
        外部結合＝共通の値のない行もNaNになって残る
    """
    print("\n======== merge ========\n")
    data1 = {"fruits": ["apple", "orange", "banana", "strawberry", "kiwifruit"],
            "year": [2001, 2002, 2001, 2008, 2006],
            "amount": [1, 4, 5, 6, 3]}

    df3 = pd.DataFrame(data1)

    data2 = {"fruits": ["apple", "orange", "banana", "strawberry", "mango"],
            "year": [2001, 2002, 2001, 2008, 2007],
            "price": [150, 120, 100, 250, 3000]}
    df4 = pd.DataFrame(data2)

    df5 = pd.merge(df3, df4, on="fruits", how="inner")
    print("inner\n",df5)
    df5 = pd.merge(df3, df4, on="fruits", how="outer")
    print("outer\n",df5)
    df5 = pd.merge(df3, df4, on="fruits")
    print("nothing\n",df5)

    # 注文情報
    order_df = pd.DataFrame([[1000, 2546, 103],
                             [1001, 4352, 101],
                             [1002, 342, 101]],
                             columns=["id", "item_id", "customer_id"])
    # 顧客情報
    customer_df = pd.DataFrame([[101, "Tanaka"],
                               [102, "Suzuki"],
                               [103, "Kato"]],
                               columns=["id", "name"])

    updated_df = pd.merge(order_df, customer_df, left_on="customer_id", right_on="id", how="outer")
    print("\n同盟でない列をキーにして結合する")
    print(updated_df)

    # インデックスをキーとして結合する
    # 注文情報
    order_df = pd.DataFrame([[1000, 2546, 103],
                             [1001, 4352, 101],
                             [1002, 342, 101]],
                             columns=["id", "item_id", "customer_id"])
    # 顧客情報
    customer_df = pd.DataFrame([["Tanaka"],
                               ["Suzuki"],
                               ["Kato"]],
                               columns=["name"])
    customer_df.index = [101, 102, 103]

    # customer_dfを元に"name"をorder_dfに結合してorder_dfに代入してください
    order_df = pd.merge(order_df, customer_df, left_on="customer_id", right_index=True, how="inner")

    # 計算処理
    print(df1 * 2)
    print(df1 ** 2)
    print(df1 % 2)

    # 要約統計量
    des = df1.describe()
    print(des)
    print(des.loc[["mean", "max", "min"]])

    # DataFrameの行間または列間の差を求める
    # 第一引数 正の値なら前の行ととの差、負の値なら後ろの行との差
    df_diff = df1.diff(-1, axis=1)
    # df_diff = df1.diff(-2, axis=0)
    print(df_diff)

    # グループ化
    # 一部の都道府県に関するDataFrameを作成
    prefecture_df = pd.DataFrame([["Tokyo", 2190, 13636, "Kanto"], ["Kanagawa", 2415, 9145, "Kanto"],
                                  ["Osaka", 1904, 8837, "Kinki"], ["Kyoto", 4610, 2605, "Kinki"],
                                  ["Aichi", 5172, 7505, "Chubu"]], 
                                 columns=["Prefecture", "Area", "Population", "Region"])
    # 出力
    print("prefecture")
    print(prefecture_df)

    grouped_region = prefecture_df.groupby("Region")
    mean_df = grouped_region.mean()
    print(mean_df)

    sum_df = grouped_region.sum()
    print(sum)






def main():
    # numpy_example()
    # pandas_example()
    DataFrameConcat()




if __name__ == '__main__':
    main()
