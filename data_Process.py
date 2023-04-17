import pandas as pd
from data_process_function import *
from sklearn.decomposition import PCA  # 在sklearn中调用PCA机器学习算法


def data_process(df):
    # df = pd.read_csv('dataset.csv')
    df = noisy_data_process(df)  # 异常噪声处理

    # 离散数据连续化-独热编码
    df1 = pd.get_dummies(df['AREA_NO'], sparse=True)
    df1.columns = ["AREA_NO_%s" % cname for cname in df1.columns]
    df2 = pd.get_dummies(df['HOLIDAY'], sparse=True)
    df2.columns = ["HOLIDAY_%s" % cname for cname in df2.columns]
    df3 = pd.get_dummies(df['ELECTRO_TYPE'], sparse=True)
    df3.columns = ["ELECTRO_TYPE_%s" % cname for cname in df3.columns]
    df.drop(columns=["AREA_NO"], inplace=True)
    df.drop(columns=["HOLIDAY"], inplace=True)
    df.drop(columns=["ELECTRO_TYPE"], inplace=True)
    df = df.join(df1).join(df2).join(df3)

    # 数据标准化
    df = my_std(df, ["WINDSPEED", "LAPSERATE", "AIRPRESSURE", "HUMIDITY", "PRECIPITATIONRANINFALL"], 'max-min')

    # 特征提取
    df4 = df[["WINDSPEED", "LAPSERATE", "AIRPRESSURE", "HUMIDITY", "PRECIPITATIONRANINFALL"]]
    data_rec = df4.values
    pca = PCA(n_components=3)  # 定义所需要分析主成分的个数n
    pca.fit(data_rec)  # 对基础数据集进行相关的计算，求取相应的主成分
    data_rec_reduction = pca.transform(data_rec)  # 进行数据的降维
    explained_var = pca.explained_variance_ratio_  # 获取贡献率
    # print("降维后变量的贡献率", explained_var)
    df5 = pd.DataFrame(data=data_rec_reduction, columns=["PCA_%d" % x for x in range(1, 4)])
    df = df.join(df5)

    # 缺失值处理
    ddf = dataS(df)

    return ddf
