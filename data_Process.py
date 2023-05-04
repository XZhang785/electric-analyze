from data_process_function import *


def data_process(df):
    df = remove_noise(df)  # 异常噪声处理

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
    df = standardize(df, ["WINDSPEED", "LAPSERATE", "AIRPRESSURE", "HUMIDITY", "PRECIPITATIONRANINFALL"], 'max-min')

    # 缺失值处理
    ddf = fill_missing_values(df)

    return ddf
