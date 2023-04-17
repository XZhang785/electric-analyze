import pandas as pd
import datetime


def get_list(date):
    return datetime.datetime.strptime(date, "%Y/%m/%d").timestamp()


def z_score_std(data_col):
    # Z-SCORE标准化
    return (data_col - data_col.mean()) / data_col.std()


def max_min_std(data_col):
    #MAX-MIN标准化
    return (data_col-data_col.min())/(data_col.max()-data_col.min())


def my_std(df, col_names, mode):
    # mode:'z-score':z_score_std;'max-min':max_min_std
    df_tem = df
    if mode == 'z-score':
        for c in col_names:
            df_tem[c] = z_score_std(df[c].values)
    elif mode == 'max-min':
        for c in col_names:
            df_tem[c] = max_min_std(df[c].values)
    return df_tem


def date_back_month(date):
    x = date
    x_year = x[0:x.find('/')]
    x = x[x.find('/') + 1:len(x) + 1]
    x_month = x[0:x.find('/')]
    x_day = x[x.find('/') + 1:len(x) + 1]
    if x_month == '1':
        x_month = '12'
        x_year = "%d" % (int(x_year) - 1)
    else:
        x_month = "%d" % (int(x_month) - 1)
    return (x_year + '/' + x_month + '/' + x_day)


def date_back_week(date):
    d = (datetime.datetime.strptime(date, "%Y/%m/%d") + datetime.timedelta(days=-7))
    x = '{:%Y/%m/%d}'.format(d)
    return x


def date_back_day(date):
    d = (datetime.datetime.strptime(date, "%Y/%m/%d") + datetime.timedelta(days=-1))
    x = '{:%Y/%m/%d}'.format(d)
    return x


def date_forward_month(date):
    x = date
    x_year = x[0:x.find('/')]
    x = x[x.find('/') + 1:len(x) + 1]
    x_month = x[0:x.find('/')]
    x_day = x[x.find('/') + 1:len(x) + 1]
    if x_month == '12':
        x_month = '1'
        x_year = "%d" % (int(x_year) + 1)
    else:
        x_month = "%d" % (int(x_month) + 1)
    return (x_year + '/' + x_month + '/' + x_day)


def date_forward_week(date):
    d = (datetime.datetime.strptime(date, "%Y/%m/%d") + datetime.timedelta(days=7))
    x = '{:%Y/%m/%d}'.format(d)
    return x


def date_forward_day(date):
    d = (datetime.datetime.strptime(date, "%Y/%m/%d") + datetime.timedelta(days=1))
    x = '{:%Y/%m/%d}'.format(d)
    return x


# 异常值处理
def noisy_data_process(_df):
    mload, wload, dload, load, tload = [], [], [], [], []
    for i in range(24):
        mload.append('MLOAD' + str(i))
        wload.append('WLOAD' + str(i))
        dload.append('DLOAD' + str(i))
        load.append('LOAD' + str(i))
        tload.append('TLOAD' + str(i))
    column_temp = ["WINDSPEED", "LAPSERATE", "AIRPRESSURE", "HUMIDITY"] + mload + wload + dload + load
    # 四分位数法去除离群点
    for _col in column_temp:
        q1, q3 = _df[_col].quantile([0.25, 0.75])
        iqr = q3 - q1
        _df = _df.drop(_df[(_df[_col] < q1 - iqr * 1.5) | (_df[_col] > q3 + iqr * 3)].index)
    # 去除0值
    _df = _df[~(_df[mload + wload + dload + load] == 0).all(axis=1)]
    return _df


def dataS(df):
    # 将MP_ID的数据类型设为int
    df["MP_ID"] = df["MP_ID"].astype(int)

    # 删除空值，并重置索引
    df = df.drop_duplicates()
    df = df.reset_index(drop=True)

    # 列名
    col = pd.DataFrame(['LOAD0', 'LOAD1', 'LOAD2', 'LOAD3', 'LOAD4', 'LOAD5',
                        'LOAD6', 'LOAD7', 'LOAD8', 'LOAD9', 'LOAD10', 'LOAD11',
                        'LOAD12', 'LOAD13', 'LOAD14', 'LOAD15', 'LOAD16', 'LOAD17',
                        'LOAD18', 'LOAD19', 'LOAD20', 'LOAD21', 'LOAD22', 'LOAD23'])
    Mcol = ("M" + col[0]).to_list()
    Wcol = ("W" + col[0]).to_list()
    Lcol = ("D" + col[0]).to_list()
    Tcol = ("T" + col[0]).to_list()
    Dcol = col[0].to_list()

    # 月数据处理（转化为前一个月的日期）
    mSTAT_CYCLE = pd.DataFrame(pd.to_datetime(df['STAT_CYCLE']) - datetime.timedelta(days=31))
    Mdf = pd.DataFrame(df[Mcol].apply(pd.to_numeric))
    Mdf.columns = Dcol
    Mdf.insert(loc=0, column='MP_ID', value=df['MP_ID'])
    Mdf.insert(loc=1, column='STAT_CYCLE', value=mSTAT_CYCLE)
    Mdf["ID"] = Mdf["MP_ID"].astype('str') + "," + Mdf["STAT_CYCLE"].astype('str')
    Mdf = Mdf.set_index('ID')

    # 周数据处理（转化为前一周的日期）
    wSTAT_CYCLE = pd.DataFrame(pd.to_datetime(df['STAT_CYCLE']) - datetime.timedelta(days=7))
    Wdf = pd.DataFrame(df[Wcol].apply(pd.to_numeric))
    Wdf.columns = Dcol
    Wdf.insert(loc=0, column='MP_ID', value=df['MP_ID'])
    Wdf.insert(loc=1, column='STAT_CYCLE', value=wSTAT_CYCLE)
    Wdf["ID"] = Wdf["MP_ID"].astype('str') + "," + Wdf["STAT_CYCLE"].astype('str')
    Wdf = Wdf.set_index('ID')

    # 前一日数据处理
    lSTAT_CYCLE = pd.DataFrame(pd.to_datetime(df['STAT_CYCLE']) - datetime.timedelta(days=1))
    Ldf = pd.DataFrame(df[Lcol].apply(pd.to_numeric))
    Ldf.columns = Dcol
    Ldf.insert(loc=0, column='MP_ID', value=df['MP_ID'])
    Ldf.insert(loc=1, column='STAT_CYCLE', value=lSTAT_CYCLE)
    Ldf["ID"] = Ldf["MP_ID"].astype('str') + "," + Ldf["STAT_CYCLE"].astype('str')
    Ldf = Ldf.set_index('ID')

    # 当日数据
    dSTAT_CYCLE = pd.DataFrame(pd.to_datetime(df['STAT_CYCLE']))
    Ddf = pd.DataFrame(df[Dcol].apply(pd.to_numeric))
    Ddf.insert(loc=0, column='MP_ID', value=df['MP_ID'])
    Ddf.insert(loc=1, column='STAT_CYCLE', value=dSTAT_CYCLE)
    Ddf["ID"] = Ddf["MP_ID"].astype('str') + "," + Ddf["STAT_CYCLE"].astype('str')
    Ddf = Ddf.set_index('ID')

    # 合并当日数据，补充缺失值
    DDdf = Ddf.combine_first(Mdf)
    DDdf = DDdf.combine_first(Wdf)
    DDdf = DDdf.combine_first(Ldf)
    DDdf["MP_ID"] = DDdf["MP_ID"].astype(int)

    # 计算昨日数据
    LLdf = pd.DataFrame(DDdf)
    LLdf['STAT_CYCLE'] = LLdf['STAT_CYCLE'] + datetime.timedelta(days=1)
    LLdf["ID"] = LLdf["MP_ID"].astype('str') + "," + LLdf["STAT_CYCLE"].astype('str')
    LLdf = LLdf.set_index('ID')
    LLdf.columns = ['MP_ID', 'STAT_CYCLE'] + Lcol

    # 计算明日数据
    TTdf = pd.DataFrame(DDdf)
    TTdf['STAT_CYCLE'] = TTdf['STAT_CYCLE'] - datetime.timedelta(days=2)
    TTdf["ID"] = TTdf["MP_ID"].astype('str') + "," + TTdf["STAT_CYCLE"].astype('str')
    TTdf = TTdf.set_index('ID')
    TTdf.columns = ['MP_ID', 'STAT_CYCLE'] + Tcol
    TTdf.head()

    # 将数据合并入df中
    df["ID"] = df["MP_ID"].astype('str') + "," + pd.to_datetime(df["STAT_CYCLE"]).astype('str')
    df = df.set_index('ID')
    df[Tcol] = None
    ddf = df.combine_first(LLdf)
    ddf = ddf.combine_first(TTdf)
    ddf = ddf.combine_first(DDdf)

    # 对列进行重新排序
    ddf = ddf[df.columns]
    ddf["MP_ID"] = ddf["MP_ID"].astype(int)
    # 删除月数据与周数据
    ddf.drop(ddf[Wcol], axis=1, inplace=True)
    ddf.drop(ddf[Mcol], axis=1, inplace=True)
    # 删除空值
    ddf.dropna(axis=0, how='any', inplace=True)
    ddf = ddf.drop_duplicates()
    ddf = ddf.reset_index(drop=True)
    return ddf
