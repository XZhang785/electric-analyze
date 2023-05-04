import pandas as pd
import datetime


def get_timestamp(date):
    return datetime.datetime.strptime(date, "%Y/%m/%d").timestamp()


def z_score_std(data_col: pd.DataFrame):
    # Z-SCORE标准化
    return (data_col - data_col.mean()) / data_col.std()


def max_min_std(data_col: pd.DataFrame):
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
    """
    计算上一个月的日期
    :param date:
    :return:
    """
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
    """
    计算上周的日期
    :param date:
    :return:
    """
    d = (datetime.datetime.strptime(date, "%Y/%m/%d") + datetime.timedelta(days=-7))
    x = '{:%Y/%m/%d}'.format(d)
    return x


def date_back_day(date):
    """
    计算前天的日期
    :param date:
    :return:
    """
    d = (datetime.datetime.strptime(date, "%Y/%m/%d") + datetime.timedelta(days=-1))
    x = '{:%Y/%m/%d}'.format(d)
    return x


def date_forward_month(date):
    """
    计算下个月的日期
    :param date:
    :return:
    """
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
    """
    计算下周的日期
    :param date:
    :return:
    """
    d = (datetime.datetime.strptime(date, "%Y/%m/%d") + datetime.timedelta(days=7))
    x = '{:%Y/%m/%d}'.format(d)
    return x


def date_forward_day(date):
    """
    计算明天的日期
    :param date:
    :return:
    """
    d = (datetime.datetime.strptime(date, "%Y/%m/%d") + datetime.timedelta(days=1))
    x = '{:%Y/%m/%d}'.format(d)
    return x


# 异常值处理
def noisy_data_process(_df):
    """
    通过四分位法去除噪声，然后去除值为0的异常点
    :param _df:
    :return:
    """
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
        q1, q3 = _df[_col].quantile([0.25, 0.75])  # 计算四分位点
        iqr = q3 - q1
        _df = _df.drop(_df[(_df[_col] < q1 - iqr * 1.5) | (_df[_col] > q3 + iqr * 3)].index)
    # 去除0值
    _df = _df[~(_df[mload + wload + dload + load] == 0).all(axis=1)]
    return _df


def dataS(df):
    """
    通过周期性数据去填充当日数据的缺失值，然后保留前一日和后一日的数据
    :param df:
    :return:
    """
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
    Dcol = ("D" + col[0]).to_list()
    Tcol = ("T" + col[0]).to_list()
    Dcol = col[0].to_list()

    # 月数据处理（转化为前一个月的日期）
    m_stat_cycle = pd.DataFrame(pd.to_datetime(df['STAT_CYCLE']) - datetime.timedelta(days=31))  # 上个月的日期
    Mdf = pd.DataFrame(df[Mcol].apply(pd.to_numeric))  # 单独取出前一个月的数据
    Mdf.columns = Dcol  # 把列名改为LOAD
    Mdf.insert(loc=0, column='MP_ID', value=df['MP_ID'])  # 插入采集点的id
    Mdf.insert(loc=1, column='STAT_CYCLE', value=m_stat_cycle)  # 插入上个月的日期
    Mdf["ID"] = Mdf["MP_ID"].astype('str') + "," + Mdf["STAT_CYCLE"].astype('str')  # 创建key
    Mdf = Mdf.set_index('ID')  # 将key作为索引

    # 周数据处理（转化为前一周的日期）
    wSTAT_CYCLE = pd.DataFrame(pd.to_datetime(df['STAT_CYCLE']) - datetime.timedelta(days=7))
    Wdf = pd.DataFrame(df[Wcol].apply(pd.to_numeric))
    Wdf.columns = Dcol
    Wdf.insert(loc=0, column='MP_ID', value=df['MP_ID'])
    Wdf.insert(loc=1, column='STAT_CYCLE', value=wSTAT_CYCLE)
    Wdf["ID"] = Wdf["MP_ID"].astype('str') + "," + Wdf["STAT_CYCLE"].astype('str')
    Wdf = Wdf.set_index('ID')

    # 前一日数据处理
    DSTAT_CYCLE = pd.DataFrame(pd.to_datetime(df['STAT_CYCLE']) - datetime.timedelta(days=1))
    Ddf = pd.DataFrame(df[Dcol].apply(pd.to_numeric))
    Ddf.columns = Dcol
    Ddf.insert(loc=0, column='MP_ID', value=df['MP_ID'])
    Ddf.insert(loc=1, column='STAT_CYCLE', value=DSTAT_CYCLE)
    Ddf["ID"] = Ddf["MP_ID"].astype('str') + "," + Ddf["STAT_CYCLE"].astype('str')
    Ddf = Ddf.set_index('ID')

    # 当日数据
    LSTAT_CYCLE = pd.DataFrame(pd.to_datetime(df['STAT_CYCLE']))
    Ldf = pd.DataFrame(df[Dcol].apply(pd.to_numeric))
    Ldf.insert(loc=0, column='MP_ID', value=df['MP_ID'])
    Ldf.insert(loc=1, column='STAT_CYCLE', value=LSTAT_CYCLE)
    Ldf["ID"] = Ldf["MP_ID"].astype('str') + "," + Ldf["STAT_CYCLE"].astype('str')
    Ldf = Ldf.set_index('ID')

    # 通过上一次记录的数据，补充缺失值
    DDdf = Ldf.combine_first(Mdf)
    DDdf = DDdf.combine_first(Wdf)
    DDdf = DDdf.combine_first(Ddf)
    DDdf["MP_ID"] = DDdf["MP_ID"].astype(int)

    # 计算昨日数据
    LLdf = pd.DataFrame(DDdf)
    LLdf['STAT_CYCLE'] = LLdf['STAT_CYCLE'] + datetime.timedelta(days=1)
    LLdf["ID"] = LLdf["MP_ID"].astype('str') + "," + LLdf["STAT_CYCLE"].astype('str')
    LLdf = LLdf.set_index('ID')
    LLdf.columns = ['MP_ID', 'STAT_CYCLE'] + Dcol

    # 计算明日数据
    TTdf = pd.DataFrame(DDdf)
    TTdf['STAT_CYCLE'] = TTdf['STAT_CYCLE'] - datetime.timedelta(days=1)
    TTdf["ID"] = TTdf["MP_ID"].astype('str') + "," + TTdf["STAT_CYCLE"].astype('str')
    TTdf = TTdf.set_index('ID')
    TTdf.columns = ['MP_ID', 'STAT_CYCLE'] + Tcol

    # 将数据合并入df中
    df["ID"] = df["MP_ID"].astype('str') + "," + pd.to_datetime(df["STAT_CYCLE"]).astype('str')
    df = df.set_index('ID')
    df[Tcol] = None
    Ddf = df.combine_first(LLdf)
    Ddf = Ddf.combine_first(TTdf)
    Ddf = Ddf.combine_first(DDdf)

    # 对列进行重新排序
    Ddf = Ddf[df.columns]
    Ddf["MP_ID"] = Ddf["MP_ID"].astype(int)
    # 删除月数据与周数据
    Ddf.drop(Ddf[Wcol], axis=1, inplace=True)
    Ddf.drop(Ddf[Mcol], axis=1, inplace=True)
    # 删除空值
    Ddf.dropna(axis=0, how='any', inplace=True)
    Ddf = Ddf.drop_duplicates()
    Ddf = Ddf.reset_index(drop=True)
    return Ddf
