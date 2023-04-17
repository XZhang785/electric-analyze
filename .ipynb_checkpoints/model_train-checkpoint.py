import data_Process
from pyspark import SQLContext
from pyspark.context import SparkContext
from pyspark import SparkConf
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression, LinearRegressionModel
from pyspark.ml.regression import RandomForestRegressor, RandomForestRegressionModel
import json
import time

def train(filePath='/electric-analyse/data/input/dataset_train.csv'):
    sc = SparkContext('local')
    sqlContext = SQLContext(sc)
    spark = SparkSession.builder.config(conf=SparkConf()).config("spark.debug.maxToStringFields", "200").getOrCreate()
    data = spark.read.csv(filePath, header=True,inferSchema=True)

    df = data.toPandas()
    # df = pd.read_csv('dataset_train.csv')
    df = data_Process.data_process(df)
    sdf = sqlContext.createDataFrame(df)
    print(df.shape)

    mload, wload, dload, load, tload = [], [], [], [], []
    for i in range(24):
        mload.append('MLOAD' + str(i))
        wload.append('WLOAD' + str(i))
        dload.append('DLOAD' + str(i))
        load.append('LOAD' + str(i))
        tload.append('TLOAD' + str(i))

    featureCol = ['AREA_NO_35401', 'AREA_NO_35402', 'AREA_NO_35403', 'AREA_NO_35404',
                  'AREA_NO_35405', 'AREA_NO_35406', 'AREA_NO_35408',
                  'AREA_NO_35409', 'HOLIDAY_0', 'HOLIDAY_1', 'HOLIDAY_2',
                  'ELECTRO_TYPE_0', 'ELECTRO_TYPE_100', 'ELECTRO_TYPE_101',
                  'ELECTRO_TYPE_201', 'ELECTRO_TYPE_202', 'ELECTRO_TYPE_203',
                  'ELECTRO_TYPE_300', 'ELECTRO_TYPE_401', 'ELECTRO_TYPE_402',
                  'ELECTRO_TYPE_403', 'ELECTRO_TYPE_405', 'ELECTRO_TYPE_500', 'PCA_1',
                  'PCA_2', 'PCA_3'] + dload + load

    assembler = VectorAssembler(inputCols=featureCol, outputCol="features")
    assembled = assembler.transform(sdf)

    evaluate_data = {}
    train_df = assembled
    print(train_df.count())

    # 线性回归
    lr_rmse = []
    lr_mae = []
    res_col_to_show = ['MP_ID', 'STAT_CYCLE'] + tload

    for Col in tload:
        print('lr_' + Col)
        # 训练模型
        temp = time.time()
        lr = LinearRegression(labelCol=Col, predictionCol='lr_' + Col)
        lr_model = lr.fit(train_df)
        # 保存模型
        lr_model.save('model/lr/lr_model_' + Col)

    # 评估指标记录
    time_train = time.time() - temp
    evaluate_data['lr'] = {
        'time': time_train / train_df.count()
    }

    # 随机森林
    for Col in tload:
        print('rf_p_' + Col)
        temp = time.time()
        rf = RandomForestRegressor(labelCol=Col, predictionCol='rf_p_' + Col, maxDepth=8, seed=66)
        rf_model = rf.fit(train_df)
        rf_model.save('model/rf/rf_model_' + Col)

    time_train = time.time() - temp

    evaluate_data['rf'] = {
        'time': time_train / train_df.count()
    }

    print(evaluate_data)

    with open("train_time_data.json", "w") as fp:
        json.dump(evaluate_data, fp)
    print("OK")

    
if __name__ == "__main__":
    train()