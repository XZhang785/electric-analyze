import data_Process
from pyspark import SparkConf
from pyspark.context import SparkContext
from pyspark.sql import SQLContext
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression, GBTRegressor, RandomForestRegressor
from pyspark.sql import SparkSession
import json
import time


def train(spark, filePath='/electric-analyse/data/input/dataset_train.csv'):
    spark = spark
    sc = spark.sparkContext
    sql = SQLContext(sc)
    # 读取 CSV 文件并转换为 Spark DataFrame
    data = spark.read.csv(filePath, header=True, inferSchema=True)
    df = data.toPandas()
    df = data_Process.data_process(df)
    sdf = sql.createDataFrame(df)

    # 特征列
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
                  'ELECTRO_TYPE_403', 'ELECTRO_TYPE_405', 'ELECTRO_TYPE_500',
                  "WINDSPEED", "LAPSERATE", "AIRPRESSURE", "HUMIDITY", "PRECIPITATIONRANINFALL"] + dload + load

    assembler = VectorAssembler(inputCols=featureCol, outputCol="features")
    assembled = assembler.transform(sdf)

    evaluate_data = {}
    train_df = assembled

    # 线性回归
    temp = time.time()
    for col in tload:
        # 训练模型
        lr = LinearRegression(labelCol=col, predictionCol='lr_' + col)
        lr_model = lr.fit(train_df)
        # 保存模型
        lr_model.write().overwrite().save('model/lr/lr_model_' + col)

    # 评估指标记录
    time_train = time.time() - temp
    evaluate_data['lr'] = {
        'time': time_train
    }

    # 随机森林
    temp = time.time()
    for col in tload:
        # 训练模型
        rf = RandomForestRegressor(labelCol=col, predictionCol='rf_p_' + col, maxDepth=8, seed=66)
        rf_model = rf.fit(train_df)
        # 保存模型
        rf_model.write().overwrite().save('model/rf/rf_model_' + col)

    # 评估指标记录
    time_train = time.time() - temp
    evaluate_data['rf'] = {
        'time': time_train
    }

    # 梯度提升树
    temp = time.time()
    for col in tload:
        # 训练模型
        gbt = GBTRegressor(labelCol=col, predictionCol='gbt_p_' + col, maxIter=10)
        gbt_model = gbt.fit(train_df)
        # 保存模型
        gbt_model.write().overwrite().save('model/gbt/gbt_model_' + col)

    # 评估指标记录
    time_train = time.time() - temp
    evaluate_data['gbt'] = {
        'time': time_train
    }

    with open("train_time_data.json", "w") as fp:
        json.dump(evaluate_data, fp)


if __name__ == "__main__":
    spark = SparkSession.builder.master('spark://Master032004134:7077').appName("electric_analyze").getOrCreate()
    spark.sparkContext.setLogLevel("Error")
    train(spark)
