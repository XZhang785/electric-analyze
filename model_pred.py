import data_Process
from pyspark import SQLContext
from pyspark.context import SparkContext
from pyspark import SparkConf
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegressionModel
from pyspark.ml.regression import GBTRegressionModel
from pyspark.ml.regression import RandomForestRegressionModel
from pyspark.ml.evaluation import RegressionEvaluator
import numpy as np
import json
import time


def pred(filePath='/electric-analyse/data/input/dataset_test.csv'):
    sc = SparkContext('local')
    sqlContext = SQLContext(sc)
    spark = SparkSession.builder.config(conf=SparkConf()).getOrCreate()
    data = spark.read.csv(filePath, header=True, inferSchema=True)

    # df = pd.read_csv('dataset_test.csv')
    df = data.toPandas()
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
    test_df = assembled
    print(test_df.count())

    # 线性回归预测
    lr_rmse = []
    lr_mae = []
    res_col_to_show = ['MP_ID', 'STAT_CYCLE'] + tload
    pd_test_df = test_df.select(res_col_to_show).toPandas()
    time_pred = time.time()

    for col in tload:
        lr_model = LinearRegressionModel.load('model/lr/lr_model_' + col)
        res_df = lr_model.transform(test_df)
        test_p = lr_model.evaluate(test_df)
        # RMSE、MAE计算
        lr_rmse.append(test_p.rootMeanSquaredError)
        lr_mae.append(test_p.meanAbsoluteError)
        # 保存预测结果
        pd_test_df['lr_' + col] = res_df.select('lr_' + col).toPandas()

    # 评估指标记录
    time_pred = time.time() - time_pred
    evaluate_data['lr'] = {
        'avg_rmse': sum(lr_rmse) / len(lr_rmse),
        'avg_mae': sum(lr_mae) / len(lr_mae),
        'time': time_pred / 24
    }

    # 随机森林预测
    rf_error = []
    time_pred = time.time()
    for col in tload:
        print('rf_p_' + col)
        rf_model = RandomForestRegressionModel.load('model/rf/rf_model_' + col)
        rf_pred = rf_model.transform(test_df)
        evaluator_rmse = RegressionEvaluator(labelCol=col, predictionCol='rf_p_' + col, metricName="rmse")
        evaluator_mae = RegressionEvaluator(labelCol=col, predictionCol='rf_p_' + col, metricName="mae")
        rmse = evaluator_rmse.evaluate(rf_pred)
        mae = evaluator_mae.evaluate(rf_pred)
        rf_error.append([rmse, mae])
        pd_test_df['rf_p_' + col] = rf_pred.select('rf_p_' + col).toPandas()

    time_pred = time.time() - time_pred
    rf_error = np.array(rf_error)
    evaluate_data['rf'] = {
        'avg_rmse': rf_error.mean(axis=0)[0],
        'avg_mae': rf_error.mean(axis=0)[1],
        'time': time_pred / 24
    }

    # 梯度提升树预测
    gbt_error = []
    time_pred = time.time()

    for col in tload:
        gbt_model = GBTRegressionModel.load('model/gbt/gbt_model_' + col)
        gbt_pred = gbt_model.transform(test_df)
        evaluator_rmse = RegressionEvaluator(labelCol=col, predictionCol='gbt_p_' + col, metricName="rmse")
        evaluator_mae = RegressionEvaluator(labelCol=col, predictionCol='gbt_p_' + col, metricName="mae")
        # RMSE、MAE计算
        rmse = evaluator_rmse.evaluate(gbt_pred)
        mae = evaluator_mae.evaluate(gbt_pred)
        gbt_error.append([rmse, mae])
        # 保存预测结果
        pd_test_df['gbt_p_' + col] = gbt_pred.select('gbt_p_' + col).toPandas()
    time_pred = time.time() - time_pred
    evaluate_data['gbt'] = {
        'avg_rmse': gbt_error.mean(axis=0)[0],
        'avg_mae': gbt_error.mean(axis=0)[1],
        'time': time_pred / 24
    }

    pd_test_df = sqlContext.createDataFrame(pd_test_df)
    pd_test_df = pd_test_df.drop('STAT_CYCLE')
    pd_test_df.write.mode('overwrite').csv('/electric-analyse/data/output/pred_res.csv', header=True)

    print(evaluate_data)
    with open("evaluate_data.json", "w") as fp:
        json.dump(evaluate_data, fp)

    print('OK')


if __name__ == "__main__":
    pred()
