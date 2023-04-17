import data_Process
from pyspark import SQLContext
from pyspark.context import SparkContext
from pyspark import SparkConf
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression, LinearRegressionModel
from pyspark.ml.regression import RandomForestRegressor, RandomForestRegressionModel
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql import functions as F
import pyspark.sql.types as T
import pmdarima as pm
import numpy as np
import json
import time

def pred(filePath='/electric-analyse/data/input/dataset_test.csv'):
    sc = SparkContext('local')
    sqlContext = SQLContext(sc)
    spark = SparkSession.builder.config(conf=SparkConf()).getOrCreate()
    data = spark.read.csv(filePath, header=True,inferSchema=True)

    # df = pd.read_csv('dataset_test.csv')
    df = data.toPandas()
    df = data_Process.data_process(df)
    sdf = sqlContext.createDataFrame(df)
    print(df.shape)

    mload, wload, dload, load, tload = [],[],[],[],[]
    for i in range(24):
        mload.append('MLOAD'+str(i))
        wload.append('WLOAD'+str(i))
        dload.append('DLOAD'+str(i))
        load.append('LOAD'+str(i))
        tload.append('TLOAD'+str(i))

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

    for Col in tload:
        lr_model = LinearRegressionModel.load('model/lr/lr_model_' + Col)
        res_df = lr_model.transform(test_df)
        test_p = lr_model.evaluate(test_df)
        # RMSE、MAE计算
        lr_rmse.append(test_p.rootMeanSquaredError)
        lr_mae.append(test_p.meanAbsoluteError)
        # 保存预测结果
        pd_test_df['lr_' + Col] = res_df.select('lr_' + Col).toPandas()

    # 评估指标记录
    time_pred = time.time() - time_pred
    evaluate_data['lr'] = {
        'rmse': list(lr_rmse),
        'mae': list(lr_mae),
        'avg_rmse': sum(lr_rmse) / len(lr_rmse),
        'avg_mae': sum(lr_mae) / len(lr_mae),
        'time': time_pred / test_df.count()
    }

    # 随机森林预测
    rf_error = []
    time_pred = time.time()
    for Col in tload:
        print('rf_p_' + Col)
        rf_model = RandomForestRegressionModel.load('model/rf/rf_model_' + Col)
        rf_pred = rf_model.transform(test_df)
        evaluator_rmse = RegressionEvaluator(labelCol=Col, predictionCol='rf_p_' + Col, metricName="rmse")
        evaluator_mae = RegressionEvaluator(labelCol=Col, predictionCol='rf_p_' + Col, metricName="mae")
        rmse = evaluator_rmse.evaluate(rf_pred)
        mae = evaluator_mae.evaluate(rf_pred)
        rf_error.append([rmse, mae])
        pd_test_df['rf_p_' + Col] = rf_pred.select('rf_p_' + Col).toPandas()

    time_pred = time.time() - time_pred
    rf_error = np.array(rf_error)
    evaluate_data['rf'] = {
        'rmse': list(rf_error[:, 0]),
        'mae': list(rf_error[:, 1]),
        'avg_rmse': rf_error.mean(axis=0)[0],
        'avg_mae': rf_error.mean(axis=0)[1],
        'time': time_pred / test_df.count()
    }

    # pd_test_df.to_csv('pred_res.csv')
    pd_test_df = sqlContext.createDataFrame(pd_test_df)
    # pd_test_df.printSchema()
    pd_test_df = pd_test_df.drop('STAT_CYCLE')    
    pd_test_df.write.mode('overwrite').csv('/electric-analyse/data/output/pred_res.csv',header=True)

    # ARIMA预测

    assembler_arima = VectorAssembler(inputCols=dload + load, outputCol="features")
    df_arima = assembler_arima.transform(sdf)
    df_arima, _ = df_arima.randomSplit([0.002, 0.998])


    # 输入为一个数组，由DLOAD+LOAD组成
    def myARIMA(mydata):
        arima = pm.auto_arima(mydata, error_action='ignore',
                              suppress_warnings=True, maxiter=5,
                              seasonal=True, m=12)
        tload_pred = arima.predict(n_periods=24)
        tload_pred = [float(x) for x in tload_pred]
        return tload_pred


    def calc_error(myRow):
        pred = myRow['arima_pred']
        real = myRow[tload].values
        return pred - real


    time_pred = time.time()
    df_arima = df_arima.withColumn('arima_pred',
                                   F.UserDefinedFunction(myARIMA, T.ArrayType(T.DoubleType()))(df_arima.features))
    arima_pred = df_arima.select(['MP_ID', 'STAT_CYCLE', 'arima_pred'] + tload).toPandas()
    error = arima_pred.apply(calc_error,axis=1).values
    mae = np.abs(error.mean(axis=0))
    rmse = np.sqrt((error**2).mean(axis=0).astype('float'))
    a = arima_pred['arima_pred'].values
    a = [x for x in a]
    ls_ar = ['arima_' + str(i) for i in range(24)]
    arima_pred[ls_ar] = a
    # arima_pred.to_csv('arima_pred.csv')
    arima_pred = arima_pred.drop(labels='arima_pred', axis=1)
    arima_pred = sqlContext.createDataFrame(arima_pred)
    arima_pred.printSchema()
    arima_pred = arima_pred.drop('STAT_CYCLE')
    arima_pred.write.mode('overwrite').csv('/electric-analyse/data/output/arima_pred.csv',header=True)

    time_pred = time.time() - time_pred
    evaluate_data['arima'] = {
        'rmse': list(rmse),
        'mae': list(mae),
        'avg_rmse': rmse.mean(),
        'avg_mae': mae.mean(),
        'time': time_pred / df_arima.count()
    }

    print(evaluate_data)


    with open("evaluate_data.json", "w") as fp:
        json.dump(evaluate_data, fp)
    
    print('OK')
    

if __name__ == "__main__":
    pred()
