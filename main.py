# -*- coding: utf-8 -*- 
# @Time : 2023/3/29 14:51 
# @Author : zhangqinming
# @File : main.py
from pyspark.sql.session import SparkSession
from model_train import train
from model_pred import pred
if __name__ == '__main__':
    spark = SparkSession.builder.master('spark://Master032004134:7077').appName("electric_analyze").getOrCreate()
    spark.sparkContext.setLogLevel("Error")
    train(spark)
    pred(spark)
