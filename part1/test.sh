#!/bin/bash
../start.sh
source ../env.sh
hdfs dfsadmin -safemode leave
/usr/local/hadoop/bin/hdfs dfs -rm -r /part1/input/
/usr/local/hadoop/bin/hdfs dfs -rm -r /part1/output
/usr/local/hadoop/bin/hdfs dfs -mkdir -p /part1/input/
/usr/local/hadoop/bin/hdfs dfs -copyFromLocal ../nba-data/shot_logs.csv /part1/input/
/usr/local/spark/bin/spark-submit --master=spark://$SPARK_MASTER:7077 ./nba-kmeans.py hdfs://$SPARK_MASTER:9000/part1/input/
/../stop.sh
