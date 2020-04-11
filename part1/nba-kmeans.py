# -*- coding: utf-8 -*-
#
# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.  See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from __future__ import print_function
import sys
from pyspark.sql import SparkSession,SQLContext
from pyspark.sql.functions import col, greatest, least, when, mean
import random as rd
import numpy as np

reload(sys)
sys.setdefaultencoding('utf8')
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: k-means <file>", file=sys.stderr)
        sys.exit(-1)

    spark = SparkSession\
        .builder\
        .appName("nbatest")\
        .getOrCreate()

    #Data loading and preprocessing
    data = spark.read.csv(sys.argv[1],header =True)
    df = data.rdd.map(lambda x: (x.player_name, x.SHOT_RESULT, x.SHOT_CLOCK, x.SHOT_DIST, x.CLOSE_DEF_DIST)).toDF() #extract only player name, shot result, shot clock, shot dist, close def dist
    df = df.select(df._1, df._2, df._3.cast("float"), df._4.cast("float"), df._5.cast("float"))
    df = df.dropna() #drop all empty values

    #construction of initial centroids using K means ++
    X=np.array(df.select(df._3, df._4, df._5).collect())
    m=X.shape[0] #number of training examples
    n=X.shape[1] #number of features
    n_iter=10
    i=rd.randint(0,X.shape[0])
    Centroid=np.array([X[i]])
    K=4
    for k in range(1,K):
        D=np.array([]) 
        for x in X:
            D=np.append(D,np.min(np.sum((x-Centroid)**2)))
        prob=D/np.sum(D)
        cummulative_prob=np.cumsum(prob)
        r=rd.random()
        i=0
        for j,p in enumerate(cummulative_prob):
            if r<p:
                i=j
                break
        Centroid=np.append(Centroid,[X[i]],axis=0)
    #centroids are [shotclock,shotdist,closedefdist]
    centroid1 = (Centroid[0])
    centroid2 = (Centroid[1])
    centroid3 = (Centroid[2])
    centroid4 = (Centroid[3])

    for i in range(n_iter):
        #distance computations
        df1 = df.withColumn('distZone1',((df._3 - centroid1[0])**2 + (df._4 - centroid1[1])**2 + (df._5 - centroid1[2])**2)**0.5 )
        df2 = df1.withColumn('distZone2',((df._3 - centroid2[0])**2 + (df._4 - centroid2[1])**2 + (df._5 - centroid2[2])**2)**0.5 )
        df3 = df2.withColumn('distZone3',((df._3 - centroid3[0])**2 + (df._4 - centroid3[1])**2 + (df._5 - centroid3[2])**2)**0.5 )
        df4 = df3.withColumn('distZone4',((df._3 - centroid4[0])**2 + (df._4 - centroid4[1])**2 + (df._5 - centroid4[2])**2)**0.5 )
        df5 = df4.select(df4._1 \
			   , df4._2 \
			   , df4._3 \
			   , df4._4 \
			   , df4._5 \
			   , df4.distZone1 \
			   , df4.distZone2 \
			   , df4.distZone3 \
			   , df4.distZone4 \
			   , least("distZone1","distZone2","distZone3","distZone4").alias('minDis'))
        #assigning clusters
        df5 = df5.withColumn('prediction',when(df5.minDis == df5.distZone1,"Zone1") \
					     .when(df5.minDis == df5.distZone2,"Zone2") \
					     .when(df5.minDis == df5.distZone3,"Zone3") \
					     .otherwise("Zone4"))
        #creating SQLContext and registering previous df as a table to build new centroids
        sqlcontext = SQLContext(spark)
        sqlcontext.registerDataFrameAsTable(df5,"df")
        #building new centroid1
        Zone1df = sqlcontext.sql("SELECT  * FROM df WHERE df.prediction = 'Zone1'")
        Zone1dfmeans = Zone1df.select(mean(col("_3")).alias("shotclockmean") \
			      , mean(col("_4")).alias("shotdistmean") \
			      , mean(col("_5")).alias("closedefmean")).collect()
        shotclock = Zone1dfmeans[0]["shotclockmean"]
        shotdist = Zone1dfmeans[0]["shotdistmean"]
        closedef = Zone1dfmeans[0]["closedefmean"]
        centroid1 = np.array([shotclock,shotdist,closedef])
        #building new centroid2
        Zone2df = sqlcontext.sql("SELECT  * FROM df WHERE df.prediction = 'Zone2'")
        Zone2dfmeans = Zone2df.select(mean(col("_3")).alias("shotclockmean") \
			      , mean(col("_4")).alias("shotdistmean") \
			      , mean(col("_5")).alias("closedefmean")).collect()
        shotclock = Zone2dfmeans[0]["shotclockmean"]
        shotdist = Zone2dfmeans[0]["shotdistmean"]
        closedef = Zone2dfmeans[0]["closedefmean"]
        centroid2 = np.array([shotclock,shotdist,closedef])
        #building new centroid3
        Zone3df = sqlcontext.sql("SELECT  * FROM df WHERE df.prediction = 'Zone3'")
        Zone3dfmeans = Zone3df.select(mean(col("_3")).alias("shotclockmean") \
			      , mean(col("_4")).alias("shotdistmean") \
			      , mean(col("_5")).alias("closedefmean")).collect()

        shotclock = Zone3dfmeans[0]["shotclockmean"]
        shotdist = Zone3dfmeans[0]["shotdistmean"]
        closedef = Zone3dfmeans[0]["closedefmean"]
        centroid3 = np.array([shotclock,shotdist,closedef])
        #building new centroid 4
        Zone4df = sqlcontext.sql("SELECT  * FROM df WHERE df.prediction = 'Zone4'")
        Zone4dfmeans = Zone4df.select(mean(col("_3")).alias("shotclockmean") \
			      , mean(col("_4")).alias("shotdistmean") \
			      , mean(col("_5")).alias("closedefmean")).collect()

        shotclock = Zone4dfmeans[0]["shotclockmean"]
        shotdist = Zone4dfmeans[0]["shotdistmean"]
        closedef = Zone4dfmeans[0]["closedefmean"]
        centroid4 = np.array([shotclock,shotdist,closedef])

    #counting how many ocurrences of each combination of player, outcome and zone

    df6 = sqlcontext.sql("""SELECT df._1
                                    , df._2
                                    , df.prediction
                                    , count(minDis) count
                                 FROM df
                             GROUP BY df._1
                                    , df._2
                                    , df.prediction""")

    #joining the table with itself, to have missed and made shots in a single row and to calculate hit rates for each zone
    sqlcontext.registerDataFrameAsTable(df6,"df")
    df7 = sqlcontext.sql("""SELECT A._1 AS playername
			       	    , A._2 AS made
			      	    , A.prediction
				    , A.count
				    , B._2 AS missed
				    , B.count
				    , A.count/(B.count + A.count) AS hitrate
			         FROM (
				    SELECT *
			              FROM df
			             WHERE df._2 = 'made') A
			   INNER JOIN (
			       	    SELECT *
			              FROM df
			             WHERE df._2 = 'missed') B
			           ON A._1 = B._1
			          AND A.prediction = B.prediction""")

    #creating a table with just 1 row per player; each row holds all the neccesary information
    sqlcontext.registerDataFrameAsTable(df7,"df")
    df8 = sqlcontext.sql("""SELECT df.playername
			            , MAX(CASE WHEN df.prediction = 'Zone1' THEN df.hitrate END) AS Zone1hitrate
		  	            , MAX(CASE WHEN df.prediction = 'Zone2' THEN df.hitrate END) AS Zone2hitrate
			            , MAX(CASE WHEN df.prediction = 'Zone3' THEN df.hitrate END) AS Zone3hitrate
		    	            , MAX(CASE WHEN df.prediction = 'Zone4' THEN df.hitrate END) AS Zone4hitrate
			         FROM df
		             GROUP BY df.playername""")

    #adding a column with the highest hitrate to the previous table
    df9 = df8.select(df8.playername \
			 , df8.Zone1hitrate \
			 , df8.Zone2hitrate \
			 , df8.Zone3hitrate \
			 , df8.Zone4hitrate \
			 , greatest("Zone1hitrate","Zone2hitrate","Zone3hitrate","Zone4hitrate").alias("besthitrate"))

    #using the besthitrate column to determine the best zone for each player
    df10 = df9.withColumn("bestzone",when(df9.Zone1hitrate == df9.besthitrate, "Zone1") \
					 .when(df9.Zone2hitrate == df9.besthitrate, "Zone2") \
					 .when(df9.Zone3hitrate == df9.besthitrate, "Zone3") \
					 .when(df9.Zone4hitrate == df9.besthitrate, "Zone4"))

    sqlcontext.registerDataFrameAsTable(df10,"df")
    print("The NBA player have been classified into four confortable zones, with the following structure: [shotclock,shotdist,closedefdist]")
    print("The four zones are:")
    print("Zone1: %s" % (centroid1))
    print("Zone2: %s" % (centroid2))
    print("Zone3: %s" % (centroid3))
    print("Zone4: %s" % (centroid4))
    print("The best zones for James Harden, Chris Paul, Stephen Curry and Lebron James are:")
    df = sqlcontext.sql("""SELECT playername, besthitrate, bestzone
		  	          FROM df
				 WHERE df.playername IN ('james harden'
							   , 'chris paul'
							   , 'stephen curry'
							   , 'lebron james')""")

    df.show()

    spark.stop()
