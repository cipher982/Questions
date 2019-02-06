#import findspark
#findspark.init()
import re
import pandas as pd
from datetime import datetime
import random
from word2number import w2n

from pyspark.sql.functions import udf, monotonically_increasing_id
from pyspark.sql import DataFrameStatFunctions as statFunc
import pyspark.sql.functions as func
from pyspark import SparkContext, SparkConf, SQLContext
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("capgeminiApp").getOrCreate()
sc = SparkContext()
sqlContext = SQLContext(sc)

# Step 1
df_pandas = pd.DataFrame({"Input 1" : ['Green','Green','Blue','Red','Yellow','Red','Red','Blue'],
                          "Input 2" : ['1','0','4','5','7','9','6','5'],
                          "Input 3" : ['1.3','1.445','1.2','1.3','1.325','1.4','1.72158','1'],
                          "Description" : ['DescriptionONE','DescriptionONE','DescriptionTWO','Description THREE','Description Four','DescriptionONE','Description THREE','DescriptionONE']   
                         })
df1 = sqlContext.createDataFrame(df_pandas)

# Step 2
def step_2a(string):
    return re.sub('Description THREE', 'DescriptionTHREE', string)
step_2a=func.udf(step_2a)
def step_2b(string):
    return re.sub('Description Four', 'DescriptionFOUR', string)
step_2b=func.udf(step_2b)
df2 = df1.withColumn('Description', step_2b(step_2a(df1['Description'])))
df2.show()

# Step 3
df3 = df2.withColumn('Input 3', df2['Input 3'].cast(DoubleType()))
df3 = df3.withColumn('Input 3', func.format_number(df3['Input 3'], 4))
df3 = df3.withColumn('Input 3', df3['Input 3'].cast(StringType()))
df3.show()

# Step 4
df_pandas = pd.DataFrame({'col1' : ['Green','Yellow','Red','Blue'],
                          'col2' : ['Night','Morning','Afternoon','Evening']   
                         })
df4 = sqlContext.createDataFrame(df_pandas)
df4.show()

def step_11(df1, df2):
    try:
        # Step 5 
        df2 = df2.withColumnRenamed("col1", "Input 1")\
                 .withColumnRenamed("col2", "Day Period")
    except:
        print("Code failed on step 5")
    
    # Step 6
    try:
        df = df1.join(df2, "Input 1", "left")
    except:
        print("Code failed on step 6")
    
    # Step 7 
    try:
        def step_7(idx):
            random.seed(idx)
            test = date.fromordinal(random.randint(0, date.today().toordinal()))
            return str(test)
        step_7 = func.udf(step_7)
        df = df.withColumn("temp_index", monotonically_increasing_id())
        df = df.withColumn("Date", step_7(df['temp_index'])).drop("temp_index")
    except:
        print("Code failed on step 7")
    
    # Step 8
    try:
        df = df.filter((func.col('Input 3') >= 1.31))\
               .filter((func.col('Input 1') == 'Red') | (func.col('Input 1') == 'Green'))
    except:
        print("Code failed on step 8")
    
    # Step 9
    try:
        df = df.withColumn('temp_date', func.unix_timestamp(df["Date"].cast(DateType())))
        middle_date = statFunc(df).approxQuantile("temp_date", [0.5], 0)
        df = df.withColumn("flag", func.when((df["temp_date"] > middle_date[0]) & \
                                             (df["Input 2"] > 1),
                                          value=1).otherwise(0)).drop('temp_date')
    except:
        print('Code failed on step 9')
    
    # Step 10
    try:
        result_list = {}
        for i,row in enumerate(df.rdd.collect()):
            input3 = row['Input 3']
            desc = w2n.word_to_num(row['Description'].split('Description')[1])
            input2_min = df.agg({"Input 2": "min"}).collect()[0]['min(Input 2)']
            try:
                result = (float(input3) + float(desc)) / (float(input2_min))
            except ZeroDivisionError:
                result = None
            result_df = sqlContext.createDataFrame([result], "string").toDF("result_{}".format(i))
            result_df.show()
            result_list['result_{}'.format(i)] = result_df
        return result_list
    except:
        print("Code failed on step 10")
        return None

final = step_11(df1=df3, df2=df4)






