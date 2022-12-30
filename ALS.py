    ### Big Data Movie Recommendations using ALS recommendation model with hyperparameter tuning ###
import pyspark
import pandas as pd
import numpy as np
import math
from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.types import StructField, StructType, StringType, LongType, IntegerType, FloatType
from pyspark.sql.functions import col, column
from pyspark.sql.functions import expr
from pyspark.sql.functions import split
from pyspark.sql import Row
from pyspark.mllib.recommendation import ALS


ss=SparkSession.builder.appName("Recommendation Using Big Data").getOrCreate()
ss.sparkContext.setCheckpointDir("~/scratch")


ratingsbig_DF = ss.read.csv("./ratings-large (1).csv", header=True, inferSchema=True)
ratingsbig_DF.printSchema()


rating_schema = StructType([ StructField("UserID", IntegerType(), False ), StructField("MovieID", IntegerType(), True),              StructField("Rating", FloatType(), True ), StructField("RatingID", IntegerType(), True ),])


ratings_DF = ss.read.csv("./ratings-large.csv", schema= rating_schema, header=False, inferSchema=False)
ratings_DF.printSchema()


users_DF = ratings_DF.select("UserID")
users_DF.show(3)

UnqUsr_DF = users_DF.dropDuplicates()
UserCnt= UnqUsr_DF.count()

movies_DF = ratings_DF.select("MovieID")
UnqMovie_DF = movies_DF.dropDuplicates()
MovieCnt = UnqMovie_DF.count()
print("User Count =", UserCnt, "Movie Count =", MovieCnt)


# 259,137 users in the big movie dataset
# 39,443 number of movies in the big movie dataset

############################################################################################################################################
ratings2_DF = ratings_DF.sample(withReplacement=False, fraction=0.003, seed=19).select("UserID","MovieID","Rating")
ratings2_RDD= ratings2_DF.rdd

# split ratings2_DF into training, validation, and testing
training_RDD, validation_RDD, test_RDD = ratings2_RDD.randomSplit([3, 1, 1], 521)


# Prepare input (UserID, MovieID) for validation and for testing
training_input_RDD = training_RDD.map(lambda x: (x[0], x[1]) )
validation_input_RDD = validation_RDD.map(lambda x: (x[0], x[1]) ) 
testing_input_RDD = test_RDD.map(lambda x: (x[0], x[1]) )


model = ALS.train(training_RDD, 4, seed=41, iterations=30, lambda_=0.1)
training_prediction_RDD = model.predictAll(training_input_RDD).map(lambda x: ( (x[0], x[1]), x[2] ) )

training_prediction_RDD.take(3)
training_evaluation_RDD = training_RDD.map(lambda y: ( (y[0], y[1]), y[2]) ).join(training_prediction_RDD)
training_evaluation_RDD.take(3)
training_error = math.sqrt(training_evaluation_RDD.map(lambda z: (z[1][0] - z[1][1])**2).mean())
print(training_error)
# training error is 0.12554731880797582

validation_prediction_RDD = model.predictAll(validation_input_RDD).map(lambda x: ( (x[0], x[1]), x[2] ) )
validation_prediction_RDD.take(5)
validation_evaluation_RDD = validation_RDD.map(lambda y: ( (y[0], y[1]), y[2] )).join(validation_prediction_RDD)
validation_evaluation_RDD.take(5)
validation_error = math.sqrt(validation_evaluation_RDD.map(lambda z: (z[1][0] - z[1][1])**2).mean())
print(validation_error)
# validation error is huge at 3.888641479565653

# Reduce sparsity of dataset to reduce the error 
sampling_ratio = 0.003
sampled_User_DF = UnqUsr_DF.sample(withReplacement=False, fraction=sampling_ratio, seed=19)
sampled_rating_DF = sampled_User_DF.join(ratings_DF, "UserID", "inner")

sampled_User_DF.count()
sampled_rating_DF.count()

ratings2_DF = sampled_rating_DF.select("UserID","MovieID","Rating")
ratings2_RDD = ratings2_DF.rdd


# Split Systematic Sampled Data into Training Data, Validation Data, and Testing Data
training_RDD, validation_RDD, test_RDD = ratings2_RDD.randomSplit([3, 1, 1], 137)


training_input_RDD = training_RDD.map(lambda x: (x[0], x[1]) )
validation_input_RDD = validation_RDD.map(lambda x: (x[0], x[1]) ) 
testing_input_RDD = test_RDD.map(lambda x: (x[0], x[1]) )

model2 = ALS.train(training_RDD, 4, seed=37, iterations=30, lambda_=0.1)

training_prediction_RDD = model2.predictAll(training_input_RDD).map(lambda x: ( (x[0], x[1]), x[2] ) )
training_prediction_RDD.take(3)

training_evaluation_RDD = training_RDD.map(lambda y: ( (y[0], y[1]), y[2]) ).join(training_prediction_RDD)
training_evaluation_RDD.take(3)

training_error = math.sqrt(training_evaluation_RDD.map(lambda z: (z[1][0] - z[1][1])**2).mean())
print(training_error)
# training error now is 0.5776585296698834


validation_prediction_RDD = model2.predictAll(validation_input_RDD).map(lambda x: ( (x[0], x[1]), x[2] ) )
validation_prediction_RDD.take(5)

validation_evaluation_RDD = validation_RDD.map(lambda y: ( (y[0], y[1]), y[2]) ).join(validation_prediction_RDD)
validation_evaluation_RDD.take(5)

validation_error = math.sqrt(validation_evaluation_RDD.map(lambda z: (z[1][0] - z[1][1])**2).mean())
print(validation_error)
# validation error is now 0.9561062191200534


# ALS with all possible combinations of values for three hyperparameteres for the recommendation model
ratings4_DF = ratings_DF.select("UserID","MovieID","Rating")
ratings4_RDD = ratings4_DF.rdd

training_RDD, validation_RDD, test_RDD = ratings4_RDD.randomSplit([3, 1, 1], 137)

training_input_RDD = training_RDD.map(lambda x: (x[0], x[1]) )
validation_input_RDD = validation_RDD.map(lambda x: (x[0], x[1]) ) 
testing_input_RDD = test_RDD.map(lambda x: (x[0], x[1]) )

training_input_RDD.persist()
validation_input_RDD.persist()
testing_input_RDD.persist()

validation_input_RDD.persist()

hyperparams_eval_df = pd.DataFrame( columns = ['k', 'regularization', 'iterations', 'validation RMS', 'testing RMS'] )
index =0 
lowest_validation_error = float('inf')
iterations_list = [15, 30]
regularization_list = [0.1, 0.2]
rank_list = [4, 8, 12]
for k in rank_list:
    for regularization in regularization_list:
        for iterations in iterations_list:
            seed = 37
            model = ALS.train(training_RDD, k, seed=seed, iterations=iterations, lambda_=regularization)
            validation_prediction_RDD= model.predictAll(validation_input_RDD).map(lambda x: ( (x[0], x[1]), x[2] )   )
            validation_evaluation_RDD = validation_RDD.map(lambda y: ( (y[0], y[1]), y[2]) ).join(validation_prediction_RDD)
            # Calculate RMSE
            validation_error = math.sqrt(validation_evaluation_RDD.map(lambda z: (z[1][0] - z[1][1])**2).mean())
            hyperparams_eval_df.loc[index] = [k, regularization, iterations, validation_error, float('inf')]
            index = index + 1
            if validation_error < lowest_validation_error:
                best_k = k
                best_regularization = regularization
                best_iterations = iterations
                best_index = index - 1
                lowest_validation_error = validation_error
print('The best rank k is ', best_k, ', regularization = ', best_regularization, ', iterations = ',      best_iterations, '. Validation Error =', lowest_validation_error)


# Use testing data to evaluate model using the best hyperparameters 

seed = 37
model = ALS.train(training_RDD, best_k, seed=seed, iterations=best_iterations, lambda_=best_regularization)
testing_prediction_RDD=model.predictAll(testing_input_RDD).map(lambda x: ((x[0], x[1]), x[2]))
testing_evaluation_RDD= test_RDD.map(lambda x: ((x[0], x[1]), x[2])).join(testing_prediction_RDD)
testing_error = math.sqrt(testing_evaluation_RDD.map(lambda x: (x[1][0]-x[1][1])**2).mean())
print('The Testing Error for rank k =', best_k, ' regularization = ', best_regularization, ', iterations = ',       best_iterations, ' is : ', testing_error)

print(best_index)

hyperparams_eval_df.loc[best_index]=[best_k, best_regularization, best_iterations, lowest_validation_error, testing_error]

schema3= StructType([ StructField("k", FloatType(), True), StructField("regularization", FloatType(), True ),                   StructField("iterations", FloatType(), True), StructField("Validation RMS", FloatType(), True), StructField("Testing RMS", FloatType(), True)])


HyperParams_Tuning_DF = ss.createDataFrame(hyperparams_eval_df, schema3)


output_path = "./results"
HyperParams_Tuning_DF.rdd.saveAsTextFile(output_path)

# best hyper-parameters: k = 12, regularization = 0.1, iterations = 30
# validation error = 0.8174323040709253


ss.stop()





