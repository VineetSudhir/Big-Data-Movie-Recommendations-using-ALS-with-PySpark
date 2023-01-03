# Big-Data-Movie-Recommendations-using-ALS-with-PySpark
Big Data Movie Recommendations using ALS recommendation model with hyperparameter tuning

Dataset was too big to upload @ 163 MB

Reduced Sparsity of dataset by sampling from Big dataset.
Test among multiple combinations of values for the hyperparameters.
Utilized persist, tuned the hyperparameters, and reduced the sparsity of the dataset to lower the validation error.

Validation error using big dataset is approximately 389% 
Validation error after reducing sparsity of big dataset is approximately 95.6%
Validation error after choosing best hyperparameters is approximately 81.7% 

best hyper-parameters: k = 12, regularization = 0.1, iterations = 30, validation error = 0.8174323040709253
