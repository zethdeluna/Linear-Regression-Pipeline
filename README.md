# Linear Regression Pipeline
In this project, we'll be implementing the machine learning Linear Regression model to predict the values of houses in the housing market. 
Three functions were developed to make hyperparameter optimizations easier to play with:
1. `transform_features()`: cleans and transforms the features to make them usable for machine learning models
    * this includes: filling in missing values, converting data to the proper data type, and engineering new features that are mose useful for the linear regression model
2. `select_features()`: selects features that have the strongest correlations to the target feature
    * this process uses the `pandas.DataFrame.corr` function to generate a heatmap, then selects the features above a minimum correlation value (this value can be determined experimentally)
3. `train_and_test()`: this function takes in the dataframe obtained from `transform_features()`, the selected features from `select_features()`, and a number `k`
    * if `k = 1` the function implements a holdout validation model
    * if `k > 1` the function implements a cross validation model with `k` folds
