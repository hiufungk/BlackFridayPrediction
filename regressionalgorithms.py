from __future__ import division  # floating point division
import numpy as np
import math
import dataloader as dtl
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error, make_scorer
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_validate
from sklearn.neural_network import MLPRegressor

trainsize = 400000 
testsize = 100000 

trainset, testset = dtl.load_black_friday(trainsize,testsize)

Xtrain = trainset[0]
ytrain = trainset[1]

Xtest = testset[0]
ytest = testset[1]
np.save("ytest", ytest)

number_of_fold = 5

RidgeAlgs = {
    "Ridge_0.1": linear_model.Ridge(alpha=0.1),                
    "Ridge_1.0": linear_model.Ridge(alpha=1.0),                
    "Ridge_10.0": linear_model.Ridge(alpha=10.0)                
}

min_avg_error = np.inf
best_ridge = None
best_ridge_name = ""
for learner_name in RidgeAlgs:
    print(learner_name)
    scores = cross_validate(RidgeAlgs[learner_name], Xtrain, y=ytrain, scoring=make_scorer(mean_absolute_error), cv=number_of_fold)
    avg_error = np.mean(scores["test_score"])
    if(avg_error < min_avg_error):
        min_avg_error = avg_error
        best_ridge = RidgeAlgs[learner_name]
        best_ridge_name = learner_name
model_ridge = best_ridge.fit(Xtrain,ytrain)
predictions_ridge = model_ridge.predict(Xtest)
np.save(best_ridge_name, predictions_ridge)
r2Ridge = model_ridge.score(Xtest, ytest)
maeRidge = mean_absolute_error(ytest, predictions_ridge)
print("r2Ridge " + str(r2Ridge))
print("maeRidge " + str(maeRidge))


RandomForestAlgs = {
    "RandomForestRegressor_80": RandomForestRegressor(n_estimators=80),                
    "RandomForestRegressor_20": RandomForestRegressor(n_estimators=20),
    "RandomForestRegressor_50": RandomForestRegressor(n_estimators=50)
}

min_avg_error = np.inf
best_random_forest = None
best_random_forest_name = ""
for learner_name in RandomForestAlgs:
    print(learner_name)
    scores = cross_validate(RandomForestAlgs[learner_name], Xtrain, y=ytrain, scoring=make_scorer(mean_absolute_error), cv=number_of_fold)
    avg_error = np.mean(scores["test_score"])
    if(avg_error < min_avg_error):
        min_avg_error = avg_error
        best_random_forest = RandomForestAlgs[learner_name]
        best_random_forest_name = learner_name
model_random_forest = best_random_forest.fit(Xtrain,ytrain)
predictions_random_forest = model_random_forest.predict(Xtest)
np.save(best_random_forest_name, predictions_random_forest)
r2RFR = best_random_forest.score(Xtest, ytest)
maeRFR = mean_absolute_error(ytest, predictions_random_forest)
print("r2RFR " + str(r2RFR))
print("maeRFR " + str(maeRFR))

NNAlgs = {}
num_node_list = [4, 8, 16]
step_size_list = [0.1, 0.01, 0.001]
for num_node in num_node_list:
    for step_size in step_size_list:
        param = "NN_" + str(num_node) + "_" + str(step_size)
        NNAlgs[param] = MLPRegressor(hidden_layer_sizes=(num_node,), learning_rate_init=step_size)
min_avg_error = np.inf
best_NN = None
bestNN_name = ""
for learner_name in NNAlgs:
    print(learner_name)
    scores = cross_validate(NNAlgs[learner_name], Xtrain, y=ytrain, scoring=make_scorer(mean_absolute_error), cv=number_of_fold)
    avg_error = np.mean(scores["test_score"])
    if(avg_error < min_avg_error):
        min_avg_error = avg_error
        best_NN = NNAlgs[learner_name]
        bestNN_name = learner_name
model_NN = best_NN.fit(Xtrain,ytrain)
predictions_NN = model_NN.predict(Xtest)
np.save(bestNN_name, predictions_NN)

r2NN = best_NN.score(Xtest, ytest)
maeNN = mean_absolute_error(ytest, predictions_NN)
print("r2NN " + str(r2NN))
print("maeNN " + str(maeNN))





