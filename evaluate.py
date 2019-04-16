import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn import preprocessing
from sklearn.metrics import mean_absolute_error


def get_abs_error(predictions):
    return np.abs(np.subtract(ytest, predictions))

ytest = np.load("ytest.npy")

results = []
names = ["Ridge", "Random Forest", "Neural Network"]

#get ridge abs errors
ridge_predictions = np.load("Ridge_10.0.npy")
ridge_abs_error = get_abs_error(ridge_predictions)
results.append(ridge_abs_error)

#get random forest abs errors
random_forest_predictions = np.load("RandomForestRegressor_80.npy")
random_forest_abs_error = get_abs_error(random_forest_predictions)
results.append(random_forest_abs_error)

#get neural network abs errors
nn_predictions = np.load("NN_16_0.1.npy")
nn_abs_error = get_abs_error(nn_predictions)
results.append(nn_abs_error)

null_hyp = "null hypothesis: ridge, random forest are same"
print(null_hyp)
print("ridge_mean_abs_error: " + str(mean_absolute_error(ytest, ridge_predictions)))
print("random_forest_mean_abs_error: " + str(mean_absolute_error(ytest, random_forest_predictions)))
rv = stats.ttest_rel(ridge_abs_error,random_forest_abs_error)
p_value = rv[1]
print("pvalue 1: " + str(p_value))
if(p_value < 0.05):
    print("reject null hypothesis")
else:
    print("not reject null hypothesis")

print()

null_hyp = "null hypothesis: ridge, NN are same"
print(null_hyp)
print("ridge_mean_abs_error: " + str(mean_absolute_error(ytest, ridge_predictions)))
print("nn_mean_abs_error: " + str(mean_absolute_error(ytest, nn_predictions)))
rv = stats.ttest_rel(ridge_abs_error,nn_abs_error)
p_value = rv[1]
print("pvalue 1: " + str(p_value))
if(p_value < 0.05):
    print("reject null hypothesis")
else:
    print("not reject null hypothesis")

print()

#null hypothesis: ridge, random forest are same
null_hyp = "null hypothesis: NN, random forest are same"
print(null_hyp)
print("nn_mean_abs_error: " + str(mean_absolute_error(ytest, nn_predictions)))
print("random_forest_mean_abs_error: " + str(mean_absolute_error(ytest, random_forest_predictions)))
rv = stats.ttest_rel(nn_abs_error,random_forest_abs_error)
p_value = rv[1]
print("pvalue 1: " + str(p_value))
if(p_value < 0.05):
    print("reject null hypothesis")
else:
    print("not reject null hypothesis")

#generate box plots
fig = plt.figure()
fig.suptitle("Algorithm Comparison")
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()