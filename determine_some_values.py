import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn import ensemble
from sklearn.metrics import mean_absolute_error
from sklearn.externals import joblib



#our purpose here is to determine which combination
# of GradienBoostingRegression params are the best



houses=pd.read_csv('ml_house_data_set.csv')



#i deleted those attribut coz i found them unusefull
del houses['house_number']
del houses['unit_number']
del houses['street_name']
del houses['zip_code']

#I have in garage_type  column three type  attached, detached and none
#get_dummies create new three column is_garage_type_attached, is garage_type_detached and is_garage_type_none
#with boolean value true or false, that's make it easy for training

houses_af=pd.get_dummies(houses, columns=['garage_type','city'])

#delete sale_price from  train data  let only the predictor value

del houses_af['sale_price']

#make it array without attribut name
 x=houses_af.values

#get only the target value for train
y=houses['sale_price'].values

#split the data to training data and testing one
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)

model=ensemble.GradientBoostingRegressor()

# those attrs are the responsible for the performance of  our  gradientBoosting function
# now we only determine the right combination for the best model

#
param_grid = {
    'n_estimators': [500, 1000, 3000],
    'max_depth': [4, 6],
    'min_samples_leaf': [3, 5, 9, 17],
    'learning_rate': [0.1, 0.05, 0.02, 0.01],
    'max_features': [1.0, 0.3, 0.1],
    'loss': ['ls', 'lad', 'huber']
}

# define the grid search we want to run. Run it with 4 cpus in parallel.
gs_cv = GridSearchCV(model, param_grid, n_jobs=4)

# Run the grid search - on only the training data!
gs_cv.fit(x_train, y_train)

# Print the params that gave us the best result!
print(gs_cv.best_params_)




