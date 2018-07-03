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

# Fit regression model
model = ensemble.GradientBoostingRegressor(
    n_estimators=1000,
    learning_rate=0.1,
    max_depth=6,
    min_samples_leaf=9,
    max_features=0.1,
    loss='huber'
)
model.fit(X_train, y_train)

# Save the trained model to a file so we can use it in other programs
joblib.dump(model, 'trained_model.pkl')

# Find the error rate on the training set
mse = mean_absolute_error(y_train, model.predict(x_train))
print("Training Set Mean Absolute Error: %.4f" % mse)

# Find the error rate on the test set
mse = mean_absolute_error(y_test, model.predict(x_test))
print("Test Set Mean Absolute Error: %.4f" % mse)

