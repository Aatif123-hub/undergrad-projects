# Sources used
# https://matplotlib.org/2.0.2/api/pyplot_api.html
# https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.iloc.html
# https://www.geeksforgeeks.org/random-forest-regression-in-python/#
# https://scikit-learn.org/0.16/modules/generated/sklearn.ensemble.RandomForestRegressor.html


import pandas
import matplotlib.pyplot as pyplot

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


dataframe = pandas.read_csv('Census.csv', encoding='latin_1')
# Used to omit columns from features
# Poverty (17) is the target, ChildPoverty (18) is influenced by Poverty and thus will skew the data
# County Id is ignored because we're already using County
ignoredFeatures = (0, 17, 18)

featureIndices = [i for i in range(3, 17)] + [i for i in range(19, 37)]

ignoredColumns = ['CountyId', 'Poverty', 'ChildPoverty']

features = dataframe.columns.copy().delete(ignoredFeatures)

label_encoder = LabelEncoder()

X_Cat = dataframe.iloc[:,1:3].apply(label_encoder.fit_transform)
X_Num = dataframe.iloc[:,featureIndices].values
X = pandas.concat([X_Cat, pandas.DataFrame(X_Num)], axis=1).values

y = dataframe.iloc[:,17].values

# https://stackoverflow.com/questions/14940743/selecting-excluding-sets-of-columns-in-pandas
x_categorical = dataframe[dataframe.columns[~dataframe.columns.isin(ignoredColumns)]].select_dtypes(include=['object']).apply(label_encoder.fit_transform)
x_numerical = dataframe[dataframe.columns[~dataframe.columns.isin(ignoredColumns)]].select_dtypes(exclude=['object']).values
x = pandas.concat([x_categorical, pandas.DataFrame(x_numerical)], axis=1).values

regressor = Pipeline([("imputer", SimpleImputer(missing_values=0, strategy="mean")),
    ("forest", RandomForestRegressor(n_estimators=10, random_state=0, oob_score=True))])

regressor.fit(x, y)

predictions = regressor.predict(X)

print(f"MAE: {mean_absolute_error(y, predictions)}")
print(f"MSE: {mean_squared_error(y, predictions)}")
print(f"R2: {r2_score(y, predictions)}")

pyplot.pie(regressor.named_steps["forest"].feature_importances_, labels=features, labeldistance=1.8)
pyplot.title('Poverty Regression Analysis Results')
pyplot.show()