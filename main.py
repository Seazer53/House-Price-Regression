import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, SGDRegressor, RidgeCV, LassoCV, ElasticNetCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures


def plot_data(y_test, y_pred, key):
    x_ax = range(len(y_test))
    plt.plot(x_ax, y_test, label="original")
    plt.plot(x_ax, y_pred, label="predicted")
    plt.title("Original and predicted data for " + key)
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.legend(loc='best', fancybox=True, shadow=True)
    plt.grid(True)
    plt.show()


def print_metrics(mae, r2, mse, real, predicted, key):

    # Calculate MAE, R2, MSE Metrics
    mae[key] = mean_absolute_error(predicted, real)
    r2[key] = r2_score(predicted, real)
    mse[key] = mean_squared_error(predicted, real)


data = pd.read_csv("boston.csv")

pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)

# print(data.head())

y = data["MEDV"]
X = data.drop("MEDV", axis=1)

# print(y.head())
# print(X.head())

# print(X.describe(include=['float']).T)

# Correlation map
f, ax = plt.subplots(figsize=(18, 18))
sns.heatmap(X.corr(), annot=True, linewidths=.5, fmt='.1f', ax=ax)
plt.show()

standardizer = StandardScaler()
X = standardizer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

models = {'Linear Regression': LinearRegression(), 'Stochastic Gradient Descent': SGDRegressor(),
          'Ridge Regression': RidgeCV(), 'Lasso Regression': LassoCV(),
          'Elastic Net Regression': ElasticNetCV(), 'Polynomial Regression': PolynomialFeatures(degree=2)}

mae, r2, mse = {}, {}, {}

for key in models.keys():
    if key == "Polynomial Regression":
        X_train_pol_2 = models[key].fit_transform(X_train)
        X_test_pol_2 = models[key].transform(X_test)

        lr = LinearRegression()

        lr.fit(X_train_pol_2, y_train)
        pol_pred = lr.predict(X_test_pol_2)

        print_metrics(mae, r2, mse, y_test, pol_pred, key)
        plot_data(y_test, pol_pred, key)

    else:
        # Fit the regression model
        models[key].fit(X_train, y_train)

        # Regression
        regression = models[key].predict(X_test)
        print_metrics(mae, r2, mse, y_test, regression, key)
        plot_data(y_test, regression, key)

df_model = pd.DataFrame(index=models.keys(), columns=['MAE', 'R2', 'MSE'])
df_model['MAE'] = mae.values()
df_model['R2'] = r2.values()
df_model['MSE'] = mse.values()

print(df_model)
