# House Price Regression
 [Boston house prices dataset](https://www.kaggle.com/fedesoriano/the-boston-houseprice-data) has been used for this regression example. 6 algorithms have been selected and as metrics mean averaged error, mean squared error, r score have been used.
 
 ## Algorithms
 
 I have selected 6 algorithms which are linear regression, stochastic gradient descent (SGDRegressor), ridge regression, lasso regression, elastic net regression and polynomial regression.
 
 ## Data Preprocessing
 First we need to check if there's a missing value or not. After this done we should start with feature selection. I used correlation map to determine correlated features and these features eliminated.

![image](https://user-images.githubusercontent.com/43848140/151824949-a270e831-b691-4b1b-be7f-a374e7177fd7.png)

Next we need to make data normalization before we train our dataset. This can be done in two lines of code.

```
standardizer = StandardScaler()
X = standardizer.fit_transform(X)
```
## Train and Test Split

Then we need to split dataset as train and test, I have decided to use %30 test data, %70 training data. Also set the random_state parameter to 0 for now.

```
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
```

I created dictionaries for algorithms and metrics in order to run all of them in just one for loop. Except for polynomial regression the other algorithms don't need a parameter.

```
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
```

Polynomial regression has a special case because it uses LinearRegression() function so because of that I need to include if else statement.

## Results
To visualize this I used PCA(Principle Component Analysis) and reduced dataset dimensionality to 2D. As you can see the figure below for linear regression, except for the extreme values it predicted really well. And the other algorithms have almost the same figure like linear regression has.

![image](https://user-images.githubusercontent.com/43848140/151827098-31bc638b-87ea-4c6e-bc65-ac3f681edf6a.png)

But polynomial regression had better prediction than the others. This algorithm predicted extreme values much better than the others.

![image](https://user-images.githubusercontent.com/43848140/151827299-02fb9416-a5dd-4699-a98b-580637f7d2bc.png)

And here are the performance metrics results:

| Algorithm             | MAE           | R2             | MSE         | 
| --------------------- | ------------- | ---------------| ----------- |
| Linear Regression     | 3.609904      | 0.529101       | 27.195966   |
| SGD                   | 3.626181      | 0.516821       | 27.699630   |
| Ridge Regression      | 3.607967      | 0.509841       | 27.699272   | 
| Lasso Regression      | 3.608212      | 0.520186       | 27.452436   | 
| Elastic Net           | 3.606997      | 0.511639       | 27.671762   | 
| Polynomial Regression | 3.054786      | 0.698639       | 29.037945   | 

## Conclusion
As you can see from the results most of the algorithms performed nearly equal except polynomial regression. That's because polynomial regression algorithm creates high order features and can be able to better predict these values. You can tune some parameters and might be able to get better results.





