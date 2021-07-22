import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from prettytable import PrettyTable
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import SGDRegressor, Lasso, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR


def analyse(features, data, number_features, target, train):
    # Check null
    print(data.isnull().any())
    # Correlation matrix
    matrix = train[number_features + target].corr()
    heat = np.array(matrix)
    heat[np.tril_indices_from(heat)] = False
    fig, ax = plt.subplots()
    fig.set_size_inches(20, 10)
    sns.set(font_scale=1.0)
    sns.heatmap(matrix, mask=heat, vmax=1.0, vmin=0.0, square=True, annot=True, cmap="Reds")
    features.remove('atemp')


def model_selection_analysis(features, target, test, train, val):
    x_train = train[features].values
    y_train = train[target].values.ravel()
    val = val.sort_values(by=target)
    x_val = val[features].values
    y_val = val[target].values.ravel()
    x_test = test[features].values
    table = PrettyTable()
    table.field_names = ["Model", "Mean Squared Error", "R² score"]
    models = [
        SGDRegressor(max_iter=1000, tol=1e-3),
        Lasso(alpha=0.1),
        Ridge(alpha=.5),
        SVR(gamma='auto', kernel='linear'),
        SVR(gamma='auto', kernel='rbf'),
        RandomForestRegressor(random_state=0, n_estimators=300)
    ]
    for model in models:
        model.fit(x_train, y_train)
        y_res = model.predict(x_val)

        mse = mean_squared_error(y_val, y_res)
        score = model.score(x_val, y_val)

        table.add_row([type(model).__name__, format(mse, '.2f'), format(score, '.2f')])
    print(table)
    return x_train, x_val, y_train, y_val