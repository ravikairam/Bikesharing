from sklearn.ensemble import RandomForestRegressor
import pickle
import pathlib
base_path = str(pathlib.Path(__file__).parent.resolve())


def train_random_forest(x_train, y_train, save_file=True):
    model = RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=None,
                                  max_features='auto', max_leaf_nodes=None,
                                  min_impurity_decrease=0.0, min_impurity_split=None,
                                  min_samples_leaf=1, min_samples_split=4,
                                  min_weight_fraction_leaf=0.0, n_estimators=200, n_jobs=None,
                                  oob_score=False, random_state=None, verbose=0, warm_start=False)
    model.fit(x_train, y_train)
    filename = base_path + '/models/random_forest.sav'
    if save_file:
        pickle.dump(model, open(filename, 'wb'))
    return model
