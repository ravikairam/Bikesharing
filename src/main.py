import sys
sys.path.append('.')
from prettytable import PrettyTable
import pandas as pd


import random
import pathlib

from src.analysis import analyse, model_selection_analysis
from src.evaluation import evaluate
from src.preprocessing import split_and_select_features
from src.training import train_random_forest



def main():
    # Make results reproducible
    random.seed(100)
    base_path = pathlib.Path(__file__).parent.resolve()
    data = pd.read_csv(str(base_path) + "/data/hour.csv")
    features, number_features, target, test, train, val = split_and_select_features(data)

    print(data[number_features].describe())

    analyse(features, data, number_features, target, train)

    x_train, x_val, y_train, y_val = model_selection_analysis(features, target, test, train, val)

    table = PrettyTable()
    table.field_names = ["Model", "Dataset", "MSE", "MAE", 'RMSLE', "R² score"]
    model = train_random_forest(x_train, y_train)

    evaluate(model, table, x_train, y_train, 'training')
    evaluate(model, table, x_val, y_val, 'validation')

    print(table)


if __name__ == "__main__":
    main()
