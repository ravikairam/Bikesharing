import os
import pickle
import argparse
import pandas as pd
import numpy as np
import pathlib
base_path = str(pathlib.Path(__file__).parent.resolve())


def load_random_forest_model(file_path=base_path + '/models/random_forest.sav'):
    model = None
    if os.path.exists(file_path):
        model = pickle.load(open(file_path, 'rb'))

    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bike Prediction for a given csv")
    parser.add_argument('test_file', help='Test file path for testing', required=True)
    args = parser.parse_args()
    test_file = args.test_file

    model = load_random_forest_model()
    samples = pd.read_csv(test_file)
    pred = model.predict(samples)
    print(pred)
