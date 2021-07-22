

def split_and_select_features(data):
    data.sample(frac=1.0, replace=True, random_state=1)
    split_train = int(60 / 100 * len(data))
    split_val = int(80 / 100 * len(data))
    train = data[:split_train]
    val = data[split_train:split_val]
    test = data[split_val:]
    category_features = ['season', 'holiday', 'mnth', 'hr', 'weekday', 'workingday', 'weathersit']
    number_features = ['temp', 'atemp', 'hum', 'windspeed']
    features = category_features + number_features
    target = ['cnt']
    return features, number_features, target, test, train, val