import numpy as np
from ML_CNN.data_preparation import create_training_data
from ML_CNN.utils import check_folders
from ML_CNN.model import create_model, train_model


def main():
    if check_folders('./TEMP_SEG') is False:
        raise Exception(
            'Non sono presenti immagini nelle cartelle DB_PROBS_SEG e/o DB_NORMAL_SEG')

    X = []
    y = []

    training_data = create_training_data()

    for features, label in training_data:
        X.append(features)
        y.append(label)

    X = np.array(X)
    X = X.reshape(-1, X.shape[1], X.shape[2], 1)

    model = create_model(X, y)
    train_model(model, X, y)


if __name__ == '__main__':
    main()
