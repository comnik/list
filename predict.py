import csv
import datetime
from itertools              import chain

import numpy                as np
import matplotlib.pyplot    as plt
from   sklearn              import linear_model, metrics, cross_validation, grid_search


def logscore(gtruth, gpred):
    """
    Returns the score based on the loss function given in the assignment.
    """
    gpred = np.clip(gpred, 0, np.inf)
    logdif = np.log(1 + gtruth) - np.log(1 + gpred)
    return np.sqrt(np.mean(np.square(logdif)))


def least_squares_loss(gtruth, gpred):
    """
    Returns the score based on the least squares loss function.
    """
    gpred = np.clip(gpred, 0, np.inf)
    dif = gtruth - gpred
    return np.sqrt(np.mean(np.square(dif)))


def transform_back(ypred):
    """
    Thanks Etienne. For nothing.
    """
    return np.exp(ypred) - 1


epoch = datetime.datetime.utcfromtimestamp(0)


def period(data, per, n):
    """
    Returns n periodic features from data.
    """
    per_params = (per/x for x in range(1, n))
    sines = (np.sin(data / (2*np.pi*y)) for y in per_params)
    cosines = (np.cos(data / (2*np.pi*y)) for y in per_params)

    return list(chain(*zip(sines, cosines)))


def to_feature_vec(row):
    """
    Returns the feature-vector representation of a piece of input data.
    """
    date_str = row[0]
    a, b, c, temp, e, f = [float(col) for col in row[1:]]

    date = get_date(date_str)
    minutes = (date - epoch).total_seconds() / 60

    return [date.hour, a, b, c, temp, e, f] + period(date.hour, 24, 24) + period(minutes, 60, 60)


def get_date(s):
    """
    Helper function that reads dates in our expected format.
    """
    return datetime.datetime.strptime(s, '%Y-%m-%d %H:%M:%S')


def get_features(inpath):
    """
    Reads our input data from a csv file.
    """
    with open(inpath, 'r') as fin:
        reader = csv.reader(fin, delimiter=',')
        X = [to_feature_vec(row) for row in reader]

    return np.atleast_1d(X)


def main():
    plt.ion()

    # Read labelled training data.
    # We train on the tuples (x, y).
    X = get_features('project_data/train.csv')
    Y = np.genfromtxt('project_data/train_y.csv', delimiter=',')
    Y = np.log(1 + Y)

    # Split training and test data.
    Xtrain, Xtest, Ytrain, Ytest = cross_validation.train_test_split(X, Y, train_size=0.75)
    # Xtrain = X
    # Ytrain = Y

    regressor = linear_model.LinearRegression()
    regressor.fit(Xtrain, Ytrain)

    # print(regressor.coef_)
    # print(regressor.intercept_)

    # Hplot = Xtrain[:, 0]
    # Xplot = np.atleast_1d([[x] for x in Hplot])
    Xplot = Xtrain[:, 0]
    Yplot = regressor.predict(Xtrain) #predictions

    plt.plot(Xplot, Ytrain, 'bo') # input data
    plt.plot(Xplot, Yplot, 'ro', linewidth = 3) # prediction
    plt.show()

    scorefun = metrics.make_scorer(least_squares_loss)
    scores = cross_validation.cross_val_score(regressor, X, Y, scoring = scorefun, cv = 5)
    print('Scores: ', scores)
    print('Mean: ', np.mean(scores), ' +/- ', np.std(scores))

    # regressor_ridge = linear_model.Ridge()
    # param_grid = {'alpha' : np.linspace(0,100,10)} # number of alphas is arbitrary
    # n_scorefun = metrics.make_scorer(lambda x, y: -logscore(x,y))     #logscore is always maximizing... but we want the minium
    # gs = grid_search.GridSearchCV(regressor_ridge, param_grid,scoring = n_scorefun, cv = 5)
    # gs.fit(Xtrain,Ytrain)
    # print(gs.best_estimator_)
    # print(gs.best_score_)

    Xval = get_features('project_data/validate.csv')
    # Ypred = gs.best_estimator_.predict(Xval)
    Ypred = transform_back(regressor.predict(Xval))
    # print(Ypred)
    np.savetxt('out/validate_y.txt', Ypred)

    raw_input('Press any key to exit...')


if __name__ == "__main__":
    main()
