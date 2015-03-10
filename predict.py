import csv
import datetime
import pprint
from itertools              import chain, groupby

import numpy                as np
import matplotlib.pyplot    as plt
from   sklearn              import svm, linear_model, metrics, cross_validation, grid_search, preprocessing, feature_extraction, ensemble


Poly = preprocessing.PolynomialFeatures(degree=5)


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

    # We first generate all the parameters, e.g. (24/1 24/2 24/3 ...)
    per_params = (per/x for x in range(1, n))

    # We then calculate both components of our periodic features (sin, cos)
    sines = (np.sin(data / (2*np.pi*y)) for y in per_params)
    cosines = (np.cos(data / (2*np.pi*y)) for y in per_params)

    # Combine the two separate streams of sines and cosines into
    # one stream of (sin, cos) tuples. Those then need to be chained into a single, long list.
    return list(chain(*zip(sines, cosines)))


boole = lambda x: 1 if x else 0

feature_names = ["hour", "bias", "a", "c", "temp", "hum", "precip"] + ["hour %s" % i for i in range(1, 48)] + ["b == %s" % i for i in range(4)]


def to_feature_vec(row):
    """
    Returns the feature-vector representation of a piece of input data.
    """
    bias = 1

    date_str = row[0]
    a, b, c, temp, hum, precip = [float(col) for col in row[1:]]

    date = get_date(date_str)
    minutes = (date - epoch).total_seconds() / 60

    year, number, weekday = date.isocalendar()

    categories = [boole(b == x) for x in range(4)]
    hours = [boole(date.hour == x) for x in range(0, 25)]
    polynomials = list(chain(*Poly.fit_transform([date.hour, temp, bias])))
    weekdays = [boole(weekday == x) for x in range(1, 8)]

    return [date.hour, bias, a, c, temp, hum, precip] + hours + categories + polynomials + weekdays + list(date.isocalendar())
             # + period(date.hour, 24, 24)


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


def linear_regression(Xtrain, Ytrain, X, Y):
    """
    Trains a model using linear regression.
    """

    # regressor = linear_model.LinearRegression()
    regressor = ensemble.RandomForestRegressor()
    regressor.fit(Xtrain, Ytrain)

    scorefun = metrics.make_scorer(least_squares_loss)
    scores = cross_validation.cross_val_score(regressor, X, Y, scoring = scorefun, cv = 5)
    print('Scores: ', scores)
    print('Mean: ', np.mean(scores), ' +/- ', np.std(scores))

    return regressor


def kernelized_regression(Xtrain, Ytrain, X, Y):
    """
    Trains a model using a kernelized support vector regression.
    """

    param_grid = {
        'C':        10.0 ** np.arange(3, 9),
        'gamma':    10.0 ** np.arange(-5, -3)
    }

    model = svm.SVR(kernel='linear', C=1e5, cache_size=1000)

    scorefun = metrics.make_scorer(least_squares_loss) #logscore is always maximizing... but we want the minium
    # gs = grid_search.GridSearchCV(model, param_grid, scoring = scorefun, cv = 5)
    # gs.fit(Xtrain, Ytrain)
    # model = gs.best_estimator_
    model.fit(Xtrain, Ytrain)

    # print('Best Score: ', gs.best_score_)
    # print('Best estimator', gs.best_estimator_)

    scores = cross_validation.cross_val_score(model, X, Y, scoring = scorefun, cv = 5)
    print('Scores: ', scores)
    print('Mean: ', np.mean(scores), ' +/- ', np.std(scores))

    return model


def ridge_regression(Xtrain, Ytrain, X, Y):
    """
    Trains a model using a ridge regression.
    """

    regressor = linear_model.Ridge()
    param_grid = {'alpha' : np.linspace(0, 100, 10)} # number of alphas is arbitrary
    scorefun = metrics.make_scorer(lambda x, y: -logscore(x, y)) # logscore is always maximizing... but we want the minium
    gs = grid_search.GridSearchCV(regressor, param_grid, scoring = scorefun, cv = 5)
    gs.fit(Xtrain,Ytrain)

    model = gs.best_estimator_

    print(gs.best_estimator_)
    print(gs.best_score_)

    return model


def group_predictions(X, Y, feature_index=0):
    """
    Returns an iterator over tuples of the form (hour_of_the_day, [predictions]).
    """

    fst = lambda t: t[feature_index]
    return groupby(sorted(zip(X, Y), key=fst), key=fst)


def rand_jitter(arr):
    stdev = .01*(max(arr)-min(arr))
    return arr + np.random.randn(len(arr)) * stdev


def main():
    plt.ion()

    # Read labelled training data.
    # We train on the tuples (x, y).
    X = get_features('project_data/train.csv')
    Y = np.log(1 + np.genfromtxt('project_data/train_y.csv', delimiter=','))

    # Split training and test data.
    Xtrain, Xtest, Ytrain, Ytest = cross_validation.train_test_split(X, Y, train_size=0.75)

    # Train the model.
    model = linear_regression(Xtrain, Ytrain, X, Y)
    # model = kernelized_regression(Xtrain, Ytrain, X, Y)
    # model = ridge_regression(Xtrain, Ytrain, X, Y)

    # print('Coefficients: ', model.coef_)
    # print(model.intercept_)
    pp = pprint.PrettyPrinter(indent=4)
    # pp.pprint(coef_dict)

    Hplot = np.array(range(0, 24))
    Xplot = Xtest[:, 0]
    Yplot = model.predict(Xtest)

    mean_prediction = np.array([np.mean(list(g)) for k, g in group_predictions(preprocessing.scale(Xplot), Yplot)])

    plt.plot(Xplot, Ytest, 'bo', alpha = 0.1) # input data
    plt.plot(Xplot, Yplot, 'go', alpha = 0.3)
    plt.plot(Hplot, mean_prediction, 'r', linewidth = 3)
    # plt.plot(Xplot, np.exp(Yplot - Ytest), 'ro', linewidth = 3, alpha = 0.1) # prediction
    plt.show()

    ### Output ###
    model.fit(X, Y)

    Xval = get_features('project_data/validate.csv')
    Ypred = transform_back(model.predict(Xval))
    np.savetxt('out/validate_y.txt', Ypred)

    raw_input('Press any key to exit...')


if __name__ == "__main__":
    main()
