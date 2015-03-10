import csv
import datetime
from itertools import chain
import os
import shutil
from math import *
import pprint
from itertools              import chain

import numpy                as np
import matplotlib.pyplot    as plt
from   sklearn import linear_model, metrics, cross_validation
# from   sklearn              import svm, linear_model, metrics, cross_validation, grid_search, preprocessing, feature_extraction #niko


Poly = preprocessing.PolynomialFeatures(degree=3)


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
    per_params = (per / x for x in range(1, n + 1))

    # We then calculate both components of our periodic features (sin, cos)
    sines = (sin(data / (2 * pi * y)) for y in per_params)
    cosines = (cos(data / (2 * pi * y)) for y in per_params)

    # Combine the two separate streams of sines and cosines into
    # one stream of (sin, cos) tuples. Those then need to be chained into a single, long list.
    return list(chain(*zip(sines, cosines)))


boole = lambda x: 1 if x else 0

feature_names = ["hour", "bias", "a", "c", "temp", "hum", "precip"] + ["hour %s" % i for i in range(1, 48)] + ["b == %s" % i for i in range(4)] # niko output


def to_feature_vec(row):
    """
    Returns the feature-vector representation of a piece of input data.
    """
    bias = 1

    date_str = row[0]
    a, b, c, temp, hum, precip = [float(col) for col in row[1:]]

    date = get_date(date_str)
    minutes = (date - epoch).total_seconds() / 60

    (year, week, day) = date.isocalendar() # etienned
    # year, number, weekday = date.isocalendar() # niko

    logistic = lambda x: 1 / (1 + exp(-x))
    
    # categories = [boole(b == x) for x in range(4)]                            # nikofeatures
    # hours = [boole(date.hour == x) for x in range(0, 25)]                     # nikofeatures
    # polynomials = list(chain(*Poly.fit_transform([a,b,c,temp,hum,precip])))   # nikofeatures

    if asinh(max(date.hour, 4.33085471291034) - 3.23315698150681) >= 3.23315698150681:
        lolparameter = 0.7310585786300049
    else:
        lolparameter = 0.5
        
    # my return, contains alchemy -- etienned
    return [date.hour, bias, a, b, c, temp, hum, precip, year, week, day, date.minute
           ] + period(date.minute, 60, 60) + period(date.hour, 24, 24) + period(day, 7, 7) + period(date.hour, 24 * 7,
                                                                                                    24 * 7) + period(
        date.day, 365, 50) + [
               atan(week) * min(0.628387914132889, min(a, b)),
               # minutes**2*math.asinh(max(date.hour, 4.30604529803556) - 3.25027364193086),
               # minutes**2*math.asinh(max(date.hour, 4.35826554407763) - 3.2762492513184),
               logistic(week) * atan(0.268294337091847 * week) * min(0.615528009111349, a, b),
               atan2(week, 3.87347963586697) * min(lolparameter, a, b),
               boole(b == 0), boole(b == 1),
               sin(a),
               sin(2.260304361624 * sin(-115.200026399002 / (12.3098089533563 + date.hour))),
               sin(0.054465515709371 * week),
               minutes,
               minutes ** 2,
               b ** 8.94248740512325e-7 * sin(0.0552724359814239 * week),
               cos(precip) * sin(0.0552724359814239 * week)
           ] + [boole(date.hour == x) for x in range(0, 24)]

    # niko's return:
    # return [date.hour, bias, a, c, temp, hum, precip] \
    #         + hours \
    #         + period(date.hour, 24, 24) \
    #         + categories \
    #         + polynomials \
    #         + list(date.isocalendar()) \
    #         + [boole(weekday == x) for x in range(1, 8)] # one-hot weekday # why 8? -- etienned


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


# function only used by niko:

def linear_regression(Xtrain, Ytrain, X, Y):
    """
    Trains a model using linear regression.
    """

    regressor = linear_model.LinearRegression()
    regressor.fit(Xtrain, Ytrain)

    scorefun = metrics.make_scorer(least_squares_loss)
    scores = cross_validation.cross_val_score(regressor, X, Y, scoring = scorefun, cv = 5)
    print('Scores: ', scores)
    print('Mean: ', np.mean(scores), ' +/- ', np.std(scores))

    return regressor

# function only used by niko:

def kernelized_regression(Xtrain, Ytrain, X, Y):
    """
    Trains a model using a kernelized support vector regression.
    """

    param_grid = {
        'C':        10.0 ** np.arange(-2, 9),
        # 'gamma':    10.0 ** np.arange(-5, 4)
    }

    model = svm.SVR(kernel='poly', cache_size=400)

    scorefun = metrics.make_scorer(lambda x, y: least_squares_loss(x,y)) #logscore is always maximizing... but we want the minium
    # gs = grid_search.GridSearchCV(model, param_grid, scoring = scorefun, cv = 5)
    # gs.fit(Xtrain, Ytrain)
    # model = gs.best_estimator_
    model.fit(Xtrain, Ytrain)

    # print('Best Score: ', gs.best_score_)
    scores = cross_validation.cross_val_score(model, X, Y, scoring = scorefun, cv = 5)
    print('Scores: ', scores)
    print('Mean: ', np.mean(scores), ' +/- ', np.std(scores))

    return model

# function only used by niko:

def ridge_regression(Xtrain, Ytrain, X, Y):
    """
    Trains a model using a ridge regression.
    """

    regressor = linear_model.Ridge()
    param_grid = {'alpha' : np.linspace(0, 100, 10)} # number of alphas is arbitrary
    scorefun = metrics.make_scorer(lambda x, y: -least_squares_loss(x, y)) # logscore is always maximizing... but we want the minium
    gs = grid_search.GridSearchCV(regressor, param_grid, scoring = scorefun, cv = 5)
    gs.fit(Xtrain,Ytrain)

    model = gs.best_estimator_

    print(gs.best_estimator_)
    print(gs.best_score_)

    return model


def main():
    plt.ion()

    # Read labelled training data.
    # We train on the tuples (x, y).
    X = get_features('project_data/train.csv')
    Y = np.log(1 + np.genfromtxt('project_data/train_y.csv', delimiter=','))

    # Split training and test data.
    # Xtrain, Xtest, Ytrain, Ytest = cross_validation.train_test_split(X, Y, train_size=0.75)
    Xtrain = X
    Ytrain = Y

    regressor = linear_model.LinearRegression()
    regressor.fit(Xtrain, Ytrain)
    
    # what niko does:
    
    # # Train the model.
    # # model = linear_regression(Xtrain, Ytrain, X, Y)
    # model = kernelized_regression(Xtrain, Ytrain, X, Y)
    # # model = ridge_regression(Xtrain, Ytrain, X, Y)

    print('Coefficients: ', regressor.coef_)
    # print(regressor.intercept_)
    
    # Maybe better:
    # coef_dict = dict(zip(feature_names, regressor.coef_))
    # # print('Coefficients: ', regressor.coef_)
    # pp = pprint.PrettyPrinter(indent=4)
    # pp.pprint(coef_dict)
    # # print(regressor.intercept_)

    # Hplot = Xtrain[:, 0]
    # Xplot = np.atleast_1d([[x] for x in Hplot])
    Xplot = Xtrain[:, 0]
    Yplot = regressor.predict(Xtrain)  # predictions

    # plt.plot(Xplot, Ytrain, 'bo')  # input data
    # plt.plot(Xplot, Yplot, 'ro', linewidth=3)  # prediction
    # plt.plot(Xtrain[:, 0], Xtrain[:, 7], 'bo')
    plt.plot(Xplot, Yplot - Ytrain, 'ro', linewidth=0, alpha=0.01)
    plt.show()

    scorefun = metrics.make_scorer(least_squares_loss)
    scores = cross_validation.cross_val_score(regressor, X, Y, scoring=scorefun, cv=5)
    print('Scores: ', scores)
    print('Mean: ', np.mean(scores), ' +/- ', np.std(scores))

    # regressor_ridge = linear_model.Ridge()
    # param_grid = {'alpha' : np.linspace(0, 100, 10)} # number of alphas is arbitrary
    # n_scorefun = metrics.make_scorer(lambda x, y: -least_squares_loss(x,y)) #logscore is always maximizing... but we want the minium
    # gs = grid_search.GridSearchCV(regressor_ridge, param_grid, scoring = n_scorefun, cv = 5)
    # gs.fit(Xtrain, Ytrain)

    # # print(gs.best_estimator_)
    # # print(gs.best_score_)

    # scores = cross_validation.cross_val_score(gs.best_estimator_, X, Y, scoring = scorefun, cv = 5)
    # print('Scores: ', scores)
    # print('Mean: ', np.mean(scores), ' +/- ', np.std(scores))

    # Xval = get_features('project_data/validate.csv')
    # # Ypred = gs.best_estimator_.predict(Xval)
    # Ypred = transform_back(regressor.predict(Xval))
    # # print(Ypred)
    # np.savetxt('out/validate_y.txt', Ypred)
    folder = str(np.mean(scores))
    if not os.path.exists(folder):
        os.makedirs(folder)
        shutil.copy2('predict.py', folder)

        # raw_input('Press any key to exit...')

def main_niko():
    plt.ion()

    # Read labelled training data.
    # We train on the tuples (x, y).
    X = get_features('project_data/train.csv')
    Y = np.log(1 + np.genfromtxt('project_data/train_y.csv', delimiter=','))

    # Split training and test data.
    Xtrain, Xtest, Ytrain, Ytest = cross_validation.train_test_split(X, Y, train_size=0.75)

    # Train the model.
    # model = linear_regression(Xtrain, Ytrain, X, Y)
    model = kernelized_regression(Xtrain, Ytrain, X, Y)
    # model = ridge_regression(Xtrain, Ytrain, X, Y)

    # coef_dict = dict(zip(feature_names, regressor.coef_))
    # # print('Coefficients: ', regressor.coef_)
    # pp = pprint.PrettyPrinter(indent=4)
    # pp.pprint(coef_dict)
    # # print(regressor.intercept_)

    Xplot = Xtest[:, 0]
    Yplot = model.predict(Xtest)

    # plt.plot(Xplot, Ytest, 'bo', alpha = 0.1) # input data
    # plt.plot(Xplot, Yplot, 'ro', linewidth = 3, alpha = 0.1)
    plt.plot(Xplot, Yplot - Ytest, 'ro', linewidth = 3, alpha = 0.1) # prediction
    plt.show()

    ### Output ###
    model.fit(X, Y)

    Xval = get_features('project_data/validate.csv')
    Ypred = transform_back(model.predict(Xval))
    np.savetxt('out/validate_y.txt', Ypred)

    raw_input('Press any key to exit...')

if __name__ == "__main__":
    main()