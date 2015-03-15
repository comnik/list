import csv
import datetime
import pprint
from itertools              import chain, groupby

import numpy                as np
import matplotlib.pyplot    as plt
from   sklearn              import svm, linear_model, metrics, cross_validation, grid_search, preprocessing, feature_extraction, ensemble

import os
import shutil


Poly = preprocessing.PolynomialFeatures(degree=3)


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

boole = lambda x: 1 if x else -1


def to_feature_vec(row):
    """
    Returns the feature-vector representation of a piece of input data.
    """
    bias = 1

    date_str = row[0]
    a, b, c, d, e, f = [float(col) for col in row[1:]]

    date = get_date(date_str)
    minutes = (date - epoch).total_seconds() / 60

    year, number, weekday = date.isocalendar()

    categories  = [boole(b == x) for x in range(4)]
    hours       = [boole(date.hour == x) for x in range(0, 25)]
    days        = [boole((date.month * 31 + date.day) == x) for x in range(1, 372)]
    polynomials = list(chain(*Poly.fit_transform([date.hour, date.month, d, year, number, weekday, minutes])))
    weekdays    = [boole(weekday == x) for x in range(1, 8)]
    weeks       = [boole(((year-2012)*53+number) == x) for x in range(54, 162)]


    # return [date.hour, bias, a, c, d, e, f] + hours + categories + polynomials + weekdays + list(date.isocalendar()) + days
    return [date.hour, a, b, c, d, e, f, minutes, year, number, weekday, (date.month * 31 + date.day), date.month, (year*53+number)] + hours + categories + polynomials + weekdays + list(date.isocalendar()) + days + weeks

def get_date(s):
    """
    Helper function that reads dates in our expected format.
    """
    return datetime.datetime.strptime(s, '%Y-%m-%d %H:%M:%S')


def get_features(inpath):
    """
    Reads our input data from a csv file
    and returns the feature-matrix.
    """

    with open(inpath, 'r') as fin:
        reader = csv.reader(fin, delimiter=',')
        X = [to_feature_vec(row) for row in reader]

    return np.atleast_1d(X)


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
    print("Getting features...")
    X = get_features('project_data/train.csv')
    Y = np.log(1 + np.genfromtxt('project_data/train_y.csv', delimiter=','))

    # Split training and test data.
    Xtrain, Xtest, Ytrain, Ytest = cross_validation.train_test_split(X, Y, train_size=0.75)

    # Train the model.
    print("Training the model...")
    # model = linear_model.LinearRegression()
    model = ensemble.RandomForestRegressor(n_estimators=30, n_jobs=-1)
    model.fit(Xtrain, Ytrain)

    scorefun = metrics.make_scorer(least_squares_loss)
    scores = cross_validation.cross_val_score(model, X, Y, scoring = scorefun, cv = 5)
    print('Scores: ', scores)
    print('Mean: ', np.mean(scores), ' +/- ', np.std(scores))

    # Plot.
    print("Plotting...")
    Xplot = Xtest[:, 0]
    Yplot = model.predict(Xtest)

    plt.plot(rand_jitter(Xplot), Ytest, 'bo', alpha = 0.1) # training data
    plt.plot(rand_jitter(Xplot), Yplot, 'go', alpha = 0.3) # prediction

    # mean_prediction = np.array([np.mean(list(g)) for k, g in group_predictions(preprocessing.scale(Xplot), Yplot)])
    # plt.plot(np.array(range(0, 24)), mean_prediction, 'r', linewidth = 3) # mean prediction
    # plt.plot(Xplot, np.exp(Yplot - Ytest), 'ro', linewidth = 3, alpha = 0.1) # residuals
    plt.show()

    # Re-train on the full training set
    print("Re-fitting...")
    model.fit(X, Y)

    # Archive our code & results
    folder = str(np.mean(scores))
    if not os.path.exists(folder):
        os.makedirs(folder)
    shutil.copy2('predict.py', folder)

    Xval = get_features('project_data/validate.csv')
    Ypred = transform_back(model.predict(Xval))
    np.savetxt(folder + '/validate_y.txt', Ypred)

    Xval = get_features('project_data/test.csv')
    Ypred = transform_back(model.predict(Xval))
    np.savetxt(folder + '/test_y.txt', Ypred)

    raw_input('Press any key to exit...')


if __name__ == "__main__":
    main()
