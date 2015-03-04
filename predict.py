import csv
import datetime

import numpy                as np
import matplotlib.pyplot    as plt
from   sklearn              import linear_model, metrics, cross_validation, grid_search


def logscore(gtruth, gpred):
    gpred = np.clip(gpred,0,np.inf)
    logdif = np.log(1 + gtruth) - np.log(1 + gpred)
    return np.sqrt(np.mean(np.square(logdif)))


def to_feature_vec(t):
    """
    Returns the feature-vector representation of a piece of input data.
    """
    return [t, np.exp(t)]


def get_date(row):
    """
    Helper function that reads dates in our expected format.
    """
    return datetime.datetime.strptime(row[0], '%Y-%m-%d %H:%M:%S')


def get_features(inpath):
    """
    Reads our input data from a csv file.
    """
    with open(inpath, 'r') as fin:
        reader = csv.reader(fin, delimiter=',')
        dates = (get_date(row) for row in reader)
        X = [to_feature_vec(t.hour) for t in dates]

    return np.atleast_2d(X)


def main():
    # Read labelled training data.
    # We train on the tuples (x, y).
    X = get_features('project_data/train.csv')
    Y = np.genfromtxt('project_data/train_y.csv', delimiter=',')

    # Split training and test data.
    Xtrain, Xtest, Ytrain, Ytest = cross_validation.train_test_split(X, Y, train_size=0.75)
    plt.plot(Xtrain[:, 0], Ytrain, 'bo')
    plt.show()

    # regressor = linear_model.LinearRegression()
    # regressor.fit(Xtrain, Ytrain)

    # print(regressor.coef_)
    # print(regressor.intercept_)

    # Hplot = range(25)
    # Xplot = np.atleast_2d([get_features(x) for x in Hplot])
    # Yplot = regressor.predict(Xplot) #predictions

    # plt.plot(Xtrain[:, 0], Ytrain, 'bo') # input data
    # plt.plot(Hplot,Yplot,'r',linewidth = 3) # prediction
    # plt.show()

    # logscore(Ytest,regressor.predict(Xtest))
    # scorefun = metrics.make_scorer(logscore)
    # scores = cross_validation.cross_val_score(regressor,X,Y,scoring=scorefun,cv = 5)
    # print('mean : ', np.mean(scores),' +/- ' ,np.std(scores))

    # regressor_ridge = linear_model.Ridge()
    # param_grid = {'alpha' : np.linspace(0,100,10)} # number of alphas is arbitrary
    # n_scorefun = metrics.make_scorer(lambda x, y: -logscore(x,y))     #logscore is always maximizing... but we want the minium
    # gs = grid_search.GridSearchCV(regressor_ridge, param_grid,scoring = n_scorefun, cv = 5)
    # gs.fit(Xtrain,Ytrain)
    # print(gs.best_estimator_)
    # print(gs.best_score_)

    # Xval = get_features('project_data/validate.csv')
    # Ypred = gs.best_estimator_.predict(Xval)
    # print(Ypred)
    # np.savetxt('out/validate_y.txt', Ypred)


if __name__ == "__main__":
    main()
