import csv
import datetime

import numpy                as np
import matplotlib.pyplot    as plt
from   sklearn              import linear_model, metrics, cross_validation, grid_search


def logscore(gtruth,gpred):
    gpred = np.clip(gpred,0,np.inf)
    logdif = np.log(1 + gtruth) - np.log(1 + gpred)
    return np.sqrt(np.mean(np.square(logdif)))


def get_features(t):
    return [t, np.exp(t)]


def read_path(inpath):
    X = []
    with open(inpath,'r') as fin:
        reader = csv.reader(fin, delimiter=',')
        for row in reader:
            t = datetime.datetime.strptime(row[0], '%Y-%m-%d %H:%M:%S')
            X.append(get_features(t.hour))
    return np.atleast_2d(X)


def main():
    X = read_path('project_data/train.csv')
    Y = np.genfromtxt('project_data/train_y.csv', delimiter=',')
    # print X.shape
    # print Y.shape

    #always split training and test data!
    Xtrain, Xtest, Ytrain, Ytest = cross_validation.train_test_split(X, Y, train_size=0.75)
    # print Xtrain.shape
    # print Xtest.shape
    # plt.plot(Xtrain[:,0], Ytrain, 'bo')
    # plt.show()

    regressor = linear_model.LinearRegression()
    regressor.fit(Xtrain,Ytrain)

    print regressor.coef_
    print regressor.intercept_

    Hplot = range(25)
    Xplot = np.atleast_2d([get_features(x) for x in Hplot])
    Yplot = regressor.predict(Xplot) #predictions

    plt.plot(Xtrain[:, 0], Ytrain, 'bo') # input data
    plt.plot(Hplot,Yplot,'r',linewidth = 3) # prediction
    plt.show()

    logscore(Ytest,regressor.predict(Xtest))
    scorefun = metrics.make_scorer(logscore)
    scores = cross_validation.cross_val_score(regressor,X,Y,scoring=scorefun,cv = 5)
    print 'mean : ', np.mean(scores),' +/- ' ,np.std(scores)

    regressor_ridge = linear_model.Ridge()
    param_grid = {'alpha' : np.linspace(0,100,10)} # number of alphas is arbitrary
    n_scorefun = metrics.make_scorer(lambda x, y: -logscore(x,y))     #logscore is always maximizing... but we want the minium
    grid_search = grid_search.GridSearchCV(regressor_ridge, param_grid,scoring = n_scorefun, cv = 5)
    grid_search.fit(Xtrain,Ytrain)
    print grid_search.best_estimator_
    print grid_search.best_score_
    Xval = read_path('validate.csv')
    Ypred = grid_search.best_estimator_.predict(Xval)
    print Ypred
    np.savetxt('validate_y.txt', Ypred)

if __name__ == "__main__":
    main()
