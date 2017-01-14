import pandas as pd
import numpy as np
import scipy.special as special


#from scipy.stats.stats import pearsonr
def _chk_asarray(a, axis):
    if axis is None:
        a = np.ravel(a)
        outaxis = 0
    else:
        a = np.asarray(a)
        outaxis = axis

    if a.ndim == 0:
        a = np.atleast_1d(a)

    return a, outaxis

def betai(a, b, x):
    """
    Returns the incomplete beta function.

    I_x(a,b) = 1/B(a,b)*(Integral(0,x) of t^(a-1)(1-t)^(b-1) dt)

    where a,b>0 and B(a,b) = G(a)*G(b)/(G(a+b)) where G(a) is the gamma
    function of a.

    The standard broadcasting rules apply to a, b, and x.

    Parameters
    ----------
    a : array_like or float > 0

    b : array_like or float > 0

    x : array_like or float
        x will be clipped to be no greater than 1.0 .

    Returns
    -------
    betai : ndarray
        Incomplete beta function.

    """
    x = np.asarray(x)
    x = np.where(x < 1.0, x, 1.0)  # if x > 1 then return 1.0
    return special.betainc(a, b, x)

def ss(a, axis=0):
    """
    Squares each element of the input array, and returns the sum(s) of that.

    Parameters
    ----------
    a : array_like
        Input array.
    axis : int or None, optional
        Axis along which to calculate. Default is 0. If None, compute over
        the whole array `a`.

    Returns
    -------
    ss : ndarray
        The sum along the given axis for (a**2).

    See also
    --------
    square_of_sums : The square(s) of the sum(s) (the opposite of `ss`).

    Examples
    --------
    >>> from scipy import stats
    >>> a = np.array([1., 2., 5.])
    >>> stats.ss(a)
    30.0

    And calculating along an axis:

    >>> b = np.array([[1., 2., 5.], [2., 5., 6.]])
    >>> stats.ss(b, axis=1)
    array([ 30., 65.])

    """
    a, axis = _chk_asarray(a, axis)
    return np.sum(a*a, axis)

def pearsonr(x, y):
    """
    Calculates a Pearson correlation coefficient and the p-value for testing
    non-correlation.

    The Pearson correlation coefficient measures the linear relationship
    between two datasets. Strictly speaking, Pearson's correlation requires
    that each dataset be normally distributed. Like other correlation
    coefficients, this one varies between -1 and +1 with 0 implying no
    correlation. Correlations of -1 or +1 imply an exact linear
    relationship. Positive correlations imply that as x increases, so does
    y. Negative correlations imply that as x increases, y decreases.

    The p-value roughly indicates the probability of an uncorrelated system
    producing datasets that have a Pearson correlation at least as extreme
    as the one computed from these datasets. The p-values are not entirely
    reliable but are probably reasonable for datasets larger than 500 or so.

    Parameters
    ----------
    x : (N,) array_like
        Input
    y : (N,) array_like
        Input

    Returns
    -------
    (Pearson's correlation coefficient,
     2-tailed p-value)

    References
    ----------
    http://www.statsoft.com/textbook/glosp.html#Pearson%20Correlation

    """
    # x and y should have same length.
    n = len(x)
    s = pd.Series(x)
    x = np.asarray(s.values)
    s = pd.Series(y)
    y = np.asarray(s.values)
    mx = x.mean()
    my = y.mean()
    xm, ym = x - mx, y - my
    r_num = np.add.reduce(xm * ym)
    r_den = np.sqrt(ss(xm) * ss(ym))
    r = r_num / r_den

    # Presumably, if abs(r) > 1, then it is only some small artifact of floating
    # point arithmetic.
    r = max(min(r, 1.0), -1.0)
    df = n - 2
    if abs(r) == 1.0:
        prob = 0.0
    else:
        t_squared = r**2 * (df / ((1.0 - r) * (1.0 + r)))
        prob = betai(0.5*df, 0.5, df/(df+t_squared))

    return r

docs = pd.read_csv("netflix/TrainingRatings.csv", header=None)
train = docs.values[:]
docs = pd.read_csv("netflix/TestingRatings.csv", header=None)
test = docs.values[:]

user_movie = {}
tmp = []
movies = []
coef = {}
rate = {}
#coef = np.zeros((2649430, 2649430))

for i in train:
    tmp.append(int(i[1]))
movies = set(tmp)
tmp = []
people = []
for i in train:
    tmp.append(int(i[0]))
people = set(tmp)
for i in people:
    user_movie[int(i)] = {}
    rate[int(i)] = {}
    for j in movies:
        user_movie[int(i)][int(j)] = 0
        rate[int(i)][int(j)] = 0

for i in train:
    user_movie[int(i[0])][int(i[1])] = int(i[2])
#print(user_movie[1205593][8]) #deneme
coef_txt = open("coef_vals.txt", "w")
for i in people:
    coef[i] = {}
    for j in people:
        coef[i][j] = pearsonr(user_movie[i], user_movie[j])
        coef_txt.write("{0},{1},{2}".format(int(i), int(j), coef[i][j]))
tmp_u = 0 #up
tmp_d = 0 #down
coef_txt.close()
"""for i in people:
    for j in movies:
        if user_movie[i][j] == 0 :
            for k in people:
                if coef[i][k] != 0 and coef[i][k] > 0 and user_movie[k][j] != 0:
                    tmp_u += (coef[i][k] * user_movie[k][j]) #paydanin ust kismi
                    tmp_d += coef[i][k] #paydanin alt kismi
            rate[i][j] = tmp_u / tmp_d
            tmp_d = 0
            tmp_u = 0
        #zaten var olanları eklemek istersek
        else
            rate[i][j] = user_movie[i][j]"""
txt_file = open("PredictRatings.txt", "w")
for i in test:
    for k in people:
        if coef[int(i[0])][k] != 0 and coef[int(i[0])][k] > 0 and user_movie[k][int(i[1])] != 0:
            tmp_u += (coef[i][k] * user_movie[k][int(i[1])])  # paydanin ust kismi
            tmp_d += coef[i][k]  # paydanin alt kismi
        rate[int(i[0])][int(i[1])] = tmp_u / tmp_d
        txt_file.write("{0},{1},{2}".format(int(i[0]), int(i[1]), rate[int(i[0])][int(i[1])]))
        tmp_d = 0
        tmp_u = 0
txt_file.close()
