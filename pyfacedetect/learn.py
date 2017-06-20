from sklearn import svm
import numpy as np


# validation croisée
def validationCroisee(X, Y, N):
    clf = svm.SVC(kernel='linear')

    r = np.zeros(N)

    for i in range(N):
        mask = np.zeros(X.shape[0], dtype=bool)
        mask[np.arange(i, mask.size, N)] = True

        clf.fit(X[~mask, :], Y[~mask])
        r[i] = np.mean(clf.predict(X[mask]) != Y[mask])

    # print(npy.mean(r)*100)
    return np.mean(r)*100
