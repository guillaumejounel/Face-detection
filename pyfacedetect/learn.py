from sklearn import svm
import numpy as np


# validation croisée
def validationCroisee(clf, X, Y, N):

    r = np.zeros(N)

    for i in range(N):
        mask = np.zeros(X.shape[0], dtype=bool)
        mask[np.arange(i, mask.size, N)] = True

        clf.fit(X[~mask, :], Y[~mask])
        r[i] = np.mean(clf.predict(X[mask]) != Y[mask])

    # print(npy.mean(r)*100)
    return np.mean(r)*100


def graphValidationCroisee(clf, exemples, y, debut, fin, pas):
    choixC = np.zeros(((fin-debut)//pas+1, 2))
    cursor = 0
    for i in np.arange(debut,fin,pas):
        pct = round(100*(i-debut)/(fin-debut))
        print("\r" + str(pct//2*"-"+"{}%".format(pct)), end="\r")
        clf = svm.SVC(kernel='linear', C=i)
        clf.fit(exemples,y)
        choixC[cursor] = [i, validationCroisee(clf, exemples, y, 5)]
        cursor+=1
    plt.plot(choixC[:,0],choixC[:,1],'.-')