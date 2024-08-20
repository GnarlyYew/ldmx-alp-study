import numpy as np
import matplotlib.pyplot as plt
import uproot
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

combined_file = uproot.open("combined_2000MeV.root")
features_both = combined_file["outTree"]
df_both = features_both.arrays(library='pd')

#df_both = features_both.pandas.df(flatten=False)vNHits
X = df_both[["vNHits","vZAverage","vZWidth","vEav","vEDensity","vXYWidth"]].values
Y = df_both[["visSignal"]].values

#split data, fit and predict:
X_train , X_test, y_train, y_test = train_test_split(X,Y.ravel(),test_size = 0.5, random_state = 100)

# lists
names = [
    "Nearest Neighbors",
    "Linear SVM",
    "RBF SVM",
    #"Gaussian Process",
    "Decision Tree",
    "Random Forest",
    "Neural Net",
    "AdaBoost",
    "Naive Bayes",
    "QDA",
]

classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025),
    SVC(gamma=2, C=1),
    #GaussianProcessClassifier(1.0 * RBF(1.0)),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    MLPClassifier(alpha=1, max_iter=1000),
    AdaBoostClassifier(),
    GaussianNB(),
    QuadraticDiscriminantAnalysis(),
]

X, y = make_classification(
    n_features=6, n_redundant=0, n_informative=2, random_state=1, n_clusters_per_class=1
)

# iterate over classifiers
for name, clf in zip(names, classifiers):

    clf = make_pipeline(StandardScaler(), clf)
    clf.fit(X_train, y_train)
    y_new = clf.predict(X_test)

    dPred = []
    nWrong = 0
    nRight = 0
    for i,j in enumerate(y_new):
        dPred.append(y_new[i]-y_test[i])
        if abs(y_new[i]-y_test[i]) == 1:
            nWrong += 1
        else:
            nRight += 1
    print(name, 1 - nWrong/(nWrong+nRight))

#### MLP ####
regr  = MLPClassifier(random_state=0).fit(X_train, y_train)
y_new = regr.predict(X_test)

dPred = []
nWrong = 0
nRight = 0
for i,j in enumerate(y_new):
    dPred.append(y_new[i]-y_test[i])
    if abs(y_new[i]-y_test[i]) == 1:
        nWrong += 1
    else:
        nRight += 1
print("MLP accuracy", 1 - nWrong/(nWrong+nRight))
