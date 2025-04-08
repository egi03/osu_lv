import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm

from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV


def plot_decision_regions(X, y, classifier, resolution=0.02):
    plt.figure()
    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    
    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
    np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    
    # plot class examples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0],
                    y=X[y == cl, 1],
                    alpha=0.8,
                    c=colors[idx],
                    marker=markers[idx],
                    label=cl)


# ucitaj podatke
data = pd.read_csv("Social_Network_Ads.csv")
print(data.info())

data.hist()
plt.show()

# dataframe u numpy
X = data[["Age","EstimatedSalary"]].to_numpy()
y = data["Purchased"].to_numpy()

# podijeli podatke u omjeru 80-20%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, stratify=y, random_state = 10)

# skaliraj ulazne velicine
sc = StandardScaler()
X_train_n = sc.fit_transform(X_train)
X_test_n = sc.transform((X_test))

# 1
def run_svm(kernel='rbf', C=1, gamma=1):
    svm_model = svm.SVC(kernel=kernel, gamma=gamma, C=C)
    svm_model.fit(X_train_n, y_train)

    y_test_p = svm_model.predict(X_test_n)

    print(f"\n\nkernel={kernel} C={C} gamma={gamma}")
    print("Tocnost test: " + "{:0.3f}".format((accuracy_score(y_test, y_test_p))))
    print("Preciznost test: " + "{:0.3f}".format((precision_score(y_test, y_test_p))))
    print("Odziv test: " + "{:0.3f}".format((recall_score(y_test, y_test_p))))
    print("F1 test: " + "{:0.3f}".format((f1_score(y_test, y_test_p))))

    # granica odluke pomocu logisticke regresije
    plot_decision_regions(X_train_n, y_train, classifier=svm_model)
    plt.xlabel('x_1')
    plt.ylabel('x_2')
    plt.legend(loc='upper left')
    plt.title(f"kernel={kernel} C={C} gamma={gamma} -- tocnost={accuracy_score(y_test, y_test_p)}")
    plt.tight_layout()
    plt.show()

""" run_svm(C=0.1, gamma=1)
run_svm(C=0.1, gamma=10)
run_svm(C=0.1, gamma=0.1) """

for k in ['linear', 'poly', 'rbf', 'sigmoid']:
    run_svm(kernel=k, C=0.1, gamma=0.1)



#4

param_gird = {'model__C': [10,100],
              'model__gamma': [10, 0.1, 1, 0.01]}

svm_gscv = GridSearchCV(Pipeline([('scaler', StandardScaler()), ('model', svm.SVC())]), param_gird, cv=5, scoring='accuracy')
svm_gscv.fit(X_train_n, y_train)

print (svm_gscv.best_params_)
print (svm_gscv.best_score_)
run_svm(C=10, gamma=1)