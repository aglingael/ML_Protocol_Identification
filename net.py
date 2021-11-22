import os
import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_validate
from sklearn import tree
import graphviz
import csv
from copy import deepcopy
import re

def string_to_numpy(text, dtype=None):
    """
    Convert text into 1D or 2D arrays using np.matrix().
    The result is returned as an np.ndarray.
    """
    text = text.strip()
    # Using a regexp, decide whether the array is flat or not.
    # The following matches either: "[1 2 3]" or "1 2 3"
    is_flat = bool(re.match(r"^(\[[^\[].+[^\]]\]|[^\[].+[^\]])$",
                            text, flags=re.S))
    # Replace newline characters with semicolons.
    text = text.replace("]\n", "];")
    # Prepare the result.
    result = np.asarray(np.matrix(text, dtype=dtype))
    return result.flatten() if is_flat else result

prefix = "../"

## SGD
mod_sgd = "SGD"
clf_sgd = SGDClassifier(max_iter=4000, tol=0.001, n_jobs=-1)
pgrid_sgd = {
    'alpha': [1e-2, 1e-3, 1e-4],
    'penalty': ['l1', 'l2', 'elasticnet'],
    'loss': ['hinge', 'log', 'modified_huber', 'squared_hinge', 'perceptron']
    }

## DT
mod_dt = "DT"
clf_dt = DecisionTreeClassifier()
pgrid_dt = {
'max_depth': list(range(1, 16)),
'min_samples_leaf': [1, 5, 10]
}

## Adaboost
mod_ada = "ADA"
clf_ada = AdaBoostClassifier()
pgrid_ada = {
'base_estimator': [DecisionTreeClassifier(max_depth=1), DecisionTreeClassifier(max_depth=2), DecisionTreeClassifier(max_depth=3), DecisionTreeClassifier(max_depth=4)],
'n_estimators': [50, 100, 250, 1000]
}

## GBoost
mod_gb = "GBoost"
clf_gb = GradientBoostingClassifier()
pgrid_gb = {
'max_depth' : list(range(1, 5)),
'n_estimators': [50, 100, 250, 1000]
}

## RF
mod_rf = "Rand_F"
clf_rf = RandomForestClassifier()
pgrid_rf = {
    'max_depth' : list(range(1, 5)),
    'n_estimators': [50, 100, 250, 1000],
    'min_samples_leaf': [1, 5, 10]
}

## SVM
mod_svm = "SVM"
clf_svm = SVC(max_iter=2000, tol=0.01)
pgrid_svm =  {
    'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
    'gamma': ['scale', 'auto'],
    'degree': [1, 2, 3],
    'C': [1, 10, 100, 1000]
    }


with open(prefix + "results/results_net.csv", "w") as stream:
  csv_writer = csv.writer(stream, delimiter=',')
  csv_writer.writerow(["algo","params","variance","train_fold","test_fold","train","test","confusion","mean_confusion"])

  df = pd.read_feather(prefix + "store/data_net.feather")

  X = df[df.columns.to_list()[:-2]] # the dataframe columns end by plugin and target
  X["ratio_pkts"] = X.c_pkts_all / X.s_pkts_all
  X["ratio_bytes"] = X.c_bytes_all / X.s_bytes_all
  y = df.target

  for is_var in [True, False]:
    if not is_var:
      X.drop(['variance_out', 'variance_in', 'variance'], inplace=True, axis=1)
      print("\nWithout variance")
    else:
      print("\nWith variance")

    for name, clf, pgrid in zip([mod_sgd, mod_dt, mod_ada, mod_gb, mod_rf, mod_svm], [clf_sgd, clf_dt, clf_ada, clf_gb, clf_rf, clf_svm], [pgrid_sgd, pgrid_dt, pgrid_ada, pgrid_gb, pgrid_rf, pgrid_svm]):

      n_folds = 5
      skf = StratifiedKFold(n_splits=n_folds)
      fold_indexes = list(skf.split(X, y))
      gs = GridSearchCV(deepcopy(clf), deepcopy(pgrid), cv=iter(fold_indexes), n_jobs=-1, scoring='accuracy', return_train_score=True, refit=False, verbose=10)
      gs.fit(X, y)

      # print("different params:", gs.cv_results_['params'])
      best_train_scores = list(map(lambda x : round(gs.cv_results_['split{}_train_score'.format(x)][gs.best_index_],4),list(range(n_folds))))
      best_test_scores = list(map(lambda x : round(gs.cv_results_['split{}_test_score'.format(x)][gs.best_index_],4),list(range(n_folds))))
      mean_train_score = round(np.mean(best_train_scores),4)
      mean_test_score = round(np.mean(best_test_scores),4)
      print("best params:", gs.best_params_)
      print("best score", gs.best_score_)
      print("train score:", gs.cv_results_['mean_train_score'][gs.best_index_])
      print("test score:", gs.cv_results_['mean_test_score'][gs.best_index_])

      model = None
      if name == "DT":
        model = DecisionTreeClassifier(**gs.best_params_)
      elif name == "SGD":
        model = SGDClassifier(**gs.best_params_)
      elif name == "ADA":
        model = AdaBoostClassifier(**gs.best_params_)
      elif name == "GBoost":
        model = GradientBoostingClassifier(**gs.best_params_)
      elif name == "Rand_F":
        model = RandomForestClassifier(**gs.best_params_)
      elif name == "SVM":
        model = SVC(**gs.best_params_)

      print("re-run model")
      cv_results = cross_validate(model, X, y, cv=iter(fold_indexes), return_estimator=True, n_jobs=n_folds)
      # confusion_str = "\n\n".join([str(confusion_matrix(y[fold_indexes[k][1]], estim.predict(X[fold_indexes[k][1]]))) for k, estim in enumerate(cv_results['estimator'])])

      confusion_str = ""
      for k, (train_index, test_index) in enumerate(fold_indexes):
        print("confusion on fold", k+1)
        X_train, X_test = X.iloc[train_index,:], X.iloc[test_index,:]
        y_train, y_test = y[train_index], y[test_index]

        y_pred = cv_results['estimator'][k].predict(X_test)
        confusion_str += str(confusion_matrix(y_test, y_pred)) + "\n\n"

        matrices = confusion_str.strip().split("\n\n")
        matrices = list(map(lambda x : string_to_numpy(x), matrices)) 
        mean_matrix = np.mean( np.array(matrices), axis=0 )
        mean_matrix = str(mean_matrix)

        if name == "DT":
          dot_data = tree.export_graphviz(cv_results['estimator'][k], out_file=None,
                                          class_names=["fec", "monitoring", "multipath", "no_plugin"],
                                          filled=True, rounded=True,
                                          special_characters=False)
          graph = graphviz.Source(dot_data)
          filename = "dt_net_with_{}var_fold_{}".format('no_' if not is_var else '', str(k+1))
          graph.render(prefix + "/charts/" + filename)
      print("\n\n")

      csv_writer.writerow([name, str(gs.best_params_), "1" if is_var else "0", ",".join(str(v) for v in best_train_scores), ",".join(str(v) for v in best_test_scores), str(mean_train_score), str(mean_test_score), confusion_str, mean_matrix])
      stream.flush()
