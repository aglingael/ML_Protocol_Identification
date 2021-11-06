import os
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_validate
from sklearn import tree
import graphviz
import csv

prefix = "../"

df = pd.read_feather(prefix + "store/data_net.feather")
X = df[df.columns.to_list()[:-2]] # the dataframe columns end by plugin and target
X["ratio_pkts"] = X.c_pkts_all / X.s_pkts_all
X["ratio_bytes"] = X.c_bytes_all / X.s_bytes_all
y = df.target
depths = [3,4,5]

with open(prefix + "results/results_dt_shallow_net.csv", "w") as stream:
  csv_writer = csv.writer(stream, delimiter=',')
  csv_writer.writerow(["variance","depth","train_fold","test_fold","train","test","confusion"])
  
  for is_var in [True, False]:
    
    if not is_var:
      X.drop(['variance_out', 'variance_in', 'variance'], inplace=True, axis=1)
      print("\nWithout variance")
    else:
      print("\nWith variance")
    
    for depth in depths:
      print("Start depth", depth)
      clf = DecisionTreeClassifier(max_depth=depth)
      n_folds = 5
      skf = StratifiedKFold(n_splits=n_folds)
      fold_indexes = list(skf.split(X, y))
      cv_results = cross_validate(clf, X, y, cv=iter(fold_indexes), n_jobs=-1, scoring='accuracy', return_train_score=True, return_estimator=True, verbose=10)

      print("train score:", cv_results['train_score'], np.mean(cv_results['train_score']))
      print("test score:", cv_results['test_score'], np.mean(cv_results['test_score']), "\n")

      confusion_str = ""
      for k, (train_index, test_index) in enumerate(fold_indexes):
        print("confusion on fold", k+1)
        X_train, X_test = X.iloc[train_index,:], X.iloc[test_index,:]
        y_train, y_test = y[train_index], y[test_index]

        y_pred = cv_results['estimator'][k].predict(X_test)
        confusion_str += str(confusion_matrix(y_test, y_pred)) + "\n\n"

        dot_data = tree.export_graphviz(cv_results['estimator'][k], out_file=None,
                                        class_names=["fec", "monitoring", "datagram", "no_plugin"],
                                        filled=True, rounded=True,
                                        special_characters=False)
        graph = graphviz.Source(dot_data)
        filename = "dt_shallow_{}_net_with_{}var_fold_{}".format(str(depth), 'no_' if not is_var else '', str(k+1))
        graph.render(prefix + "/charts/" + filename)
      print("End depth", depth, "\n\n")

      csv_writer.writerow(["1" if is_var else "0", str(depth), ",".join(str(v) for v in cv_results['train_score']), ",".join(str(v) for v in cv_results['test_score']), str(np.mean(cv_results['train_score'])), str(np.mean(cv_results['test_score'])), confusion_str])
      stream.flush()



