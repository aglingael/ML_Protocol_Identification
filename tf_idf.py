import os
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from scipy.sparse import save_npz, load_npz
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_validate
from sklearn import tree
import dask
import graphviz
import csv
import json

prefix = "../"


with open(prefix + "results/results_tfidf.csv", "w") as stream:
  csv_writer = csv.writer(stream, delimiter=',')
  csv_writer.writerow(["header","dim_red","algo","params","train_fold","test_fold","train","test","confusion"])

  # run with header and without header
  for with_header in [True, False]:
    
    df = None 
    vec = False
    
    if with_header:
      if not os.path.isfile(prefix + "store/matrix_stems_head_dim_red.npy") or not os.path.isfile(prefix + "store/matrix_stems_head_no_dim_red.npz"):
        df = pd.read_feather(prefix + "store/tokens_nltk_stems.feather")
        vec = True
        print("data with header has been read")
    else:
      if not os.path.isfile(prefix + "store/matrix_stems_no_head_dim_red.npy") or not os.path.isfile(prefix + "store/matrix_stems_no_head_no_dim_red.npz"):
        df = pd.read_feather(prefix + "store/tokens_nltk_stems_wo_head.feather")
        vec = True
        print("data without header has been read")
    
    if vec:
      X = df.stems
      tfidf = TfidfVectorizer(stop_words='english', ngram_range=(1, 1), use_idf=True)
      
      print("tfidf launched")
      X = tfidf.fit_transform(X)
      print("tfidf done.")
      
      vocab = sorted(tfidf.vocabulary_, key=tfidf.vocabulary_.get)
      with open(prefix + ("store/stems_vocab.txt" if with_header else "store/stems_no_head_vocab.txt"), "w") as file_voc:
        json.dump(vocab, file_voc)
      
      y = df.target
      np.save(prefix + "store/target.npy", y)

      head = 'matrix_stems_head' if with_header else 'matrix_stems_no_head'
      
      print("saving sparse matrix")
      save_npz(prefix + 'store/' + head + "_no_dim_red.npz", X)
      print("done")
      
      tsvd = TruncatedSVD(n_components=300, algorithm='randomized', n_iter=3)
      print("dimension reductionality launched")
      X = tsvd.fit_transform(X)
      print("dimension reductionality done")
      print("print resulting matrix")
      np.save(prefix + 'store/' + head + "_dim_red.npy", X)
      print("done")

    else:
      with open(prefix + ("store/stems_vocab.txt" if with_header else "store/stems_no_head_vocab.txt"), "r") as file_voc:
        vocab = json.load(file_voc)
      y = np.load(prefix + "store/target.npy", allow_pickle=True)
  
    for dim_red in [True, False]:

      head = 'matrix_stems_head' if with_header else 'matrix_stems_no_head'
      dim = "_dim_red.npy" if dim_red else "_no_dim_red.npz"
      filename = head + dim

      if dim_red:
        X = np.load(prefix + 'store/' + filename, allow_pickle=True)
      else:
        X = load_npz(prefix + 'store/' + filename)

      ## SGD
      mod_sgd = "SGD"
      clf_sgd = SGDClassifier(max_iter=2000, tol=0.01, n_jobs=5)
      pgrid_sgd = {
          'alpha': [1e-2],
          'penalty': ['l1', 'l2', 'elasticnet'],
          'loss': ['hinge', 'log', 'perceptron']
      }

      ## DT
      mod_dt = "DT"
      clf_dt = DecisionTreeClassifier()
      pgrid_dt = {
          'max_depth': list(range(1, 8))
      }

      ## Adaboost
      mod_ada = "ADA"
      clf_ada = AdaBoostClassifier()
      pgrid_ada = {
          'base_estimator': [DecisionTreeClassifier(max_depth=1), DecisionTreeClassifier(max_depth=2), DecisionTreeClassifier(max_depth=3)],
          'n_estimators': [100]
      }

      ## GBoost
      mod_gb = "GBoost"
      clf_gb = GradientBoostingClassifier()
      pgrid_gb = {
          'max_depth' : list(range(1, 4)),
          'n_estimators': [100]
      }

      ## RF
      mod_rf = "Rand_F"
      clf_rf = RandomForestClassifier()
      pgrid_rf = {
          'max_depth' : list(range(1, 4)),
          'n_estimators': [100]
      }

      ## SVM
      mod_svm = "SVM"
      clf_svm = SVC(max_iter=2000, tol=0.01)
      pgrid_svm = {
          'kernel': ['poly', 'rbf'],
          'gamma': ['scale', 'auto'],
          'degree': [1, 2],
          'C': [1, 10]
      }

      # loop in each classifier and its parameters for grid-search
      for name, clf, pgrid in zip([mod_dt, mod_ada, mod_gb, mod_rf, mod_svm, mod_sgd], [clf_dt, clf_ada, clf_gb, clf_rf, clf_svm, clf_sgd], [pgrid_dt, pgrid_ada, pgrid_gb, pgrid_rf, pgrid_svm, pgrid_sgd]):
        
        print("header:", with_header, "dim_red", dim_red, "algo:", name)
        n_folds = 5
        skf = StratifiedKFold(n_splits=n_folds)
        fold_indexes = list(skf.split(X, y))
        gs = GridSearchCV(clf, pgrid, cv=iter(fold_indexes), n_jobs=15, scoring='accuracy', return_train_score=True, refit=False, verbose=10)
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
          X_train, X_test = X[train_index], X[test_index]
          y_train, y_test = y[train_index], y[test_index]
          
          y_pred = cv_results['estimator'][k].predict(X_test)
          confusion_str += str(confusion_matrix(y_test, y_pred)) + "\n\n"

        
          if name == "DT" and dim_red is False:
            dot_data = tree.export_graphviz(cv_results['estimator'][k], out_file=None, 
                                            feature_names=list(map(lambda x: "val['{}']".format(x), vocab)),
                                            class_names=["fec", "monitoring", "datagram", "no_plugin"],  
                                            filled=True, rounded=True,  
                                            special_characters=False)  
            graph = graphviz.Source(dot_data) 
            filename = "dt_with_{}header_fold_{}".format('no_' if not with_header else '', str(k+1))
            graph.render(prefix + "/charts/" + filename)  
        print("\n\n")
        
        csv_writer.writerow(["1" if with_header else "0", "1" if dim_red else "0", name, str(gs.best_params_), ",".join(str(v) for v in best_train_scores), ",".join(str(v) for v in best_test_scores), str(mean_train_score), str(mean_test_score), confusion_str])
        stream.flush()
