from sklearn import svm
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import time
import numpy as np

from constants import *
from utils import get_len_dict, split_dataset

start = time.time()

#this is just for testing-2
dataset = np.load('/content/drive/My Drive/Cross-spectrum-EEG/dataset_2.npy', allow_pickle=True)
dic = dataset.reshape(-1,1)[0][0]
test_size = 0.20
        
SVMs = [svm.SVC(kernel='linear') for i in range(NUM_SLEEP_STAGES)]  #snsemble of SVMs
target_names = ['class 0', 'class 1']

#independently training an SVM for each sleep stage
for svm_id in range(NUM_SLEEP_STAGES):
  print(f"For SVM-{svm_id}")
  X_train, X_test, Y_train, Y_test = split_dataset(dic, svm_id, test_size=test_size)
  # X_train[:] = [x / 1e4 for x in X_train]
  # X_test[:] = [x / 1e4 for x in X_test]
  
  print(f"Example of training feature vector: {X_train[svm_id]}")
  print(f"It's corresponding label: {Y_train[svm_id]}")
  SVMs[svm_id].fit(X_train, Y_train)
  #print("\n")
  Y_pred = SVMs[svm_id].predict(X_test)
  
  print(f"TEST PERFORMANCE OF  SVM-{svm_id}")
  print(f"Accuracy: {accuracy_score(Y_test, Y_pred)}")
  print(f"Confusion matrix: \n{confusion_matrix(Y_test, Y_pred)}")
  print(f"Classification Report:\n {classification_report(Y_test, Y_pred, target_names=target_names)}")
  print("*****************************************************************************")

print(f"The whole process took {time.time()-start} seconds")
