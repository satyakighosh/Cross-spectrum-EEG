#Import svm model
from sklearn import svm
import time
import numpy as np

from dataset_builder import TrainDatasetBuilder
from constants import *
from utils import get_len_dict, split_dataset

start = time.time()
train_dataset = TrainDatasetBuilder(size=5)  #size->number of patients from which segment bank will be made
#train_dataset.create_segment_bank()
train_dataset.create_trainset() #already shuffled
print("{" + "\n".join("{!r}: {!r},".format(k, v) for k, v in train_dataset.train_set.items()) + "}")

get_len_dict(train_dataset.train_set)
# print(train_dataset.train_set[0])
# print(train_dataset.train_set[0][0])
# print(train_dataset.train_set[0][0][0])
# print(train_dataset.train_set[0][0][1].shape)
print("Saving trainset")
np.save('train_set', train_dataset.train_set)

# X_train = []
# Y_train = []
# for patient_label, feature_vector in train_dataset.train_set:
#   X_train.append(feature_vector)
#   Y_train.append(patient_label)
dic = train_dataset.train_set.reshape(-1,1)[0][0]
X = []
Y = []
test_size = 0.20
svm_id = 0              #which Favg->signifies which SVM this data is meant for


SVMs = [svm.SVC(kernel='linear') for i in range(NUM_SLEEP_STAGES)]

for i in range(NUM_SLEEP_STAGES):
  svm_id = i 
  X_train, X_test, Y_train, Y_test = split_dataset(dic, svm_id, test_size=test_size)
  SVMs[svm_id].fit(X_train, Y_train)
  y_pred = SVMs[svm_id].predict(X_test)
# #Predict the response for test dataset
# #y_pred = clf.predict(X_test)
print(f"The whole process took {time.time()-start} seconds")
