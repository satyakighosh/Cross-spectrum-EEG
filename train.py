#Import svm model
#from sklearn import svm
from dataset_builder import TrainDatasetBuilder
import time
start = time.time()
train_dataset = TrainDatasetBuilder(5)
train_dataset.create_segment_bank()
train_dataset.create_trainset()
for i in range(5):
  print(train_dataset.train_set[i])

print(f"The whole process took {time.time()-start} seconds")
#Create a svm Classifier
#clf = svm.SVC(kernel='linear') # Linear Kernel

#Train the model using the training sets
#clf.fit(X_train, y_train)

#Predict the response for test dataset
#y_pred = clf.predict(X_test)
