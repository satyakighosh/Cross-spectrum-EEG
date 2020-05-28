import numpy as np
import time
import resource
import os
import random
from collections import Counter
import psutil
import humanize
import math
from line_profiler import LineProfiler


from extract import extract_anns, extract_data
from constants import *
from features import feature_gen
import line_profiler

start = time.time()
patient_list = sorted(os.listdir(TRAIN_DATA_PATH))
labels_global=[]
no_of_errors_encountered=0

class DatasetBuilder:
  @profile
  def __init__(self, ref_label, size): #size->number of patients from which segment bank will be made
    
    self.size = size
    self.data_list = []
    self.ref_label = ref_label
    self.ref_segments = np.load('/content/drive/My Drive/cross/Cross-spectrum-EEG-master/reference_segments.npy', allow_pickle=True).reshape(-1, 1)[0][0]
    
    print(f"Refs loaded in {time.time()-start} seconds")

  @profile
  def extract_random_segments_for_given_patient_during_warning(self,segment_label, patient_no):   #during warning related to AR(l)__autocorrelation lag

    current_patient_ = patient_list[patient_no]  
    patient_ann_ = current_patient_[:-4] + '-nsrr.xml'
    ann_, onset_, duration_ = extract_anns(TRAIN_ANN_PATH + patient_ann_)
    eeg_dict_, info_dict_ = extract_data(TRAIN_DATA_PATH + current_patient_, ann_, onset_, duration_[-1])
    return (int(segment_label),eeg_dict_[segment_label][np.random.choice(len(eeg_dict_[segment_label])-1)])

  @profile  
  def generate_features_with_ref_segments(self, selected_tuple,patient_no):
      
    selected_label = selected_tuple[0]
    selected_segment = selected_tuple[1]
    t, s = np.arange(len(selected_segment)), np.array(selected_segment)
    #l = 1 if selected_label == self.ref_label else 0
    F_avg = []
    for ref_segment in self.ref_segments[self.ref_label]:
      t1, s1 = t, s
      t2, s2 = np.arange(len(ref_segment)), np.array(ref_segment)
      # print(s1.shape)
      # print(s2.shape)

      #converting segments to equal lengths
      S1 = s1[np.argwhere((t1 >= min(t2)) & (t1 <= max(t2))).flatten()]
      S2 = s2[np.argwhere((t2 >= min(t1)) & (t2 <= max(t1))).flatten()]

      # print(s1.shape)
      # print(s2.shape)
      # print("*************************")
      try:
        F = feature_gen(t1, S1, t2, S2,self.ref_label)
        for i in range(len(F)):
          if math.isnan(F[i]):
            F[i]=0

        F_avg.append(F)
        #for i in range(len(F)):
          #print(f"{F[i]}({i})", end=", ")
      except Warning:
        global no_of_errors_encountered
        no_of_errors_encountered+=1
        substitute=self.extract_random_segments_for_given_patient_during_warning(selected_label,patient_no)
        self.generate_features_with_ref_segments(substitute,patient_no)
      
    #print(f"Feature vector:{np.mean(F_avg,axis=0)}")

    
    self.data_list.append((selected_label, np.mean(F_avg, axis=0)))
    #print(f"Shape of data_list[{self.ref_label}]={number_of_segments_for_ith_key}")
    #print(f"Feature vector extracted:{self.data_list[number_of_segments_for_ith_key-1]}")

 
  @profile
  def extract_random_segments_for_given_patient(self, patient_no, num_segs_chosen):   #helper

    current_patient = patient_list[patient_no]  
    patient_ann = current_patient[:-4] + '-nsrr.xml'
    ann, onset, duration = extract_anns(TRAIN_ANN_PATH + patient_ann)
    eeg_dict, info_dict = extract_data(TRAIN_DATA_PATH + current_patient, ann, onset, duration[-1])
    len_dict = {}
    
    for i in eeg_dict.keys(): 
      len_dict[i] = len(eeg_dict[i])
      #print(f"eeg_dict{i} is {eeg_dict[i]}")
      #random.shuffle(eeg_dict[i])
    #print(len_dict)

    tuples = []    #all (label, segment)
    for label in eeg_dict.keys():
      for seg in range(len_dict[label]): 
        tuples.append((int(label), eeg_dict[label][seg]))


    random.shuffle(tuples)

    selected_tuples = []
    for i in range(num_segs_chosen):
      selected_tuples.append(tuples.pop())   #popping after shuffling equivalent to sampling randomly
      print(f"{i}th segment chosen of label {selected_tuples[i][0]} from patient {patient_no}")
    #print(f"RAM: {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024} MB")   
    del tuples
    print(f"RAM: {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024} MB")   
    process = psutil.Process(os.getpid())
    print("Gen RAM Free: " + humanize.naturalsize( psutil.virtual_memory().available ), \
          " | Proc size: " + humanize.naturalsize( process.memory_info().rss))

  
    for t in selected_tuples:
      yield t

  @profile
  def create_dataset_of_particular_stage(self, num_segs_chosen):      #main
  
    segs_global = []
    for p in range(self.size): 
      segs = []
      print(f"SVM_id: {self.ref_label}, patient_no: {p}")
      segment_generator = self.extract_random_segments_for_given_patient(patient_no=p, num_segs_chosen=num_segs_chosen)
      for segment in segment_generator:
        try:
          print(segment[0])
          self.generate_features_with_ref_segments(segment,p)
          segs.append(segment[0])  #just appending the label for now, can save segment[1] for actual segment
        except Warning:
          print("Warning encountered...")
          pass
      segs_global = segs_global + segs
      print(f"segs: {np.unique(segs, return_counts=True)}")  #for this patient only
      print(f"segs_global: {np.unique(segs_global, return_counts=True)}")
      print(f"Time taken so far: {time.time()-start} seconds")
      print("\n")
      np.save(f"/content/drive/My Drive/cross/Cross-spectrum-EEG-master/clf{ref_label}.npy", dataset.data_list)
      
    print("########################")
    print("\n")
    
    print(f"segs_global: {np.unique(segs_global, return_counts=True)}")    #accumulating over all patients



ref_label = 2
dataset = DatasetBuilder(size=150, ref_label=ref_label)  #size->number of patients from which random segment  will be chosen

dataset.create_dataset_of_particular_stage(num_segs_chosen=45)
print(f"saving clf{ref_label} dataset...")

np.save(f"/content/drive/My Drive/cross/Cross-spectrum-EEG-master/clf{ref_label}.npy", dataset.data_list)
print(f"Total errors encounterd: {no_of_errors_encountered}")
print(f"Total time: {time.time()-start} seconds")