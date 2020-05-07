import numpy as np
import time
import os
import random
from collections import Counter

from extract import extract_anns, extract_data
from reference_builder import ReferenceBuilder
from constants import *
from features import feature_gen
from utils import get_len_dict


start = time.time()
patient_list = sorted(os.listdir(TRAIN_DATA_PATH))
labels_global=[]

class DatasetBuilder:

  def __init__(self, size): #size->number of patients from which segment bank will be made
    self.size = size
    #self.segment_bank = []                 #list of tuples or random segments from random patients
    self.data_dict = {0:[], 1:[], 2:[], 3:[], 4:[], 5:[]}
    
    
    ref_builder = ReferenceBuilder(REF_SEG_PER_CLASS, TRAIN_DATA_PATH, TRAIN_ANN_PATH)
    ref_builder.build_refs()
    print(f"Reference building took {time.time()-start} seconds")
    get_len_dict(ref_builder.reference_segments)
    self.ref_segments = ref_builder.reference_segments
  



  def extract_random_segment_with_specific_label(self,label):  
    
    random_patient = list(np.random.choice(os.listdir(TRAIN_DATA_PATH), size=1, replace=True))[0] #select random patient
    patient_ann = random_patient[:-4] + '-nsrr.xml'
    ann, onset, duration = extract_anns(TRAIN_ANN_PATH + patient_ann)
    eeg_dict, info_dict = extract_data(TRAIN_DATA_PATH + random_patient, ann, onset)
    #len_dict = {}
    #for i in eeg_dict.keys(): 
    # len_dict[i] = len(eeg_dict[i])
    while len(eeg_dict[label])==0:
    
      random_patient = list(np.random.choice(os.listdir(TRAIN_DATA_PATH), size=1, replace=True))[0] #select random patient
      patient_ann = random_patient[:-4] + '-nsrr.xml'
      ann, onset, duration = extract_anns(TRAIN_ANN_PATH + patient_ann)
      eeg_dict, info_dict = extract_data(TRAIN_DATA_PATH + random_patient, ann, onset)
      
    
    #if len_dict[4] and np.random.random() < 0.3 != 0:    #to increase probability of getting very rare label 4
    #  label = 4
    #else:
    #indices = [i for i in len_dict.keys() if len_dict[i] != 0]
    #print(indices)
    #label = int(np.random.choice(indices))
    print(info_dict)
    
    #label = np.random.choice()
    seg_no = np.random.choice(len(eeg_dict[label]))  #select random segment of a random class of a random patient
    return (label, eeg_dict[label][seg_no])

  def generate_segment_pairs(self, selected_tuple, label):
    #need an online training scheme
      
    selected_label = selected_tuple[0]
    selected_segment = selected_tuple[1]
    t, s = np.arange(len(selected_segment)), np.array(selected_segment)
    #l = 1 if selected_label == label else 0
    F_avg = []
    for ref_segment in self.ref_segments[label]:
      t1, s1 = t, s
      t2, s2 = np.arange(len(ref_segment)), np.array(ref_segment)
      # print(s1.shape)
      # print(s2.shape)

      #converting segments to equal lengths
      s1 = s1[np.argwhere((t1 >= min(t2)) & (t1 <= max(t2))).flatten()]
      s2 = s2[np.argwhere((t2 >= min(t1)) & (t2 <= max(t1))).flatten()]

      # print(s1.shape)
      # print(s2.shape)
      # print("*************************")
      F = feature_gen(t1, s1, t2, s2)
      F_avg.append(F)
    
    self.data_dict[label].append((selected_label, np.mean(F_avg, axis=0)))
    number_of_segments_for_ith_key=len(self.data_dict[label])
    #print(f"Shape of data_dict[{label}]={number_of_segments_for_ith_key}")
    print(f"Feature vector extracted:{self.data_dict[label][number_of_segments_for_ith_key-1]}")



  def extract_random_segments_for_given_patient(self,patient_no, label):   #helper

    global labels_global         #for getting an idea of distribution, not essential

    current_patient = patient_list[patient_no]  
    patient_ann = current_patient[:-4] + '-nsrr.xml'
    ann, onset, duration = extract_anns(TRAIN_ANN_PATH + patient_ann)
    eeg_dict, info_dict = extract_data(TRAIN_DATA_PATH + current_patient, ann, onset)
    len_dict = {}
    
    for i in eeg_dict.keys(): 
      len_dict[i] = len(eeg_dict[i])
      #print(f"eeg_dict{i} is {eeg_dict[i]}")
      random.shuffle(eeg_dict[i])


    labels = []
    
    if len_dict[label] != 0:    
      num_seg_label = np.maximum(len_dict[label]//8, 1)           #positives
      if label==4:
        num_seg_label=len_dict[label]
      if label==5:
        num_seg_label=len_dict[label]//4
      #print(f"len_dict(positive):{len_dict[label]}Number of positive segments to be taken:{num_seg_label}")

      for ith_positive_segment in range(num_seg_label):                #adding the positive segments sequentially
        #print(f"index of the positive segment{ith_positive_segment}")
        yield(int(label),eeg_dict[label][ith_positive_segment])
      labels = labels + list(np.ones(num_seg_label, dtype=int)*label)  #adding postive labels
      
      label_complement = list(range(NUM_SLEEP_STAGES))      #negatives
      label_complement.remove(label)                        #removing the positive label
      
      for l in label_complement:
        if len_dict[l]!=0:
          num_seg = np.maximum(num_seg_label//20, 1)
          if l == 4 and len_dict[l] != 0:                   #special case
            num_seg = np.maximum(num_seg_label, 1)
          if l == 5 and len_dict[l] != 0:                   #special case
            num_seg = np.maximum(num_seg_label//5, 1)
          if num_seg!=0:
          #print(f"len_dict(negative):{len_dict[l]} Number of negative segments to be taken:{num_seg}")
            for ith_neg_segment in range(num_seg):
            #print(f"{ith_neg_segment}")
              yield(int(l),eeg_dict[l][ith_neg_segment])              ##adding the negative segments sequentially
          labels = labels + list(np.ones(num_seg, dtype=int)*l)   #adding negative labels

    
    #labels_global = labels_global + labels


  def create_dataset_of_particular_stage(self,label):      #main
  
    labels_global = []
    segs_global = []
    #for p in range(len(patient_list)//2): 
    for p in range(len(patient_list)): #patient loop, for 5 patients only for understanding purpose
      segs = []
      print(f"SVM_id: {label}, patient_no: {p}")
      segment_generator = dataset.extract_random_segments_for_given_patient(patient_no=p, label=label)
      for segment in segment_generator:
        # print(segment[0], end=', ')
        self.generate_segment_pairs(segment,label)
        segs.append(segment[0])  #just appending the label for now, can save segment[1] for actual segment
      segs_global = segs_global + segs
      print(f"segs: {np.unique(segs, return_counts=True)}")  #for this patient only
      print("\n")
      
    print("########################")
    print("\n")
    
    print(f"segs_global: {np.unique(segs_global, return_counts=True)}")    #accumulating over all patients

'''
  def create_dataset(self):
    
    NUM_DATA_SEGMENTS_TO_PAIR_WITH = self.size   #num segments to pair with all the ref segments
    selected_labels = []
    
    #this segment creates positive examples for the SVMs  
    for key in range(NUM_SLEEP_STAGES):
      for i in range(NUM_DATA_SEGMENTS_TO_PAIR_WITH):
        print(f"Generating {i}th positive example for SVM-{key}:")
        tuple_positive_label=self.extract_random_segment_with_specific_label(key)
        print(f"selected_label : {tuple_positive_label[0]}")
        self.generate_segment_pairs(tuple_positive_label,key)
        print(f"Time elapsed: {time.time()-start} seconds")
        print("*************************")

    for i in range(NUM_DATA_SEGMENTS_TO_PAIR_WITH):
      
      #x = np.random.choice(list(range(len(self.segment_bank))), size=1)
      #selected_tuple = self.segment_bank[int(x)] #<-tuple
      selected_tuple = self.extract_random_segment()
      selected_labels.append(selected_tuple[0])
      print(f"selected_label : {selected_tuple[0]}")
      
      for label in range(NUM_SLEEP_STAGES):         #label of ref_segment
        self.generate_segment_pairs(selected_tuple, label)

      print(f"Pairing with patient {i+1} took {time.time()-start} seconds")
      print("*************************")
    random.shuffle(self.data_dict)
    print(Counter(selected_labels))
'''

    
'''
      for other_key in range(NUM_SLEEP_STAGES):
        if key==other_key:
          continue
        for j in range(NUM_NEGATIVE_EXAMPLES_EACH):
          tuple_negative_label=self.extract_random_segment_with_specific_label(other_key)
          print(f"selected_label : {tuple_negative_label[0]}")
          self.generate_segment_pairs(tuple_negative_label,key)
          print(f"Time elapsed: {time.time()-start} seconds")
          print("*************************")
'''






'''
  def extract_random_segment(self):   #have to try and make it class balanced
    
    random_patient = list(np.random.choice(os.listdir(TRAIN_DATA_PATH), size=1, replace=False))[0] #select random patient
    patient_ann = random_patient[:-4] + '-nsrr.xml'
    ann, onset, duration = extract_anns(TRAIN_ANN_PATH + patient_ann)
    eeg_dict, info_dict = extract_data(TRAIN_DATA_PATH + random_patient, ann, onset)
    len_dict = {}
    for i in eeg_dict.keys(): 
      len_dict[i] = len(eeg_dict[i])
    
    if len_dict[4] and np.random.random() < 0.3 != 0:    #to increase probability of getting very rare label 4
      label = 4
    else:
      indices = [i for i in len_dict.keys() if len_dict[i] != 0]
      print(indices)
      label = int(np.random.choice(indices))
    print(info_dict)
    
    #label = np.random.choice()
    seg_no = np.random.choice(len(eeg_dict[label]))  #select random segment of a random class of a random patient
    return (label, eeg_dict[label][seg_no])
    

  
  def create_segment_bank(self):   #have to try and make it class balanced
    
    patient_edfs = list(np.random.choice(os.listdir(TRAIN_DATA_PATH), size=self.size, replace=False))
    labels = list(range(NUM_SLEEP_STAGES-3))
    for patient in patient_edfs:
      patient_ann = patient[:-4] + '-nsrr.xml'
      ann, onset, duration = extract_anns(TRAIN_ANN_PATH + patient_ann)
      eeg_dict, info_dict = extract_data(TRAIN_DATA_PATH + patient, ann, onset)
      print(info_dict)
      print("*************************")
      for label in labels:
        #label = np.random.choice()
        seg_no = np.random.choice(len(eeg_dict[label]))
        self.segment_bank.append((label, eeg_dict[label][seg_no]))
  '''


dataset = DatasetBuilder(size=NUM_CHOSEN_PATIENTS)  #size->number of patients from which random segment  will be chosen
#dataset.create_segment_bank()
#dataset.create_dataset() #already shuffled

for idx in range(NUM_SLEEP_STAGES):
  dataset.create_dataset_of_particular_stage(idx)
  random.shuffle(dataset.data_dict[idx])
  if(idx==0):
    print(f"saving svm{idx}dataset...")
    np.save('/content/drive/My Drive/Cross-spectrum-EEG/svm0.npy', dataset.data_dict[idx])
  if(idx==1):
    print(f"saving svm{idx}dataset...")
    np.save('/content/drive/My Drive/Cross-spectrum-EEG/svm1.npy', dataset.data_dict[idx])
  if(idx==2):
    print(f"saving svm{idx}dataset...")
    np.save('/content/drive/My Drive/Cross-spectrum-EEG/svm2.npy', dataset.data_dict[idx])
  if(idx==3):
    print(f"saving svm{idx}dataset...")
    np.save('/content/drive/My Drive/Cross-spectrum-EEG/svm3.npy', dataset.data_dict[idx])
  if(idx==4):
    print(f"saving svm{idx}dataset...")
    np.save('/content/drive/My Drive/Cross-spectrum-EEG/svm4.npy', dataset.data_dict[idx])
  if(idx==5):
    print(f"saving svm{idx}dataset...")
    np.save('/content/drive/My Drive/Cross-spectrum-EEG/svm5.npy', dataset.data_dict[idx])




#print("{" + "\n".join("{!r}: {!r},".format(k, v) for k, v in dataset.data_dict.items()) + "}")

get_len_dict(dataset.data_dict)
# print(dataset.data_dict[0])
# print(dataset.data_dict[0][0])
# print(dataset.data_dict[0][0][0])
# print(dataset.data_dict[0][0][1].shape)
print("Saving dataset")
np.save('/content/drive/My Drive/Cross-spectrum-EEG/dataset_covering_all_patients.npy', dataset.data_dict)