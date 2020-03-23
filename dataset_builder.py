import numpy as np
import time
import os
import random

from extract import extract_anns, extract_data
from reference_builder import ReferenceBuilder
from constants import *
from features import feature_gen
from utils import get_len_dict


start = time.time()

class TrainDatasetBuilder:

  def __init__(self, size): #size->number of patients from which segment bank will be made
    self.size = size
    self.segment_bank = []                 #list of tuples or random segments from random patients
    self.train_set = {0:[], 1:[], 2:[], 3:[], 4:[], 5:[]}
    
    
    ref_builder = ReferenceBuilder(REF_SEG_PER_CLASS, TRAIN_DATA_PATH, TRAIN_ANN_PATH)
    ref_builder.build_refs()
    print(f"Reference building took {time.time()-start} seconds")
    get_len_dict(ref_builder.reference_segments)
    self.ref_segments = ref_builder.reference_segments

  def extract_random_segment(self):   #have to try and make it class balanced
    
    random_patient = list(np.random.choice(os.listdir(TRAIN_DATA_PATH), size=1, replace=False))[0] #select random patient
    patient_ann = random_patient[:-4] + '-nsrr.xml'
    ann, onset, duration = extract_anns(TRAIN_ANN_PATH + patient_ann)
    eeg_dict, info_dict = extract_data(TRAIN_DATA_PATH + random_patient, ann, onset)
    len_dict = {}
    for i in eeg_dict.keys(): 
      len_dict[i] = len(eeg_dict[i])
    
    if len_dict[4] != 0:    #to increase probability of getting very rare label 4
      label = 4
    else:
      indices = [i for i in len_dict.keys() if len_dict[i] != 0]
      label = int(np.random.choice(indices))
    print(info_dict)
    print("*************************")
    #label = np.random.choice()
    seg_no = np.random.choice(len(eeg_dict[label]))  #select random segment of random patient
    return (label, eeg_dict[label][seg_no])
    

  '''
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

  def generate_segment_pairs(self, selected_tuple, label):     #need an online training scheme
    selected_label = selected_tuple[0]
    selected_segment = selected_tuple[1]
    t, s = np.arange(len(selected_segment)), np.array(selected_segment)
    #l = 1 if selected_label == label else 0
    F_rs_avg = []
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
      F_rs = feature_gen(t1, s1, t2, s2)
      F_rs_avg.append(F_rs)
    
    self.train_set[label].append((selected_label, np.mean(F_rs_avg, axis=0)))

  def create_trainset(self):
    
    NUM_DATA_SEGMENTS_TO_PAIR_WITH = 5   #num segments in bank to pair with all the ref segments
    
    for i in range(NUM_DATA_SEGMENTS_TO_PAIR_WITH):
      
      #x = np.random.choice(list(range(len(self.segment_bank))), size=1)
      #selected_tuple = self.segment_bank[int(x)] #<-tuple
      selected_tuple = self.extract_random_segment()
      print(f"selected_label : {selected_tuple[0]}")
      
      for label in range(NUM_SLEEP_STAGES):         #label of ref_segment
        self.generate_segment_pairs(selected_tuple, label)

      print(f"Pairing with patient {i+1} took {time.time()-start} seconds")
    
    random.shuffle(self.train_set)