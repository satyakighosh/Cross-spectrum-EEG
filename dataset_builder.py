import numpy as np
import time
import os
import random

from extract import extract_anns, extract_data
from reference_builder import ReferenceBuilder
from constants import *


start = time.time()

class TrainDatasetBuilder:

  def __init__(self, size): #size->number of patients from which segment bank will be made
    self.size = size
    self.segment_bank = []                 #list of tuples or random segments from random patients
    self.train_set = []
    
    
    ref_builder = ReferenceBuilder(REF_SEG_PER_CLASS, DATA_PATH, ANN_PATH)
    ref_builder.build_refs()
    print(f"Reference building took {time.time()-start} seconds")
    ref_builder.get_len_dict(ref_builder.reference_segments)
    self.ref_segments = ref_builder.reference_segments

  def create_segment_bank(self):   #have to try and make it class balanced
    
    patient_edfs = list(np.random.choice(os.listdir(DATA_PATH), size=self.size, replace=False))
    labels = list(range(NUM_SLEEP_STAGES-3))
    for patient in patient_edfs:
      patient_ann = patient[:-4] + '-nsrr.xml'
      ann, onset, duration = extract_anns(ANN_PATH + patient_ann)
      eeg_dict, info_dict = extract_data(DATA_PATH + patient, ann, onset)
      print(info_dict)
      print("*************************")
      for label in labels:
        #label = np.random.choice()
        seg_no = np.random.choice(len(eeg_dict[label]))
        self.segment_bank.append((label, eeg_dict[label][seg_no]))

  def generate_segment_pairs(self, selected_tuple, label):     #need an online training scheme
    selected_label = selected_tuple[0]
    selected_segment = selected_tuple[1]
    t, s = np.arange(len(selected_segment)), np.array(selected_segment)
    
    for ref_segment in self.ref_segments[label]:
      t1, s1 = t, s
      t2, s2 = np.arange(len(ref_segment)), np.array(ref_segment)
      print(s1.shape)
      print(s2.shape)

      #converting segments to equal lengths
      s1 = s1[np.argwhere((t1 >= min(t2)) & (t1 <= max(t2))).flatten()]
      s2 = s2[np.argwhere((t2 >= min(t1)) & (t2 <= max(t1))).flatten()]

      print(s1.shape)
      print(s2.shape)
      print("*************************")
      l = 1 if selected_label == label else 0
      self.train_set.append((l, s1, s2))

  def create_trainset(self):
    NUM_DATA_SEGMENTS_TO_PAIR_WITH = 5   #num segments in bank to pair with all the ref segments
    for i in range(NUM_DATA_SEGMENTS_TO_PAIR_WITH):
      x = np.random.choice(list(range(len(self.segment_bank))), size=1)
      selected_tuple = self.segment_bank[int(x)] #<-tuple
      for label in range(NUM_SLEEP_STAGES):
        self.generate_segment_pairs(selected_tuple, label)
      print(f"Pairing with patient {i+1} took {time.time()-start} seconds")
    random.shuffle(self.train_set)