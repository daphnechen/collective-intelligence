import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from hmm import unsupervised_HMM

import itertools
import random

import os
from pprint import pprint

import csv

df = pd.read_csv('../data/50_salads_data_test.csv')

# master_file = open('master_file.csv', w+)
# writer = csv.writer(master_file))

# with open('master_file.csv', w+) as f:
#     writer = csv.writer(master_file))
#     entry = 
#     writer.writerow(entry)

# master_file.close()


"""
    This file generates a synthetic dataset from 50 salads dataset.
    Per discussion with Reid: dictionary to enumerate through the transitions 
        between states.

    This file heavily borrows from learn_50.py which is from learn_ci.py (Michelle Zhao)
"""

# Determine the number of unique actions in the input dataset, df, as a list
action_list = list(df['action'].unique())
print("action_list", action_list)
# print("number of unique actions: ", len(action_list))
num_unique = len(action_list)


# Create a dictionary to create a unique int to represent each action
count = 0
action_dict = {}
for a in action_list:
    action_dict[a] = count
    count += 1
print("action_dict: \n", action_dict)


# TODO: keep count of the number of unique actions from action_list


# Use original annotation files from 50 salads dataset
#   to build the input for the HMM (or other model)
annotation_dir = '../../../data/ann-ts/activityAnnotations'
hmm_input_X = []

transitions = {}
# lettuce, tomato, cheese, salt, pepper, vinegar, oil, dressing, cucumber
# where each int can take a value of {0,1,2,3} for {none,prep,core,post}
state_vec = [0,0,0,0,0,0,0,0]
# csv_header = ['user_id', 'video_num', 'start_t', 'end_t', 'action']
csv_header = ['start_t', 'end_t', 'action']

with open('master_file.csv', 'w+') as m_f:
    writer = csv.writer(m_f) # delimiter=','
    writer.writerow(csv_header)

    for filename in os.listdir(annotation_dir):
        f = os.path.join(annotation_dir, filename)
        if os.path.isfile(f):
            with open(f, 'r', encoding='utf-8') as ann_file:
                if f.endswith('.txt'):
                    data = ann_file.readlines()
                    cur_activity = []
                    # print("cur_activity: \n", cur_activity)
                    for line in data:
                        
                        # writer.writerow(line)
                        print(line)

                        # print("line: \n", line)
                        annotation = line.split(' ')[2]
                        annotation = annotation.strip('\n')
                        annotation_step = annotation.split('_')[-1] # get last element {prep, core, post}
                        # print("annotation: \n", annotation)

                        # check ingredient
                        action_segments = annotation.split('_')
                        # if 'lettuce' in action_segments:
                        #     state_idx = 0
                        # elif 'tomato' in action_segments:
                        #     state_idx = 1
                        # elif 'cheese' in action_segments:
                        #     state_idx = 2
                        # elif 'salt' in action_segments:
                        #     state_idx = 3
                        # elif 'pepper' in action_segments:
                        #     state_idx = 4
                        # elif 'vinegar' in action_segments:
                        #     state_idx = 5
                        # elif 'oil' in action_segments:
                        #     state_idx = 6
                        # elif 'dressing' in action_segments:
                        #     state_idx = 7
                        # elif 'cucumber' in action_segments:
                        # else:
                        #     # TODO: throw error
                        #     continue

                        # # TODO: initialize as non existent yet
                        # if annotation_step == 'prep':
                        #     #TODO
                        #     transitions[] = 
                        # elif annotation_step == 'core':
                        #     #TODO
                        # elif annotation_step == 'post':
                        #     #TODO
                        # else:
                        #     #TODO: throw exception
                        
                        if annotation != '':
                            annotation_as_int = action_dict[annotation]
                            # TODO: correct this after completing the dataset csv.
                            # Hacky code to account for yet-unseen actions.
                            if annotation_as_int == 54:
                                annotation_as_int = 53
                            elif annotation_as_int == 55:
                                annotation_as_int = 22
                            cur_activity.append(annotation_as_int)
                    hmm_input_X.append(cur_activity)

# pprint(hmm_input_X)
# print(hmm_input_X)
hmm_input_X = np.array(hmm_input_X)
# np.save('salad_data.npy', hmm_input_X)
# print("INPUT to HMM: \n", hmm_input_X)
# print("num sequences: \n", len(hmm_input_X))