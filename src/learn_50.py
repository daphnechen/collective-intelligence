import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from hmm import unsupervised_HMM

import itertools
import random

import os
from pprint import pprint
import json

# df = pd.read_csv('../data/50_salads_data_test.csv')
df = pd.read_csv('new_data.csv')

# synthetic_data = pd.read_json('synthetic_data.json', lines=True, orient='index')
s_d_f = open('synthetic_data.json', 'r')
synthetic_data = json.load(s_d_f)

unique_values = set()
for seq in synthetic_data:
    for v in synthetic_data[seq]:
        unique_values.add(v)
    # pprint(synthetic_data[seq])

pprint(unique_values)

# action_list = list(df['action'].unique())
action_list = list(unique_values)
# print("action_list", action_list)
print("number of unique actions: ", len(action_list))
# num_unique = len(action_list)

#####

count = 0
action_dict = {}
for a in action_list:
    action_dict[a.strip('\n')] = count
    count += 1

pprint(action_dict)

#####

# annotation_dir = '../../../data/ann-ts/activityAnnotations'
# annotation_dir = '../../../data/ann-ts/activityAnnotations/train'
# hmm_input_X = []

# ## for the "real" 50 salads data
# for filename in os.listdir(annotation_dir):
#     f = os.path.join(annotation_dir, filename)
#     if os.path.isfile(f):
#         with open(f, 'r', encoding='utf-8') as ann_file:
#             if f.endswith('.txt'):
#                 data = ann_file.readlines()
#                 cur_activity = []
#                 for line in data:
#                     annotation = line.split(' ')[2]
#                     annotation = annotation.strip('\n')
                    
#                     if annotation != '':
#                         annotation_as_int = action_dict[annotation]
#                         # if annotation_as_int == 54:
#                         #     annotation_as_int = 53
#                         # elif annotation_as_int == 55:
#                         #     annotation_as_int = 22 # TODO change
#                         cur_activity.append(annotation_as_int)
#                 hmm_input_X.append(cur_activity)

## for the synthetic data
## create the hmm input as an array of lists of ints
## serialize the annotations as ints
hmm_input_X = []
for seq in synthetic_data:
    cur_activity = []
    for annotation in synthetic_data[seq]:
        annotation_as_int = action_dict[annotation]
        cur_activity.append(annotation_as_int)
    hmm_input_X.append(cur_activity)


pprint(hmm_input_X)
hmm_input_X = np.array(hmm_input_X)
np.save('salad_data_synthetic.npy', hmm_input_X)
# print("INPUT to HMM: \n", hmm_input_X)
print("num sequences: \n", len(hmm_input_X))

#####

# Check for missing states
aggregate = []
for arr in hmm_input_X:
    aggregate.extend(arr)
unique = set(aggregate)
unique = sorted(unique)
# print(unique)

#####

N_iters = 100
n_states = 20

## Run this to generate HMM
test_unsuper_hmm = unsupervised_HMM(hmm_input_X, n_states, N_iters)
print('emission', test_unsuper_hmm.generate_emission(30))  # emission of length 30

# -------

# hidden_seqs = {}
# team_num_to_seq_probs = {}
# for j in range(len(hmm_input_X)):
#     # print("team", team_numbers[j])
#     # print("reindex", X[j][:50])
#     team_id_map = index_to_team_map[j]
#     viterbi_output, all_sequences_and_probs = test_unsuper_hmm.viterbi_all_probs(hmm_input_X[j])
#     team_num_to_seq_probs[team_id_map] = all_sequences_and_probs
#     hidden_seqs[team_id_map] = [int(x) for x in viterbi_output]


