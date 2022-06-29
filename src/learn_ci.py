import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from hmm import unsupervised_HMM

import itertools
import random

df = pd.read_csv('../notebooks/team_data.csv')

map_letter = 'A'

global_mean_walking = np.mean(df['walk_mean'])
global_mean_anxiety = np.mean(df['Anxiety_average'])

print("global_mean_walking", global_mean_walking)
print("global_mean_anxiety", global_mean_anxiety)



teams_list = list(df['team_id'].unique())
print("teams_list", teams_list)

average_minute_skill = {}
average_minute_workload = {}
average_minute_effort = {}

for map_letter in ['A', 'B']:
    average_minute_skill[map_letter] = {}
    average_minute_workload[map_letter] = {}
    average_minute_effort[map_letter] = {}
    for t in range(1, 16):
        average_minute_skill[map_letter][t] = []
        average_minute_workload[map_letter][t] = []
        average_minute_effort[map_letter][t] = []

for map_letter in ['A', 'B']:
    for t in range(1, 16):
        keyname = f'skilluse{t}_Saturn{map_letter}'
        average_minute_skill[map_letter][t].extend(list(df[keyname].to_numpy()))
        keyname = f'workload_burnt{t}_Saturn{map_letter}'
        average_minute_workload[map_letter][t].extend(list(df[keyname].to_numpy()))
        keyname = f'effort{t}_Saturn{map_letter}'
        average_minute_effort[map_letter][t].extend(list(df[keyname].to_numpy()))

for map_letter in ['A', 'B']:
    for t in range(1, 16):
        average_minute_skill[map_letter][t] = np.mean(average_minute_skill[map_letter][t])

# Classify teams into 4 groups


# Group 1 = High Walking, High Anxiety
# Group 2 = Low Walking, High Anxiety
# Group 3 = High Walking, Low Anxiety
# Group 4 = Low Walking, Low Anxiety

teams_list = df['team_id'].to_numpy()

team_walking = df['walk_mean'].to_numpy()
team_anxiety = df['Anxiety_average'].to_numpy()

team_id_to_group = {}
group_to_teams = {1: [], 2: [], 3: [], 4: []}

for i in range(len(teams_list)):
    team_id = teams_list[i]
    walk = team_walking[i]
    anx = team_anxiety[i]
    if walk >= global_mean_walking:
        if anx >= global_mean_anxiety:
            group = 1
        else:
            group = 3
    else:
        if anx >= global_mean_anxiety:
            group = 2
        else:
            group = 4

    team_id_to_group[team_id] = group
    group_to_teams[group].append(team_id)

group_no = 1
teams_in_group = group_to_teams[group_no]

# Get Minute by Minute Averages for Group
average_minute_skill = {}
average_minute_workload = {}
average_minute_effort = {}

for map_letter in ['A', 'B']:
    average_minute_skill[map_letter] = {}
    average_minute_workload[map_letter] = {}
    average_minute_effort[map_letter] = {}
    for t in range(1, 16):
        average_minute_skill[map_letter][t] = []
        average_minute_workload[map_letter][t] = []
        average_minute_effort[map_letter][t] = []

for map_letter in ['A', 'B']:
    for t in range(1, 16):
        team_df = df[df['team_id'].isin(teams_in_group)]

        keyname = f'skilluse{t}_Saturn{map_letter}'
        average_minute_skill[map_letter][t].extend(list(team_df[keyname].to_numpy()))
        keyname = f'workload_burnt{t}_Saturn{map_letter}'
        average_minute_workload[map_letter][t].extend(list(team_df[keyname].to_numpy()))
        keyname = f'effort{t}_Saturn{map_letter}'
        average_minute_effort[map_letter][t].extend(list(team_df[keyname].to_numpy()))

for map_letter in ['A', 'B']:
    for t in range(1, 16):
        average_minute_skill[map_letter][t] = np.mean(average_minute_skill[map_letter][t])
        average_minute_workload[map_letter][t] = np.mean(average_minute_workload[map_letter][t])
        average_minute_effort[map_letter][t] = np.mean(average_minute_effort[map_letter][t])

# Get team minute binary values and unique states
team_to_state = {}
unique_states = []
for team_id in teams_in_group:
    team_to_state[team_id] = {}
    for map_letter in ['A', 'B']:
        state_vector = []
        for t in range(1, 16):
            keyname = f'skilluse{t}_Saturn{map_letter}'
            skill = df[df['team_id'] == team_id][keyname].to_numpy()[0]
            if skill < average_minute_skill[map_letter][t]:
                skill = 0
            else:
                skill = 1

            keyname = f'workload_burnt{t}_Saturn{map_letter}'
            workload = df[df['team_id'] == team_id][keyname].to_numpy()[0]
            if workload < average_minute_workload[map_letter][t]:
                workload = 0
            else:
                workload = 1

            keyname = f'effort{t}_Saturn{map_letter}'
            effort = df[df['team_id'] == team_id][keyname].to_numpy()[0]
            if effort < average_minute_effort[map_letter][t]:
                effort = 0
            else:
                effort = 1

            state = (skill, workload, effort)
            state_vector.append(state)
            unique_states.append(state)

        team_to_state[team_id][map_letter] = state_vector



unique_states_list = list(set(unique_states))

state_id_to_state = dict(enumerate(unique_states_list))
state_to_state_id = {v: k for k, v in state_id_to_state.items()}

team_to_state_id_sequence = {}
team_id_map_to_state_id_sequence = {}
for team_no in team_to_state:
    team_to_state_id_sequence[team_no] = {}
    for map_letter in ['A', 'B']:
        state_seq = [state_to_state_id[x] for x in team_to_state[team_no][map_letter]]
        team_to_state_id_sequence[team_no][map_letter] = state_seq
        team_id_map_to_state_id_sequence[(team_no, map_letter)] = team_to_state[team_no][map_letter]

hmm_input_X = []
index_to_team_map = {}
counter = 0
for team_no in team_to_state_id_sequence:
    for map_letter in ['A', 'B']:
        hmm_input_X.append(team_to_state_id_sequence[team_no][map_letter])
        index_to_team_map[counter] = (team_no, map_letter)
        counter += 1

hmm_input_X = np.array(hmm_input_X)
print("****** HMM input: ******", hmm_input_X)
N_iters = 100
n_states = 5

test_unsuper_hmm = unsupervised_HMM(hmm_input_X, n_states, N_iters)

# print('emission', test_unsuper_hmm.generate_emission(10))
hidden_seqs = {}
team_num_to_seq_probs = {}
for j in range(len(hmm_input_X)):
    # print("team", team_numbers[j])
    # print("reindex", X[j][:50])
    team_id_map = index_to_team_map[j]
    viterbi_output, all_sequences_and_probs = test_unsuper_hmm.viterbi_all_probs(hmm_input_X[j])
    team_num_to_seq_probs[team_id_map] = all_sequences_and_probs
    hidden_seqs[team_id_map] = [int(x) for x in viterbi_output]


