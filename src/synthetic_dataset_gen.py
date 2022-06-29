import json
import numpy as np
from pprint import pprint
import random



def get_counts():
    '''
        This function returns a dictionary that represents the
        counts between a state and the states that it transitions to,
        where key is the state and value is a dictionary containing the 
        transitioned-to states and their counts
    '''

    count_matrix = {}

    with open('annotations_segmented.txt', 'r') as f:
        annotations = f.read().strip().split('\n')

    # get count of each annotation
    for i in range(len(annotations)-1):
        cur_annotation = annotations[i]
        next_annotation = annotations[i+1]

        if cur_annotation not in count_matrix:
            count_matrix[cur_annotation] = {}

        if next_annotation not in count_matrix[cur_annotation]:
            count_matrix[cur_annotation][next_annotation] = 0

        count_matrix[cur_annotation][next_annotation] += 1

    return count_matrix


def get_probabilities(count_matrix):
    '''
        args: 
            count_matrix: a dictionary that counts the number of
            state transitions for every state in 50_salads, given
            by the get_counts() function

        This function returns a probability matrix as a dictionary
        where keys are the state and values are a dictionary of states
        and their probability of transition to next states
    '''

    probability_matrix = {}

    for i, annotation in enumerate(count_matrix):

        total_annotation_count = 0

        for count in count_matrix[annotation].values():
            total_annotation_count += count  # denominator

        for next_annotation, next_annotation_count in count_matrix[annotation].items():
            probability = float(next_annotation_count) / float(total_annotation_count)

            if annotation not in probability_matrix:
                probability_matrix[annotation] = {}

            probability_matrix[annotation][next_annotation] = probability

    return probability_matrix


def generate_sequence(start_state, probability_matrix):
    '''
        args: 
            start_state: a str representing the starting state
                         for the sequence
            probability_matrix: a dictionary that contains states
                         and their transition probabilities to
                         other states, given by the get_probabilities()
                         function

        This function returns a sequence as a list of str, 
        generated from a provided starting state and its 
        corresponding probability matrix.
        Uses a random seed.
        # TODO: sequence of variable length
    '''

    sequence = []

    start = start_state

    sequence.append(start)

    end_state = ''
    while end_state != 'end':

        annotations = list(probability_matrix[start].keys())
        probs = list(probability_matrix[start].values())

        next_annotation = random.choices(annotations, weights=probs)[0]

        start = next_annotation

        sequence.append(next_annotation)

        end_state = next_annotation

    return sequence


starting_states = ['peel_cucumber_prep', # 10
                   'cut_cucumber_prep',  # 1
                   'cut_tomato_prep',    # 7
                   'cut_lettuce_prep',   # 7
                   'cut_cheese_prep',    # 3
                   'add_oil_prep',       # 13
                   'add_vinegar_prep',   # 6
                   'add_salt_prep',      # 4
                   'add_pepper_prep'     # 3
                   ]

starting_states_counts = [('peel_cucumber_prep', 10),
                          ('cut_cucumber_prep', 1),
                          ('cut_tomato_prep', 7),
                          ('cut_lettuce_prep', 7),
                          ('cut_cheese_prep', 3),
                          ('add_oil_prep', 13),
                          ('add_vinegar_prep', 6),
                          ('add_salt_prep', 4),
                          ('add_pepper_prep', 3)
                          ]

def generate_dataset():
    count_matrix = get_counts()
    probability_matrix = get_probabilities(count_matrix)

    idx = 0
    with open('synthetic_data.json', 'w', encoding='utf-8') as f:
        for start_state, count in starting_states_counts:
            for i in range(count * 10):
                seq = generate_sequence(start_state, probability_matrix)
                json.dump({idx : seq}, f, indent=4)
                idx += 1

generate_dataset()

# pprint(get_counts())

# count_matrix = get_counts()
# pprint(get_probabilities(count_matrix))

# probability_matrix = get_probabilities(count_matrix)
# pprint(generate_sequence('peel_cucumber_prep', probability_matrix))

