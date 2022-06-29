import numpy as np
from pprint import pprint
import random



def get_counts():

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


def generate_sequence(probability_matrix):

    sequence = []

    start = 'peel_cucumber_prep'

    sequence.append(start)

    count = 0
    while count < 25:

        annotations = list(probability_matrix[start].keys())
        probs = list(probability_matrix[start].values())

        next_annotation = random.choices(annotations, weights=probs)[0]

        start = next_annotation

        sequence.append(next_annotation)

        count += 1

    pprint(sequence)


# TODO end state


# pprint(get_counts())

count_matrix = get_counts()
# pprint(get_probabilities(count_matrix))

probability_matrix = get_probabilities(count_matrix)
generate_sequence(probability_matrix)

