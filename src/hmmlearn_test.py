from hmmlearn import hmm
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score

import nltk
from nltk import ngrams

from itertools import zip_longest
import json
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from pprint import pprint

from tqdm import tqdm

# load the data
data = np.load('salad_data_synthetic_new.npy', allow_pickle=True) # TODO make a filepath for the data
data = np.reshape(data,[len(data),1])

# format data from npy
formatted_data = []
lengths = []
for i in range(len(data)):
    formatted_data.append(data[i][0])
    lengths.append(len(data[i][0]))
# print(formatted_data)

# train an HMM and get the hidden states
model = hmm.GaussianHMM(n_components=20, n_iter=100)  # TODO change number of N from 50
model.fit(np.concatenate(formatted_data).reshape(-1, 1), lengths=lengths)
hidden_states = [model.predict(np.array(element).reshape(-1, 1)) for element in formatted_data]
# print(len(hidden_states), 'hidden states length')
# print('HIDDEN STATES: \n', hidden_states)

# --------------------------------------------------------------


def get_clusters(k, hidden_seqs):
    '''
        Helper method to get kmeans clusters
        Only works for sequences of equal length

        Args:
            k: number of clusters for k-means
            hidden_seqs: an NxH array of hidden state sequences, where
                N is the number of sequences and
                H is the number of hidden states in the sequence

        Returns:
            cluster_labels: the labels of the k clusters
            cluster_means: the coordinates of cluster centers
    '''
    features = []
    if type(hidden_seqs) is list:
        features = np.reshape(hidden_seqs, (-1, 1))
    else:
        features =  hidden_seqs.reshape(-1, 1)
    # print(len(features), 'features length')
    kmeans = KMeans(n_clusters=k).fit(features)
    cluster_labels = kmeans.labels_
    cluster_means = kmeans.cluster_centers_

    return cluster_labels, cluster_means


def get_clusters_as_ngrams(k, hidden_seqs):
    '''
        Helper method to get kmeans clusters
        Account for sequences of different length

        Args:
            k: number of clusters for k-means
            hidden_seqs: an Nx*H array of hidden state sequences, where
                N is the number of sequences and
                *H is a variable-length for number of hidden states in the sequence

        Returns:
            kmeans: the KMeans model instance
            cluster_labels: the labels of the k clusters
            cluster_means: the coordinates of cluster centers
    '''
    vocabulary = list()
    for element in hidden_seqs:
        element = element.tolist()
        vocabulary.append(str(','.join(str(e) for e in ['x' + str(e) for e in element])))
    
    vectorizer = CountVectorizer()
    matrix = vectorizer.fit_transform(vocabulary)
    matrix.toarray()

    kmeans = KMeans(n_clusters=k, random_state=0).fit(matrix)
    cluster_labels = kmeans.labels_
    cluster_means = kmeans.cluster_centers_

    return kmeans, cluster_labels, cluster_means


def evaluate_kmeans(hidden_seqs):
    '''
        Helper method to calculate the silhouette score 
        and inertia for kmeans clusters.

        Args:
            hidden_seqs: an Nx*H array of hidden state sequences, where
                N is the number of sequences and
                *H is a variable-length for number of hidden states in the sequence

        Returns:
            inertia_arr: an array of tuples containing (n_clusters, inertia)
            s_score_arr: an array of tuples containing (n_clusters, silhouette_score)
    '''
    inertia = 0
    s_score = 0
    step = 2  # step size n

    inertia_arr = []
    s_score_arr = []

    vocabulary = list()
    for element in hidden_seqs:
        element = element.tolist()
        vocabulary.append(str(','.join(str(e) for e in ['x' + str(e) for e in element])))

    # vocabulary = np.asarray(vocabulary, dtype=np.str)
    # v_input = np.empty((0, len(vocabulary)), dtype=str)
    v_input = np.array(vocabulary)
    # v_input = v_input.reshape(-1, 1)
    v_input = v_input[:len(vocabulary)].ravel()
    print('v_input shape: ', v_input.shape)
    # for i in range(len(vocabulary)):
    #     v_input = np.append(v_input, vocabulary[i], axis=0)

    # vocabulary = vocabulary.reshape(-1, 1)
    # print('this is the vocabulary list: ', v_input)

    vectorizer = CountVectorizer()
    # vectorizer = TfidfVectorizer(use_idf=True)
    matrix = vectorizer.fit_transform(v_input)
    # matrix.toarray()
    # matrix = np.array(matrix)
    matrix = matrix.reshape(-1, 1)
    # matrix = matrix[:len(vocabulary)].ravel()
    print("matrix shape: ", matrix.shape)

    previous = 0
    for n_clusters in tqdm(range(1, 20, step)):
        k_model = KMeans(
                       n_clusters=n_clusters,
                       init='k-means++',
                       max_iter=100,
                       verbose=False
        )
        k_model.fit_predict(matrix)
        inertia = k_model.inertia_
        print('inertia value: ', inertia)
        inertia_arr.append((n_clusters, inertia))
        diff = previous - inertia
        previous = inertia

        # print('matrix input: ', matrix)
        # s_score = silhouette_score(matrix, k_model)
        # s_score_arr.append((n_clusters, s_score))

    return inertia_arr #, s_score_arr

# --------------------------------------------------------------

# run clustering
k_model, labels, means = get_clusters_as_ngrams(7, hidden_states)
# print('labels: \n', labels)
# print('means: \n', means)

# --------------------------------------------------------------

# generate samples? try and plot these
# x, z = model.sample(500)
# fig, ax = plt.subplots()
# means_hmm = model.means_

# ax.plot(x[:, 0], x[:, 1], ".-", label="observations", ms=6,
#         mfc="orange", alpha=0.7)
# for i, m in enumerate(means_hmm):
#     ax.text(m[0], m[1], 'mean_%i' % (i + 1),
#             size=17, horizontalalignment='center',
#             bbox=dict(alpha=.7, facecolor='w'))
# ax.legend(loc='best')
# fig.show()

## visualize the clusters
# plt.scatter(range(len(hidden_states)), hidden_states, label="50 salads synthetic data", c=labels, cmap='rainbow')

# --------------------------------------------------------------

# plot just the first 5 sequences to minimize clutter
# fig, ax = plt.subplots()
# n = len(hidden_states)-535
# color = iter(cm.rainbow(np.linspace(0, 1, n)))
# for i in range(len(hidden_states)-535):
#     seq = hidden_states[i].tolist()
#     label=labels[i]
#     c = next(color)
#     ax.plot(seq, label="label: "+str(label), c=c)

# ax.legend(loc='best')
# ax.set_xlabel('step in sequence')
# ax.set_ylabel('hidden state #')
# fig.show()
# plt.show()
# plt.savefig('plot.png')

# --------------------------------------------------------------

# plot the inertia and silhouette score
# # inertia_arr, s_score_arr = evaluate_kmeans(hidden_states)
# inertia_arr = evaluate_kmeans(hidden_states)
# print(inertia_arr)
# xval = [x[0] for x in inertia_arr]
# yval = [x[1] for x in inertia_arr]
# plt.plot(xval, yval, color="blue")
# # plt.plot(s_score_arr, color="red")
# plt.xlabel('# clusters k')
# plt.ylabel('score')
# plt.xticks(np.arange(min(xval), max(xval)+1, 1.0))
# plt.show()

# --------------------------------------------------------------

# create the support sets // this should probably be in a separate file

# action_dict_lookup = {0: 'cut_lettuce_core', 1: 'place_cucumber_into_bowl_post', 2: 'cut_cheese_core', 3: 'add_oil_post', 4: 'cut_cucumber_prep', 5: 'place_tomato_into_bowl_post', 6: 'add_pepper_core', 7: 'cut_cucumber_core', 8: 'mix_ingredients_prep', 9: 'add_salt_prep', 10: 'place_lettuce_into_bowl_prep', 11: 'cut_cheese_prep', 12: 'cut_lettuce_prep', 13: 'serve_salad_onto_plate_core', 14: 'add_dressing_post', 15: 'place_cucumber_into_bowl_prep', 16: 'mix_ingredients_core', 17: 'end', 18: 'place_cheese_into_bowl_post', 19: 'add_salt_core', 20: 'place_lettuce_into_bowl_post', 21: 'mix_dressing_prep', 22: 'add_salt_post', 23: 'add_vinegar_prep', 24: 'serve_salad_onto_plate_prep', 25: 'mix_dressing_core', 26: 'place_cucumber_into_bowl_core', 27: 'peel_cucumber_core', 28: 'cut_cheese_post', 29: 'place_cheese_into_bowl_prep', 30: 'place_tomato_into_bowl_core', 31: 'add_pepper_post', 32: 'add_pepper_prep', 33: 'add_vinegar_core', 34: 'cut_cucumber_post', 35: 'cut_tomato_prep', 36: 'place_tomato_into_bowl_prep', 37: 'add_dressing_prep', 38: 'add_oil_core', 39: 'cut_lettuce_post', 40: 'place_cheese_into_bowl_core', 41: 'add_vinegar_post', 42: 'add_oil_prep', 43: 'cut_tomato_post', 44: 'cut_tomato_core', 45: 'mix_dressing_post', 46: 'mix_ingredients_post', 47: 'add_dressing_core', 48: 'place_lettuce_into_bowl_core', 49: 'peel_cucumber_post', 50: 'peel_cucumber_prep', 51: 'serve_salad_onto_plate_post'}

hs_map = {} # mapping between "label/cluster : (index, hidden state sequence)""
for i in range(len(labels)):
    value = (i, hidden_states[i].tolist())
    if str(labels[i]) in hs_map.keys(): # key exists
        hs_map[str(labels[i])].append(value)
    else: # key does not exist; create it
        hs_map[str(labels[i])] = [value]

with open('hs_map.json', 'w') as f:
    json.dump(hs_map, f, ensure_ascii=False, sort_keys=True, indent=4)
# print('THIS IS THE MAPPING \n', hs_map)
# print(json.dumps(hs_map, sort_keys=True, indent=4))

# use index to get corresponding value in formatted_data
support_sets = {}
for i, (label, values) in enumerate(hs_map.items()):
    for element in values:
        idx, _ = element
        
        if str(label) in support_sets.keys():
            support_sets[str(label)].append(formatted_data[idx])
        else:
            support_sets[str(label)] = [formatted_data[idx]]

with open('support_sets.json', 'w') as f:
    json.dump(support_sets, f, ensure_ascii=False, sort_keys=True, indent=4)
# print('THESE ARE THE SUPPORT SETS \n', support_sets)
# print(json.dumps(support_sets, sort_keys=True, indent=4))

# support sets with string annotations
# converted from ints in action_dict

# support_sets_annotations = {}
# for i, (label, values) in enumerate(hs_map.items()):
#     for element in values:
#         idx, _ = element
        
#         if str(label) in support_sets_annotations.keys():
#             converted_to_str = []
#             for j, int_annotation in enumerate(formatted_data[idx]):
#                 str_annotation = action_dict_lookup[int(int_annotation)]
#                 # converted_to_str.append(str_annotation)
#                 converted_to_str[j] = str_annotation
#             support_sets_annotations[str(label)].append(converted_to_str)
#         else:
#             converted_to_str = []
#             for j, int_annotation in enumerate(formatted_data[idx]):
#                 str_annotation = action_dict_lookup[int(int_annotation)]
#                 # converted_to_str.append(str_annotation)
#                 converted_to_str[j] = str_annotation
#             support_sets_annotations[str(label)] = [converted_to_str]

# with open('support_sets_annotations.json', 'w') as f:
#     json.dump(support_sets_annotations, f, ensure_ascii=False, sort_keys=True, indent=4)
# print('THESE ARE THE SUPPORT SETS W ANNOTATIONS \n', support_sets_annotations)

# --------------------------------------------------------------

# action_dict = {'serve_salad_onto_plate_prep': 0, 'cut_cheese_prep': 1, 'cut_tomato_post': 2, 'place_tomato_into_bowl_core': 3, 'mix_dressing_core': 4, 'end': 5, 'peel_cucumber_core': 6, 'peel_cucumber_post': 7, 'mix_ingredients_core': 8, 'place_lettuce_into_bowl_post': 9, 'peel_cucumber_prep': 10, 'add_dressing_core': 11, 'cut_cucumber_prep': 12, 'add_vinegar_prep': 13, 'place_lettuce_into_bowl_prep': 14, 'add_pepper_core': 15, 'cut_tomato_prep': 16, 'cut_cucumber_core': 17, 'cut_cheese_core': 18, 'mix_dressing_prep': 19, 'place_cheese_into_bowl_prep': 20, 'serve_salad_onto_plate_core': 21, 'add_vinegar_post': 22, 'add_dressing_prep': 23, 'place_cucumber_into_bowl_post': 24, 'place_cheese_into_bowl_core': 25, 'cut_lettuce_core': 26, 'place_lettuce_into_bowl_core': 27, 'add_salt_core': 28, 'add_vinegar_core': 29, 'place_cucumber_into_bowl_prep': 30, 'add_dressing_post': 31, 'cut_lettuce_prep': 32, 'place_cucumber_into_bowl_core': 33, 'add_pepper_prep': 34, 'place_cheese_into_bowl_post': 35, 'add_oil_core': 36, 'mix_ingredients_post': 37, 'add_oil_post': 38, 'add_salt_post': 39, 'cut_cucumber_post': 40, 'cut_cheese_post': 41, 'add_pepper_post': 42, 'serve_salad_onto_plate_post': 43, 'add_salt_prep': 44, 'add_oil_prep': 45, 'cut_lettuce_post': 46, 'place_tomato_into_bowl_prep': 47, 'mix_dressing_post': 48, 'mix_ingredients_prep': 49, 'cut_tomato_core': 50, 'place_tomato_into_bowl_post': 51}

