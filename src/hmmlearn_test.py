import numpy as np
from hmmlearn import hmm
from pprint import pprint
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer
from itertools import zip_longest
import pandas as pd

import nltk
from nltk import ngrams

import matplotlib.pyplot as plt
from matplotlib.pyplot import cm


# load the data
data = np.load('salad_data_synthetic.npy', allow_pickle=True)
data = np.reshape(data,[len(data),1])

# format data from npy
formatted_data = []
lengths = []
for i in range(len(data)):
    formatted_data.append(data[i][0])
    lengths.append(len(data[i][0]))

# train an HMM and get the hidden states
model = hmm.GaussianHMM(n_components=50, n_iter=100)  # TODO change number of N
model.fit(np.concatenate(formatted_data).reshape(-1, 1), lengths=lengths)
hidden_states = [model.predict(np.array(element).reshape(-1, 1)) for element in formatted_data]
print(len(hidden_states), 'hidden states length')
print('HIDDEN STATES: \n', hidden_states)


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

    return cluster_labels, cluster_means


# run clustering
labels, means = get_clusters_as_ngrams(10, hidden_states)
print('labels: \n', labels)
print('means: \n', means)

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

# plot just the first 5 sequences to minimize clutter
fig, ax = plt.subplots()
n = len(hidden_states)-535
color = iter(cm.rainbow(np.linspace(0, 1, n)))
for i in range(len(hidden_states)-535):
    seq = hidden_states[i].tolist()
    label=labels[i]
    c = next(color)
    ax.plot(seq, label="label: "+str(label), c=c)

ax.legend(loc='best')
ax.set_xlabel('step in sequence')
ax.set_ylabel('hidden state #')
fig.show()
plt.show()