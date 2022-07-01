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
import seaborn as sns
sns.set_style('darkgrid')

data = np.load('salad_data_synthetic.npy', allow_pickle=True)
data = np.reshape(data,[len(data),1])

# hidden_states = []
# for i in range(len(data)):
#     print(data[i][0])
#     model = hmm.GaussianHMM(n_components=2, n_iter=100).fit(data[i][0])

#     hidden_state = model.predict(data)
#     hidden_states.append(hidden_state)

formatted_data = []
lengths = []
for i in range(len(data)):
    formatted_data.append(data[i][0])
    lengths.append(len(data[i][0]))
# print('length of lengths arr: \n', len(lengths))
    
# formatted_data = np.array(formatted_data)
# pprint(formatted_data)
padded_formatted_data = np.array(list(zip_longest(*formatted_data, fillvalue=0))).T # since hmm.fit() only takes arrays of the same length
padded_formatted_data_reshaped = np.reshape(padded_formatted_data, (-1, 1)) # fit data to required dims for hmm.fit()
# pprint(padded_formatted_data)
# print('padded_formatted_data shape: \n', padded_formatted_data.shape)

# param = set(padded_formatted_data.ravel())
model = hmm.GaussianHMM(n_components=10, n_iter=100).fit(padded_formatted_data_reshaped, lengths=lengths) # TODO change number of N
# hidden_states = model.predict(padded_formatted_data_reshaped, lengths=lengths)
h_s = [model.predict(np.array(element).reshape(-1, 1)) for element in formatted_data]
print(len(h_s), 'hidden states length')
# print(h_s.shape, 'hidden states shape')
print('HIDDEN STATES: \n', h_s)


# run clustering
def get_clusters(hidden_seqs):
    features = []
    if type(hidden_seqs) is list:
        features = np.reshape(hidden_seqs, (-1, 1))
    else:
        features =  hidden_seqs.reshape(-1, 1)
    print(len(features), 'features length')
    kmeans = KMeans(n_clusters=10).fit(features)
    cluster_labels = kmeans.labels_
    cluster_means = kmeans.cluster_centers_

    return cluster_labels, cluster_means


def get_clusters_as_ngrams(hidden_seqs):
    '''
        Account for sequences of different length
    '''
    vocabulary = list()
    for element in hidden_seqs:
        element = element.tolist()
        vocabulary.append(str(','.join(str(e) for e in ['x' + str(e) for e in element])))
    
    vectorizer = CountVectorizer()
    matrix = vectorizer.fit_transform(vocabulary)
    matrix.toarray()

    kmeans = KMeans(n_clusters=10, random_state=0).fit(matrix)
    cluster_labels = kmeans.labels_
    cluster_means = kmeans.cluster_centers_

    return cluster_labels, cluster_means


labels, means = get_clusters_as_ngrams(h_s)
print('labels: \n', labels)
print('means: \n', means)

# df = pd.read_csv('new_data.csv')
# g = sns.PairGrid(df)
# g.map(sns.scatterplot)

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

fig, ax = plt.subplots() # len(h_s), 10
n = len(h_s)-535
color = iter(cm.rainbow(np.linspace(0, 1, n)))
for i in range(len(h_s)-535):
    seq = h_s[i].tolist()
    label=labels[i]
    # print('seq: \n', seq)
    # print('label: \n', label)
    c = next(color)
    ax.plot(seq, label="label: "+str(label), c=c) # label="50 salads synthetic data", c=label

ax.legend(loc='best')
ax.set_xlabel('step in sequence')
ax.set_ylabel('hidden state #')
fig.show()
plt.show()