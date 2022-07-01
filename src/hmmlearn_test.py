import numpy as np
from hmmlearn import hmm
from pprint import pprint
from sklearn.cluster import KMeans
from itertools import zip_longest
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')

data = np.load('salad_data_synthetic.npy', allow_pickle=True)
# pprint(data)
data = np.reshape(data,[len(data),1])

# pprint(len(data))
# print(data.shape)

# hidden_states = []
# for i in range(len(data)):
#     print(data[i][0])
#     model = hmm.GaussianHMM(n_components=2, n_iter=100).fit(data[i][0])

#     hidden_state = model.predict(data)
#     hidden_states.append(hidden_state)

formatted_data = []
for i in range(len(data)):
    formatted_data.append(data[i][0])
    
# formatted_data = np.array(formatted_data)
# pprint(formatted_data)
padded_formatted_data = np.array(list(zip_longest(*formatted_data, fillvalue=0))).T # since fit() only takes arrays of the same length
padded_formatted_data_reshaped = np.reshape(padded_formatted_data, (-1, 1))
print(padded_formatted_data_reshaped.shape, 'padded formatted data reshaped shape lmao')

model = hmm.GaussianHMM(n_components=20, n_iter=100).fit(padded_formatted_data_reshaped) # TODO change number of N
hidden_states = model.predict(padded_formatted_data_reshaped)
print(len(hidden_states), 'hidden states length')
pprint(hidden_states)

# run clustering
def get_clusters(hidden_seqs):
    features =  hidden_seqs.reshape(-1, 1)
    print(len(features), 'features length')
    kmeans = KMeans(n_clusters=20).fit(features)

    cluster_labels = kmeans.labels_
    cluster_means = kmeans.cluster_centers_

    return cluster_labels, cluster_means

labels, means = get_clusters(hidden_states)
print(labels)
print(means)

# df = pd.read_csv('new_data.csv')
# g = sns.PairGrid(df)
# g.map(sns.scatterplot)

## visualize the clusters
plt.scatter(padded_formatted_data_reshaped, padded_formatted_data_reshaped, label="50 salads synthetic data", c=labels, cmap='rainbow') # c=labels/242,
# plt.scatter(padded_formatted_data_reshaped[:,0], padded_formatted_data_reshaped[:,1], label="50 salads synthetic data", c=labels, cmap='rainbow') # c=labels/242,
# plt.scatter(padded_formatted_data[0,:], padded_formatted_data[1,:], c=labels)
plt.show()