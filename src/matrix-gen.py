import pandas as pd
transitions = ['A', 'B', 'B', 'C', 'B', 'A', 'D', 'D', 'A', 'B', 'A', 'D']

df = pd.DataFrame(transitions)

# create a new column with data shifted one space
df['shift'] = df[0].shift(-1)

# add a count column (for group by function)
df['count'] = 1

# groupby and then unstack, fill the zeros
trans_mat = df.groupby([0, 'shift']).count().unstack().fillna(0)

# normalise by occurences and save values to get transition matrix
trans_mat = trans_mat.div(trans_mat.sum(axis=1), axis=0).values

print(trans_mat)