import csv
import os
import datetime
import pandas as pd

TEST = True

annotation_dir = '../../../data/ann-ts/activityAnnotations'
csv_header = ['unique_id', 'person_id', 'video_id', 'start_t', 'end_t', 'action']

def get_timestamp(t):
    timestamp = float(t)/1000000 # convert from us to s
    hours,remainder = divmod(timestamp, 3600)
    minutes,seconds = divmod(remainder, 60)
    str_time = str(int(hours)) + ':' + str(int(minutes)) + ':' + str(int(seconds))
    return str_time


with open('new_data.csv', "w") as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        writer.writerow(csv_header)
        for filename in os.listdir(annotation_dir):
            f = os.path.join(annotation_dir, filename)
            if os.path.isfile(f):
                with open(f, 'r', encoding='utf-8') as ann_file:
                    if f.endswith('.txt'):
                        data = ann_file.readlines()

                        data_sorted = sorted(data, key=lambda line: (int(line.split()[0]), int(line.split()[1])))

                        # Write to CSV
                        for line in data_sorted:
                            elements = line.split(' ')
                            person_id = f.split('/')[-1].split('-')[0]
                            video_id = f.split('/')[-1].split('-')[1]
                            # start_t = datetime.timedelta(int(elements[0])/1000000.0) # microseconds to hour/min/sec
                            start_t = get_timestamp(int(elements[0]))
                            # end_t = datetime.timedelta(int(elements[1])/1000000.0)
                            end_t = get_timestamp(int(elements[1]))
                            unique_id = float(person_id + '.' + video_id)
                            action = elements[2]

                            writer.writerow([unique_id, person_id, video_id, start_t, end_t, action])


# Sort the csv rows in the order in which they appear in the directory
# df = pd.read_csv('new_data.csv')
# sorted_df = df.sort_values(by=['unique_id'], ascending=True)
# sorted_df.to_csv('new_data_sorted_.csv', index=False)

df = pd.read_csv('new_data.csv')

# Determine the number of unique actions in the input dataset, df, as a list
action_list = list(df['action'].unique())
# if TEST: print("action_list", action_list)
# if TEST: print("number of unique actions: ", len(action_list))
num_unique = len(action_list)

# Create a dictionary to create a unique int to represent each action
count = 0
action_dict = {}
for a in action_list:
    action_dict[a] = count
    count += 1
# if TEST: print("action_dict: \n", action_dict)

transitions = {}
# lettuce, tomato, cheese, salt, pepper, vinegar, oil, dressing, cucumber
# where each int can take a value of {0,1,2,3} for {none,prep,core,post}
state_vec = [0,0,0,0,0,0,0,0,0]
steps = {'none' : 0, 'prep' : 1, 'core': 2, 'post' : 3}
state_idx = None

with open('new_data.csv', 'r') as f:
    reader = csv.reader(f)
    prev_annotation = None
    for i, line in enumerate(reader):
        if i == 0: continue

        annotation = line[5].strip('\n')
        annotation_step = annotation.split('_')[-1] # get last element {prep, core, post}
        print("annotation: \n", annotation)
        # print(annotation_step)

        cur_annotation = annotation

        action_segments = annotation.split('_')
        # print(action_segments)

        if 'lettuce' in action_segments:
            state_idx = 0
        elif 'tomato' in action_segments:
            state_idx = 1
        elif 'cheese' in action_segments:
            state_idx = 2
        elif 'salt' in action_segments:
            state_idx = 3
        elif 'pepper' in action_segments:
            state_idx = 4
        elif 'vinegar' in action_segments:
            state_idx = 5
        elif 'oil' in action_segments:
            state_idx = 6
        elif 'dressing' or 'ingredients' in action_segments: # pretend 'cut_and_mix_ingredients' is dressing for now
            state_idx = 7
        elif 'cucumber' in action_segments:
            state_idx = 8
        else:
            print(action_segments)
            raise ValueError('Incorrect value found in action sequence.')

        # TODO: initialize as None / non existent
        if annotation_step == 'prep':
            state_vec[state_idx] = 1
        elif annotation_step == 'core':
            state_vec[state_idx] = 2
        elif annotation_step == 'post':
            state_vec[state_idx] = 3
        else:
            continue
            #TODO: throw exception
        # TODO: reset state_vec

        if prev_annotation not None:
            if transitions[state_vec] is None:
                transitions.setdefault(state_vec, [])
            transitions[state_vec].append(())

        prev_annotation = annotation