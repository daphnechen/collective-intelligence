import json
# from pprint import pprint
import numpy as np
import pprint

# load the data from previously-created files
hs_file = open('hs_map.json')
hs_map = json.load(hs_file)

ss_file = open('support_sets.json')
support_sets = json.load(ss_file)

data = np.load('salad_data_synthetic_new.npy', allow_pickle=True)
data = np.reshape(data,[len(data),1])
# pprint.pprint(data)

# format data from npy
formatted_data = []
for i in range(len(data)):
    formatted_data.append(data[i][0])

action_dict_lookup = {0: 'add_dressing_core',
                        1: 'add_dressing_post',
                        2: 'add_dressing_prep',
                        3: 'add_oil_core',
                        4: 'add_oil_post',
                        5: 'add_oil_prep',
                        6: 'add_pepper_core',
                        7: 'add_pepper_post',
                        8: 'add_pepper_prep',
                        9: 'add_salt_core',
                        10: 'add_salt_post',
                        11: 'add_salt_prep',
                        12: 'add_vinegar_core',
                        13: 'add_vinegar_post',
                        14: 'add_vinegar_prep',
                        15: 'cut_cheese_core',
                        16: 'cut_cheese_post',
                        17: 'cut_cheese_prep',
                        18: 'cut_cucumber_core',
                        19: 'cut_cucumber_post',
                        20: 'cut_cucumber_prep',
                        21: 'cut_lettuce_core',
                        22: 'cut_lettuce_post',
                        23: 'cut_lettuce_prep',
                        24: 'cut_tomato_core',
                        25: 'cut_tomato_post',
                        26: 'cut_tomato_prep',
                        27: 'end',
                        28: 'mix_dressing_core',
                        29: 'mix_dressing_post',
                        30: 'mix_dressing_prep',
                        31: 'mix_ingredients_core',
                        32: 'mix_ingredients_post',
                        33: 'mix_ingredients_prep',
                        34: 'peel_cucumber_core',
                        35: 'peel_cucumber_post',
                        36: 'peel_cucumber_prep',
                        37: 'place_cheese_into_bowl_core',
                        38: 'place_cheese_into_bowl_post',
                        39: 'place_cheese_into_bowl_prep',
                        40: 'place_cucumber_into_bowl_core',
                        41: 'place_cucumber_into_bowl_post',
                        42: 'place_cucumber_into_bowl_prep',
                        43: 'place_lettuce_into_bowl_core',
                        44: 'place_lettuce_into_bowl_post',
                        45: 'place_lettuce_into_bowl_prep',
                        46: 'place_tomato_into_bowl_core',
                        47: 'place_tomato_into_bowl_post',
                        48: 'place_tomato_into_bowl_prep',
                        49: 'serve_salad_onto_plate_core',
                        50: 'serve_salad_onto_plate_post',
                        51: 'serve_salad_onto_plate_prep'}

pp = pprint.PrettyPrinter(indent=4)
# pp.pprint(action_dict_lookup)

support_sets_annotations = {}
for i, (label, values) in enumerate(hs_map.items()):
    for element in values:
        # print('values: \n', values)
        idx, arr = element
        
        if label in support_sets_annotations.keys():
            converted_to_str = []
            for int_annotation in formatted_data[idx]:
                # print('this is int_annotation: ', int_annotation)
                str_annotation = action_dict_lookup[int_annotation]
                # print('this is str_annotation: ', str_annotation)
                converted_to_str.append(str_annotation)
            # print('this is converted_to_str: ', converted_to_str)
            support_sets_annotations[label].append(converted_to_str)
        else:
            converted_to_str = []
            for int_annotation in formatted_data[idx]:
                str_annotation = action_dict_lookup[int_annotation]
                converted_to_str.append(str_annotation)
            support_sets_annotations[label] = [converted_to_str]
            # print(support_sets_annotations[label])

with open('support_sets_annotations_test.json', 'w') as f:
    json.dump(support_sets_annotations, f, ensure_ascii=False, indent=4) # sort_keys=True,

# print('THESE ARE THE SUPPORT SETS W ANNOTATIONS \n', support_sets_annotations)

# --------------------------------------------------------------

# action_dict = {'serve_salad_onto_plate_prep': 0, 'cut_cheese_prep': 1, 'cut_tomato_post': 2, 'place_tomato_into_bowl_core': 3, 'mix_dressing_core': 4, 'end': 5, 'peel_cucumber_core': 6, 'peel_cucumber_post': 7, 'mix_ingredients_core': 8, 'place_lettuce_into_bowl_post': 9, 'peel_cucumber_prep': 10, 'add_dressing_core': 11, 'cut_cucumber_prep': 12, 'add_vinegar_prep': 13, 'place_lettuce_into_bowl_prep': 14, 'add_pepper_core': 15, 'cut_tomato_prep': 16, 'cut_cucumber_core': 17, 'cut_cheese_core': 18, 'mix_dressing_prep': 19, 'place_cheese_into_bowl_prep': 20, 'serve_salad_onto_plate_core': 21, 'add_vinegar_post': 22, 'add_dressing_prep': 23, 'place_cucumber_into_bowl_post': 24, 'place_cheese_into_bowl_core': 25, 'cut_lettuce_core': 26, 'place_lettuce_into_bowl_core': 27, 'add_salt_core': 28, 'add_vinegar_core': 29, 'place_cucumber_into_bowl_prep': 30, 'add_dressing_post': 31, 'cut_lettuce_prep': 32, 'place_cucumber_into_bowl_core': 33, 'add_pepper_prep': 34, 'place_cheese_into_bowl_post': 35, 'add_oil_core': 36, 'mix_ingredients_post': 37, 'add_oil_post': 38, 'add_salt_post': 39, 'cut_cucumber_post': 40, 'cut_cheese_post': 41, 'add_pepper_post': 42, 'serve_salad_onto_plate_post': 43, 'add_salt_prep': 44, 'add_oil_prep': 45, 'cut_lettuce_post': 46, 'place_tomato_into_bowl_prep': 47, 'mix_dressing_post': 48, 'mix_ingredients_prep': 49, 'cut_tomato_core': 50, 'place_tomato_into_bowl_post': 51}

result = []
for arr in formatted_data:
    cur_seq = []
    for int_v in arr:
        str_ann = action_dict_lookup[int_v]
        cur_seq.append(str_ann)
    result.append(cur_seq)
pprint.pprint(result)
