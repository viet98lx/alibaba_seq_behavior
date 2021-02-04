import itertools
import scipy.sparse as sp
import re
import numpy as np
import time
def calculate_transition_matrix(train_instances, item_dict, item_freq_dict, reversed_item_dict, w_behavior, mc_order):
    pair_dict = dict()
    NB_ITEMS = len(item_dict)
    print("number items: ", NB_ITEMS)
    # j = 0
    start = time.time()
    for line in train_instances:
        # print(j)
        # j += 1
        elements = line.split("|")
        user = elements[0]
        # print('User')
        basket_seq = elements[1:]
        for i in range(mc_order,len(basket_seq)):
            prev_baskets = basket_seq[i-mc_order:i]
            cur_basket = basket_seq[i]
            # prev_item_list = re.split('[\\s]+', prev_basket.strip())
            prev_item_list = []
            for basket in prev_baskets:
                prev_item_list += [(p.split(':')) for p in re.split('[\\s]+', basket.strip())]
            prev_ib_idx = [(item_dict[ib[0]], ib[1]) for ib in prev_item_list]
            cur_item_list = [p.split(':')[0] for p in re.split('[\\s]+', cur_basket.strip())]
            cur_item_idx = [item_dict[item] for item in cur_item_list]
            # for ib_pair in prev_item_list:
            #     for item in cur_item_list:
            #         t = (item_dict[ib_pair[0]], item_dict[item])
            #         if len(t) == 0:
            #             print("empty")
            #         else:
            #             print(t)
            #         if t not in pair_dict:
            #             pair_dict[t] = w_behavior[ib_pair[1]]
            #         else:
            #             pair_dict[t] += w_behavior[ib_pair[1]]
            for t in list(itertools.product(prev_ib_idx, cur_item_idx)):
                item_pair = (t[0][0], t[1])
                if item_pair in pair_dict.keys():
                    pair_dict[item_pair] += w_behavior[t[0][1]]
                else:
                    pair_dict[item_pair] = w_behavior[t[0][1]]
    end = time.time()
    print("Time to run all seq line: ", end-start)

    start_1 = time.time()
    for key in pair_dict.keys():
        pair_dict[key] /= item_freq_dict[reversed_item_dict[key[0]]]

        row = [p[0] for p in pair_dict]
        col = [p[1] for p in pair_dict]
        data = [pair_dict[p] for p in pair_dict]
        transition_matrix = sp.csr_matrix((data, (row, col)), shape=(NB_ITEMS, NB_ITEMS), dtype="float32")
        end_1 = time.time()
        print("Time to Create transition matrix: ", end_1-start_1)
        nb_nonzero = len(pair_dict)
        density = nb_nonzero * 1.0 / NB_ITEMS / NB_ITEMS
        print("Density of matrix: {:.6f}".format(density))
    return transition_matrix

def build_knowledge(training_instances, w_behavior):
    MAX_SEQ_LENGTH = 0
    item_freq_dict = {}
    user_dict = dict()

    for line in training_instances:
        elements = line.split("|")

        if len(elements) - 1 > MAX_SEQ_LENGTH:
            MAX_SEQ_LENGTH = len(elements) - 1

        user = elements[0]
        user_dict[user] = len(user_dict)

        basket_seq = elements[1:]

        for basket in basket_seq:
            ib_pair = [tuple(p.split(':')) for p in re.split('[\\s]+', basket.strip())]
            # print(ib_pair)
            for item_obs in ib_pair:
                if item_obs[0] not in item_freq_dict:
                    # print(item_obs[0])
                    # print(item_obs[1])
                    item_freq_dict[item_obs[0]] = w_behavior[item_obs[1]]
                else:
                    item_freq_dict[item_obs[0]] += w_behavior[item_obs[1]]

    items = sorted(list(item_freq_dict.keys()))
    item_dict = dict()
    item_probs = []
    for item in items:
        item_dict[item] = len(item_dict)
        item_probs.append(item_freq_dict[item])

    item_probs = np.asarray(item_probs, dtype=np.float32)
    item_probs /= np.sum(item_probs)

    reversed_item_dict = dict(zip(item_dict.values(), item_dict.keys()))
    return MAX_SEQ_LENGTH, item_dict, reversed_item_dict, item_probs, item_freq_dict, user_dict

def read_instances_lines_from_file(file_path):
    with open(file_path, "r") as f:
        lines = [line.rstrip('\n') for line in f]
        return lines

def FMC_hit_ratio(test_instances, topk, FMC_model):
    hit_count = 0
    # user_correct = set()
    # user_dict = dict()
    for line in test_instances:
        elements = line.split("|")
        # user = elements[0]
        # if user not in user_dict:
        #     user_dict[user] = len(user_dict)
        basket_seq = elements[-FMC_model.mc_order - 1:-1]
        last_basket = basket_seq[-1]
        prev_item_list = []
        for basket in basket_seq:
            prev_item_list += [p.split(':')[0] for p in re.split('[\\s]+', basket.strip())]
        list_predict_item = FMC_model.top_predicted_item(prev_item_list, topk)
        # item_list = re.split('[\\s]+', last_basket.strip())
        cur_item_list = [p.split(':')[0] for p in re.split('[\\s]+', last_basket.strip())]
        num_correct = len(set(cur_item_list).intersection(list_predict_item))
        if num_correct > 0 :
            hit_count += 1
            # user_correct.add(user)
    return hit_count / len(test_instances)


def FMC_recall(test_instances, topk, FMC_model):
    list_recall = []
    # total_correct = 0
    # total_user_correct = 0
    for line in test_instances:
        elements = line.split("|")
        user = elements[0]
        basket_seq = elements[-FMC_model.mc_order - 1:-1]
        last_basket = basket_seq[-1]
        prev_item_list = []
        for basket in basket_seq:
            prev_item_list += [p.split(':')[0] for p in re.split('[\\s]+', basket.strip())]
        list_predict_item = FMC_model.top_predicted_item(prev_item_list, topk)
        cur_item_list = [p.split(':')[0] for p in re.split('[\\s]+', last_basket.strip())]
        num_correct = len(set(cur_item_list).intersection(list_predict_item))
        # total_correct += num_correct
        # if num_correct > 0:
        #   total_user_correct += 1
        list_recall.append(num_correct / len(cur_item_list))
    return np.array(list_recall).mean()