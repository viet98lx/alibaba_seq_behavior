import re
import numpy as np
import MC_utils
from MC import MarkovChain
import argparse
import scipy.sparse as sp
import os
import json

def MC_hit_ratio(test_instances, topk, MC_model):
    hit_count = 0
    # user_correct = set()
    # user_dict = dict()
    for line in test_instances:
        elements = line.split("|")
        # user = elements[0]
        # if user not in user_dict:
        #     user_dict[user] = len(user_dict)
        basket_seq = elements[-MC_model.mc_order-1:-1]
        last_basket = basket_seq[-1]
        prev_item = []
        for prev_basket in basket_seq[:-1]:
            prev_item += re.split('[\\s]+', prev_basket.strip())
        list_predict_item = MC_model.top_predicted_item(prev_item, topk)
        item_list = re.split('[\\s]+', last_basket.strip())
        num_correct = len(set(item_list).intersection(list_predict_item))
        if num_correct > 0 :
            hit_count += 1
            # user_correct.add(user)
    return hit_count / len(test_instances)


def MC_recall(test_instances, topk, MC_model):
    list_recall = []
    # total_correct = 0
    # total_user_correct = 0
    for line in test_instances:
        elements = line.split("|")
        user = elements[0]
        basket_seq = elements[-MC_model.mc_order-1:-1]
        last_basket = basket_seq[-1]
        # prev_basket = basket_seq[-2]
        prev_item = []
        for prev_basket in basket_seq[:-1]:
            prev_item += re.split('[\\s]+', prev_basket.strip())
        list_predict_item = MC_model.top_predicted_item(prev_item, topk)
        item_list = re.split('[\\s]+', last_basket.strip())
        num_correct = len(set(item_list).intersection(list_predict_item))
        # total_correct += num_correct
        # if num_correct > 0:
        #   total_user_correct += 1
        list_recall.append(num_correct / len(item_list))
    return np.array(list_recall).mean()

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', help='The directory of input', type=str, default='../data/')
    parser.add_argument('--output_dir', help='The directory of output', type=str, default='../saved_models/')
    parser.add_argument('--model_name', help='Model name ', type=str, default='mc')
    parser.add_argument('--mc_order', help='Markov order', type=int, default=1)
    parser.add_argument('--w_behavior', help='Weight behavior file', type=str, default=None)
    args = parser.parse_args()

    data_dir = args.input_dir
    o_dir = args.output_dir
    model_name = args.model_name
    mc_order = args.mc_order
    w_behavior_file = args.w_behavior

    train_data_path = data_dir+'train_lines.txt'
    train_instances = MC_utils.read_instances_lines_from_file(train_data_path)
    nb_train = len(train_instances)
    print(nb_train)

    test_data_path = data_dir+'test_lines.txt'
    test_instances = MC_utils.read_instances_lines_from_file(test_data_path)
    nb_test = len(test_instances)
    print(nb_test)

    split_train = int(0.2*nb_train)
    split_test = int(0.2*nb_test)

    train_instances = train_instances[:split_train]
    test_instances = test_instances[:split_test]

    if w_behavior_file is None:
        w_behavior = {'buy': 1, 'cart': 0.5, 'fav': 0.5, 'pv':0.5}
    else:
        with open(w_behavior_file, 'r') as fp:
            w_behavior = json.load(fp)

    ### build knowledge ###
    # common_instances = train_instances + test_instances
    print("---------------------@Build knowledge-------------------------------")
    MAX_SEQ_LENGTH, item_dict, reversed_item_dict, item_probs, item_freq_dict, user_dict = MC_utils.build_knowledge(train_instances+test_instances, w_behavior)
    print('Build knowledge done')
    transition_matrix = MC_utils.calculate_transition_matrix(train_instances, item_dict, item_freq_dict, reversed_item_dict, w_behavior, mc_order)
    sp_matrix_path = model_name+'_transition_matrix_MC.npz'
    # nb_item = len(item_dict)
    # print('Density : %.6f' % (transition_matrix.nnz * 1.0 / nb_item / nb_item))
    if not os.path.exists(o_dir):
        os.makedirs(o_dir)
    saved_file = os.path.join(o_dir, sp_matrix_path)
    print("Save model in ", saved_file)
    sp.save_npz(saved_file, transition_matrix)

    mc_model = MarkovChain(item_dict, reversed_item_dict, item_freq_dict, w_behavior, transition_matrix, mc_order)
    for topk in [5, 10, 15]:
        print("Top : ", topk)
        hit_rate = MC_hit_ratio(test_instances, topk, mc_model)
        recall = MC_recall(test_instances, topk, mc_model)
        print("hit ratio: ", hit_rate)
        print("recall: ", recall)