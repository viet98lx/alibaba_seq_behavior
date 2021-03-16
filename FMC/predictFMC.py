import sys, os, pickle, argparse
import re
import random
import FMC_utils
from FMC import FMC

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', help='The directory of input', type=str, default='../data/')
    parser.add_argument('--output_dir', help='The directory of output', type=str, default='../saved_models/')
    parser.add_argument('--model_name', help='Model name ', type=str, default='fpmc')
    parser.add_argument('--load_file', help='Load file name ', type=str, default='W_H')
    parser.add_argument('--nb_predict', help='# of predict', type=int, default=10)
    parser.add_argument('--topk', help='# of predict', type=int, default=10)
    parser.add_argument('--n_factor', help='# of factor', type=int, default=4)
    parser.add_argument('--mc_order', help='Markov order', type=int, default=1)
    # parser.add_argument('--example_file', help='Example_file', type=str, default=None)
    args = parser.parse_args()

    f_dir = args.input_dir
    o_dir = args.output_dir
    model_name = args.model_name
    nb_predict = args.nb_predict
    load_file = args.load_file
    topk = args.topk
    # ex_file = args.example_file
    n_factor = args.n_factor
    mc_order = args.mc_order

    data_dir = f_dir
    train_data_path = data_dir + 'train_lines.txt'
    train_instances = FMC_utils.read_instances_lines_from_file(train_data_path)
    nb_train = len(train_instances)
    # print(nb_train)

    test_data_path = data_dir + 'test_lines.txt'
    test_instances = FMC_utils.read_instances_lines_from_file(test_data_path)
    nb_test = len(test_instances)
    # print(nb_test)
    print("---------------------@Build knowledge-------------------------------")
    MAX_SEQ_LENGTH, item_dict, reversed_item_dict, item_probs, item_freq_dict, user_dict = FMC_utils.build_knowledge(train_instances+test_instances)

    if not os.path.exists(o_dir):
        os.makedirs(o_dir)
    # saved_file = os.path.join(o_dir, 'transition_matrix_MC.npz')
    # print("Save model in ", saved_file)
    # H = np.load_npz(saved_file)
    fmc_model = FMC(item_dict, reversed_item_dict, item_freq_dict, n_factor, mc_order)
    fmc_model.load(load_file)

    # if ex_file is not None:
    #     ex_instances = FMC_utils.read_instances_lines_from_file(ex_file)
    # else :
    #     ex_instances = test_instances
    topk = 50
    print('Predict to outfile')
    predict_file = os.path.join(o_dir, 'predict_' + model_name + '.txt')
    FMC_utils.write_predict(predict_file, test_instances, topk, fmc_model)
    print('Predict done')
    ground_truth, predict = FMC_utils.read_predict(predict_file)
    for topk in [5, 10, 15, 20]:
        print("Top : ", topk)
        # hit_rate = MC_hit_ratio(test_instances, topk, mc_model)
        # recall = MC_recall(test_instances, topk, mc_model)
        hit_rate = FMC_utils.hit_ratio(ground_truth, predict, topk)
        recall = FMC_utils.recall(ground_truth, predict, topk)
        print("hit ratio: ", hit_rate)
        print("recall: ", recall)

