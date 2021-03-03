import scipy.sparse as sp
import numpy as np

class MarkovChain():
  def __init__(self, item_dict, reversed_item_dict, item_freq_dict, weight_behaivor, transition_matrix, mc_order):
    self.item_freq_dict = item_freq_dict
    self.item_dict = item_dict
    self.reversed_item_dict = reversed_item_dict
    self.nb_items = len(item_dict)
    # self.sp_matrix_path = sp_matrix_path
    self.mc_order = mc_order
    self.w_behavior = weight_behaivor
    self.transition_matrix = transition_matrix

  def top_predicted_item(self, previous_baskets, topk):
    candidate = np.zeros(self.nb_items)
    # last_basket = previous_baskets[-1]
    transition_score = np.zeros(len(previous_baskets[0]))
    for i in range(1, len(previous_baskets)):
      prev_score = transition_score
      prev_basket_idx = [self.item_dict[item] for item in previous_baskets[i-1]]
      cur_basket_idx = [self.item_dict[item] for item in previous_baskets[i]]
      transition_score = np.array((prev_score + np.log(self.transition_matrix[prev_basket_idx, cur_basket_idx].todense())).sum(axis=0))[0]
      transition_score = transition_score / len(previous_baskets[i-1])

    last_basket_idx = [self.item_dict[item] for item in previous_baskets[-1]]
    # for item_idx in prev_basket_idx:
    # for i in range(len(self.behavior_dict)):
    candidate = (transition_score + np.array(self.transition_matrix[last_basket_idx, :].todense()).sum(axis=0))[0]
    candidate = candidate / len(last_basket_idx)
    topk_idx = np.argpartition(candidate, -topk)[-topk:]
    sorted_topk_idx = topk_idx[np.argsort(candidate[topk_idx])]
    topk_item = [self.reversed_item_dict[item] for item in sorted_topk_idx]
    # print("Done")
    return topk_item