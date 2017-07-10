import numpy as np
import numpy.random as nr
import matplotlib.pyplot as plt

from sklearn.datasets import load_digits
from NumBandit import NumBandit

# Global Variables
e = 0.1 # % of time to explore
batch_size = 100 # Obvious what this is


def load_shuffled_digits():
    """
    Function to load sklearn handwritten digit dataset and return the shuffled digits and corresponding targets
    In the original dataset they are ordered 1-9 repeating
    :return:
    """
    digits, target = load_digits()['data'], load_digits()['target'] # Assigned to variables, to prevent having to call
                                                                    # load_digits repeatedly.
    data = np.array([[digits[n], target[n]] for n in range(len(digits))])
    nr.shuffle(data)
    return data.T[0], data.T[1]


def make_all_bandits(p_digits, input_shape):
    """
    This function creates a dictionary with the key as the digit to be predicted and the value as the bandit that
    predicts that digits
    :param p_digits: the set containing possible digits that can be predicted
    :return: bandit_dic: the dictionary containing the bandits
    """
    bandit_dic = {}
    for digit in p_digits:
        bandit_dic[digit] = NumBandit(input_shape, digit)
    return bandit_dic


def predict_rewards(bandit_dic, image):
    """
    Predicts rewards from each action
    :param bandit_dic: Dictionary containing all bandits
    :param image: the image containing the digit we would like to work out
    :return: a dict, with the bandit target var and the predicted reward
    """
    predicted_rewards = {}
    for key in bandit_dic:
        predicted_rewards[bandit_dic[key].target_var] = bandit_dic[key].make_prediction(image)
    return predicted_rewards


def e_greedy(pred_rews):
    """
    Takes in rewards, and chooses an action e-greedily, where e is a global var.
    :param pred_rews:
    :return: chosen action
    """
    if nr.random() > e:
        return max(pred_rews, key=pred_rews.get)
    else:
        return nr.choice(pred_rews.keys())


def update_weights(train_batch, all_bandits):
    for number in possible_digits:
        if len(train_batch[number]['ims']) == 0:
            continue
        all_bandits[number].net_train(np.array(train_batch[number]['ims']), np.array(train_batch[number]['rews']))
    print target[0], predict_rewards(all_bandits, np.atleast_2d(digits[0]))


# Data Dependent Global Variables
digits, target = load_shuffled_digits()
input_shape = digits[0].shape[0]
possible_digits = list(set(target))
actions = len(possible_digits)

# Make the bandits
all_bandits = make_all_bandits(possible_digits, input_shape)

# Make the location where training data will be filed.
train_batch = dict()
for digit in possible_digits:
    train_batch[digit] = {'ims': [], 'rews': []}


regret = [] # Where we track our loss

for index, image in enumerate(digits):

    im = np.atleast_2d(image) # This is how keras wants the input
    pred_rewards = predict_rewards(all_bandits, im) # Sees which bandit will return the best result
    action = e_greedy(pred_rewards) # Chooses an action e-greedily on the predicted results

    train_batch[action]['ims'].append(image)
    train_batch[action]['rews'].append(float(action == target[index]))
    regret.append(1.0 - float(action == target[index]))

    if (index % batch_size == 0) & (index != 0):
        update_weights(train_batch, all_bandits)


xax = range(len(digits))
yax = np.cumsum(regret)
plt.plot(xax, yax)
plt.show()








