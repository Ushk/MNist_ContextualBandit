# MNist_ContextualBandit

## Introduction
Contextual Bandits are an improvement on the standard Multi-Armed Bandit Technique. They take into account the "context" - the state of the environment 
when choosing an action. There are a variety of ways that this relationship between environmental state and choosing an action can take place, most of which posit
some reward that is dependent on the state, and then choose according to some decision criteria on this reward. 

I wanted to improve my understanding of the Contextual Bandit algorithm, by implementing one. I also wanted to improve my knowledge of Neural Networks, so I used 
a Contextual Bandit algorithm where each "arm" of the Contextual Bandit corresponds to an NN that predicts the probability of a certain outcome. Arms were chosen 
according to an epsilon-greedy strategy: choose the current best arm with probability 1-epsilon, and choose an arm at random with probability epsilon/(num arms).

The Contextual Bandit is essentially a reinforcement learning algorithm, and there is no such thing as a reinforcement learning dataset, for obvious reasons. The 
Contextual Bandit is also not equivalent to the full reinforcement learning problem, since there is only one action and this action does not affect the environmental
state. In order to create something to work on, I took the MNist Dataset, and got my Contextual Bandit to learn it in an online way. The pseudocode is below:


### Pseudocode 
- Load Data
- Initialise e = 0.1
- Initialise Batch Size = 100
- Make a Neural Network for each handwritten digit.
- Create Dictionary to store training data
- For digit in mnist digit dataset:
    - Make sure it is appropriate format
    - See what reward each bandit predicts for that digit
- Choose an action (i.e a number) e-greedily
- Record the digit
- Record the reward, using Reward(1 if action==correct number, else 0)
- Append reward
- If loop index % batch size = 0:
    - Update weights
- Plot regret

## Results

Well, the weights of the correct number go to ~.95-.99, and the others go to 0. This is obviously a good sign. However, the regret (cumulative difference between what you predicted and what you should have predicted) in a Bandit by definition grows
as O(log(N)) where N is the number of predictions (i.e. predicted digits in this case). The regret in this case does not appear to grow linearly, although I didn't put a vast amount of effort into thinking about this (because am on holiday in Portugal
and doing this while I wait for the sunburn to stop hurting!) 

## TODOs

1. Try this on a different dataset, like the keras digit dataset, which is 60000, as opposed to ~1700
2. Look into this regret not growing logarithmically. Could it be that the sklearn version of the mnist dataset is simply too small? 
3. Investigate a more intelligent strategy than e-greedy. To me e-greedy is the Dan Brown (the author) of Bandit Algorithms. It works, but you know from the beginning there's better out there.
