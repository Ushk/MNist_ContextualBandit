import numpy as np
from keras.models import Sequential
from keras.layers import Dense

class NumBandit:

    def __init__(self, size, tar):
        self.input_size = size
        self.target_var = tar
        self.net = self.make_net()

    def make_net(self):
        Bandit = Sequential()
        Bandit.add(Dense(128, input_dim=64, activation='sigmoid'))
        Bandit.add(Dense(1, activation='sigmoid'))
        Bandit.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return Bandit

    def net_train(self, input_batch, target_batch):
        self.net.fit(input_batch, target_batch, batch_size=len(input_batch), nb_epoch=15, verbose=False)
        #self.net.train_on_batch(input_batch, target_batch)

    def make_prediction(self, input):
        return self.net.predict(input)[0][0]
