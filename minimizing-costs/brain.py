from keras.layers import Input, Dense
from keras.models import Model
from keras.optimizers import Adam

class Brain(object):
    def __init__(
            self,
            lr=0.001,
            num_actions=5,
        ):
        self.lr = lr

        states = Input(shape = (3,))
        x1 = Dense(units=64, activation='sigmoid')(states)
        x2 = Dense(units=32, activation='sigmoid')(x1)
        q_values = Dense(units=num_actions, activation='softmax')(x2)
        self.model = Model(inputs=states, outputs=q_values)

        self.model.compile(loss='mse', optimizer=Adam(lr))
