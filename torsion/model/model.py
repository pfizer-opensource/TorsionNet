"""ANN to predict torsional strain from Atomic Environment vectors
"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam

def get_model(num_feat=293, lr=1e-3, drop_out=0.1, layer_dims='10-5-1-0.2'):
    model = Sequential()
    act_fn = 'relu'

    layer_dims = [float(d) for d in layer_dims.split('-')]

    model.add(
        Dense(
            int(num_feat * layer_dims[0]), input_dim=num_feat,
            kernel_initializer='normal', activation=act_fn))
    model.add(BatchNormalization())
    model.add(Dropout(drop_out))

    for layer_dim in layer_dims[1:-1]:
        model.add(Dense(int(num_feat * layer_dim), activation=act_fn))
        model.add(BatchNormalization())
        model.add(Dropout(drop_out))

    model.add(Dense(int(num_feat * layer_dims[-1]), activation=act_fn))
    model.add(Dropout(drop_out))
    model.add(Dense(1))

    adam = Adam(lr=lr)
    model.compile(loss='logcosh', optimizer=adam)

    return model