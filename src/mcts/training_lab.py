import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow warnings
from tensorflow import keras
from keras import layers


#============================================================#============================================================#
#============================================================#============================================================#
#============================================================#============================================================#

DATAFILE = 'othello_train_data_model_enhanced.npz'
BASE_MODEL = 'model_v3.keras'

NETWORK_ARCHITECTURE = [
    layers.Input(shape=(8,8,1)),

    layers.Conv2D(32, (3, 3), padding='same'),
    layers.BatchNormalization(),
    layers.ReLU(),
    
    layers.Conv2D(32, (3, 3), padding='same'),
    layers.BatchNormalization(),
    layers.ReLU(),

    layers.Conv2D(32, (3, 3), padding='same'),
    layers.BatchNormalization(),
    layers.ReLU(),

    layers.Flatten(),
    layers.Dense(64, kernel_regularizer=keras.regularizers.l2(0.001)),
    layers.BatchNormalization(),
    layers.ReLU(),


    layers.Dense(1, activation='tanh')
    ]

OPTIMIZER = keras.optimizers.Adam(learning_rate=0.001)

CALLBACKS = [keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(monitor='val_r2_score',factor=0.5,  patience=5,  min_lr=1e-5,verbose=1)]

PRINT_DATA_SAMPLE = True  

#============================================================#============================================================#
#============================================================#============================================================#
#============================================================#============================================================#


def load_data(filename=DATAFILE):
    data = np.load(f'data/{filename}' ,allow_pickle=True)
    X, y = data['X'], data['y']
    X = np.where(X == 3, 0, X)
    return X, y

def preprocess_data(X, y,verbose=False):
    scaler = MinMaxScaler(feature_range=(-1, 1))
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train = scaler.fit_transform(X_train.reshape(-1, 64)).reshape(-1, 8, 8)
    X_val = scaler.transform(X_val.reshape(-1, 64)).reshape(-1, 8, 8)

    print("Training data shape:", X_train.shape, y_train.shape)
    print("Validation data shape:", X_val.shape, y_val.shape)
    if verbose:
        randi = np.random.randint(0, len(X_train))
        print("Sample training data:", X_train[randi])
        print("Sample training label:", y_train[randi])
    return X_train, X_val, y_train, y_val

def build_model(architecture=NETWORK_ARCHITECTURE, optimizer=OPTIMIZER):
    model = keras.Sequential(architecture)
    model.compile(optimizer=optimizer, loss='mse', metrics=['r2_score'])
    return model

def train_model(model, X_train, y_train, callbacks=CALLBACKS, epochs=100):
    history = model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=epochs,
        batch_size=32,
        callbacks=callbacks,
    )
    return history

def evaluate_model(model, X_val, y_val):
    results = model.evaluate(X_val, y_val, verbose=0)
    print('====================== Evaluation Results ======================')
    print(f"\nValidation Loss: {results[0]}, Validation R2 Score: {results[1]}\n")
    print('================================================================')
    return results

def save_model(model, filename):
    model.save(f'../../models/{filename}')
    print(f"Model saved to {filename}")

def main(verbose= False):
    X, y = load_data()
    X_train, X_val, y_train, y_val = preprocess_data(X, y, verbose=verbose)

    model = build_model(NETWORK_ARCHITECTURE, OPTIMIZER) if not BASE_MODEL else keras.models.load_model(f'models/{BASE_MODEL}')
    history = train_model(model, X_train, y_train, CALLBACKS)

    evaluate_model(model, X_val, y_val)
    inpt = ""
    while inpt.lower() not in ['y', 'n']:
        inpt = input("\nDo you want to save the model? (y/n): ")
        if inpt.lower() == 'y':
            fn = input(f"\nEnter the filename to save the model: ")
            if fn == '':
                fn = "sample_model.keras"
            if not fn.endswith('.keras'):
                fn += '.keras'
            save_model(model, fn)
        elif inpt.lower() == 'n':
            print("Model not saved.")

if __name__ == "__main__":
    main(verbose=PRINT_DATA_SAMPLE)