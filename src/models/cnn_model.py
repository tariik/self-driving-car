from tensorflow.keras import layers, models

def build_cnn_model(input_shape, num_actions):
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    # ...existing code...
    model.add(layers.Flatten())
    model.add(layers.Dense(num_actions, activation='linear'))
    return model
