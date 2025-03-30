from tensorflow.keras import layers, models

def build_dqn_model(state_size, num_actions):
    model = models.Sequential()
    model.add(layers.Dense(64, activation='relu', input_dim=state_size))
    # ...existing code...
    model.add(layers.Dense(num_actions, activation='linear'))
    return model
