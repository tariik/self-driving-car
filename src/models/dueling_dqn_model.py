from tensorflow.keras import layers, models, Input

def build_dueling_dqn_model(state_size, num_actions):
    inputs = Input(shape=(state_size,))
    dense = layers.Dense(64, activation='relu')(inputs)
    # División en dos flujos: valor y ventaja
    value = layers.Dense(1)(dense)
    advantage = layers.Dense(num_actions)(dense)
    
    # Combinar valor y ventaja
    output = layers.Add()([value, advantage])
    model = models.Model(inputs=inputs, outputs=output)
    return model
