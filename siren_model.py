# Neural Networking
import tensorflow as tf
import numpy as np

class Sine(tf.keras.layers.Layer):
    def __init__(self, w0=1.0):
        super().__init__()
        self.w0 = w0

    def call(self, inputs):
        return tf.sin(self.w0 * inputs)

# Custom Weight Initializer
class SIRENInitializer(tf.keras.initializers.Initializer):
    def __init__(self, w0=1.0):
        self.w0 = w0

    def __call__(self, shape, dtype=None):
        input_dim = shape[0]
        limit = np.sqrt(6 / input_dim) / self.w0
        return tf.random.uniform(shape, minval=-limit, maxval=limit, dtype=dtype)

# Building the full model
def build_siren_model(input_dim, hidden_layers=15, hidden_units=200, w0=30.0, final_activation='sigmoid'):
    inputs = tf.keras.Input(shape=(input_dim,))

    # First layer
    x = tf.keras.layers.Dense(
        hidden_units,
        kernel_initializer=SIRENInitializer(w0=1.0),
        use_bias=False)(inputs)
    x = Sine(w0=1.0)(x)

    # Hidden layers
    for _ in range(hidden_layers):
        x = tf.keras.layers.Dense(
            hidden_units,
            kernel_initializer=SIRENInitializer(w0=w0))(x)
        x = Sine(w0=w0)(x)

    # Output layer
    outputs = tf.keras.layers.Dense(3, activation=final_activation)(x)

    model = tf.keras.Model(inputs, outputs)
    return model