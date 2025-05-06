# Neural Networking
import tensorflow as tf
import numpy as np

@tf.keras.utils.register_keras_serializable()
class Sine(tf.keras.layers.Layer):
    def __init__(self, w0=1.0, **kwargs):
        super().__init__(**kwargs)
        self.w0 = w0

    def call(self, inputs):
        return tf.sin(self.w0 * inputs)

    def get_config(self):
        config = super().get_config()
        config.update({"w0": self.w0})
        return config


@tf.keras.utils.register_keras_serializable()
class SIRENInitializer(tf.keras.initializers.Initializer):
    def __init__(self, w0=1.0):
        self.w0 = w0

    def __call__(self, shape, dtype=None):
        input_dim = shape[0]
        limit = np.sqrt(6 / input_dim) / self.w0
        return tf.random.uniform(shape, minval=-limit, maxval=limit, dtype=dtype)

    def get_config(self):
        return {"w0": self.w0}

class ModelConfig:
    def __init__(
        self,
        input_dim=7,
        hidden_layers=14,
        hidden_units=256,
        w0=1.0,
        w0_initial=30.0,
        final_activation='sigmoid'
    ):
        self.input_dim = input_dim
        self.hidden_layers = hidden_layers
        self.hidden_units = hidden_units
        self.w0 = w0
        self.w0_initial = w0_initial
        self.final_activation = final_activation

def build_siren_model(config: ModelConfig) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(config.input_dim,))

    # First layer: high-frequency initializer and activation
    x = tf.keras.layers.Dense(
        config.hidden_units,
        kernel_initializer=SIRENInitializer(w0=config.w0_initial),
        use_bias=True)(inputs)
    x = Sine(w0=config.w0_initial)(x)

    # Hidden layers
    for _ in range(config.hidden_layers):
        x = tf.keras.layers.Dense(
            config.hidden_units,
            kernel_initializer=SIRENInitializer(w0=config.w0),
            use_bias=True)(x)
        x = Sine(w0=config.w0)(x)

    # Output layer
    outputs = tf.keras.layers.Dense(3, activation=config.final_activation)(x)

    return tf.keras.Model(inputs, outputs)
