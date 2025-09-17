from tensorflow.keras.layers import Layer, GlobalAveragePooling2D, GlobalMaxPooling2D, Dense, Conv2D, Concatenate, Multiply
from tensorflow.keras import backend as K
try:
    # Keras 3
    from keras.saving import register_keras_serializable
except Exception:
    # TF 2.x fallback
    from tensorflow.keras.utils import register_keras_serializable

@register_keras_serializable(package="custom")
class CBAM(Layer):
    def __init__(self, reduction_ratio=8, **kwargs):
        super().__init__(**kwargs)
        self.reduction_ratio = reduction_ratio
        self._filters = None  # sera déterminé au build()

    def build(self, input_shape):
        self._filters = int(input_shape[-1])
        hidden = max(1, self._filters // self.reduction_ratio)

        # Channel attention MLP (partagée)
        self.shared_dense_one = Dense(hidden, activation='relu')
        self.shared_dense_two = Dense(self._filters)

        # Spatial attention conv
        self.spatial_conv = Conv2D(1, kernel_size=7, strides=1, padding='same', activation='sigmoid')

        super().build(input_shape)

    def call(self, inputs):
        # === Channel Attention ===
        avg_pool = GlobalAveragePooling2D()(inputs)  # (B, C)
        max_pool = GlobalMaxPooling2D()(inputs)      # (B, C)

        avg_pool = self.shared_dense_one(avg_pool)
        avg_pool = self.shared_dense_two(avg_pool)

        max_pool = self.shared_dense_one(max_pool)
        max_pool = self.shared_dense_two(max_pool)

        ch = K.sigmoid(avg_pool + max_pool)          # (B, C)
        ch = K.expand_dims(K.expand_dims(ch, 1), 1)  # (B, 1, 1, C)
        x = Multiply()([inputs, ch])                 # (B, H, W, C)

        # === Spatial Attention ===
        avg_pool = K.mean(x, axis=-1, keepdims=True) # (B, H, W, 1)
        max_pool = K.max(x,  axis=-1, keepdims=True) # (B, H, W, 1)
        concat = Concatenate(axis=-1)([avg_pool, max_pool])  # (B, H, W, 2)
        sp = self.spatial_conv(concat)                        # (B, H, W, 1)

        return Multiply()([x, sp])                    # (B, H, W, C)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"reduction_ratio": self.reduction_ratio})
        return cfg
