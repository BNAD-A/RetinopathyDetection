import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import CSVLogger

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
log_dir = os.path.join(base_dir, 'outputs', 'logs')
os.makedirs(log_dir, exist_ok=True)
csv_logger = CSVLogger(os.path.join(log_dir, 'mobileNetV2FocalLoss_history.csv'))

npz_train = os.path.join(base_dir, 'processed_train_images.npz')
npz_val = os.path.join(base_dir, 'processed_val_images.npz')

train_data = np.load(npz_train)
X_train, y_train = train_data['X'], train_data['y']

val_data = np.load(npz_val)
X_val, y_val = val_data['X'], val_data['y']

classes = np.unique(y_train)
class_weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
class_weight_dict = {int(k): float(v) for k, v in zip(classes, class_weights)}
alpha = [class_weight_dict.get(i, 1.0) for i in range(5)]

def focal_loss(gamma=2., alpha=None):
    def focal_loss_fixed(y_true, y_pred):
        y_true = tf.cast(y_true, tf.int32)
        y_true_one_hot = tf.one_hot(y_true, depth=y_pred.shape[-1])
        cross_entropy = tf.keras.losses.categorical_crossentropy(y_true_one_hot, y_pred)
        p_t = tf.reduce_sum(y_true_one_hot * y_pred, axis=-1)
        if alpha is not None:
            alpha_tensor = tf.constant(alpha, dtype=tf.float32)
            alpha_factor = tf.gather(alpha_tensor, y_true)
        else:
            alpha_factor = 1.0
        modulating_factor = tf.pow(1.0 - p_t, gamma)
        return alpha_factor * modulating_factor * cross_entropy
    return focal_loss_fixed

base_model = tf.keras.applications.MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
base_model.trainable = False

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(5, activation='softmax')
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
              loss=focal_loss(gamma=2., alpha=alpha),
              metrics=['accuracy'])

model.summary()

model.fit(X_train, y_train,
          validation_data=(X_val, y_val),
          epochs=20,
          batch_size=32,
          callbacks=[csv_logger])

model_save_path = os.path.join(base_dir, 'models', 'mobileNetV2FocalLoss.keras')
model.save(model_save_path)
print(f"Modèle sauvegardé à : {model_save_path}")
