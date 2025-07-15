import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import CSVLogger

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
log_dir = os.path.join(base_dir, 'outputs', 'logs')
os.makedirs(log_dir, exist_ok=True)
csv_logger = CSVLogger(os.path.join(log_dir, 'mobileNetV2_history.csv'))

npz_train = os.path.join(base_dir, 'processed_train_images.npz')
npz_val = os.path.join(base_dir, 'processed_val_images.npz')
npz_test = os.path.join(base_dir, 'processed_test_images.npz')

train_data = np.load(npz_train)
X_train, y_train = train_data['X'], train_data['y']

val_data = np.load(npz_val)
X_val, y_val = val_data['X'], val_data['y']

classes = np.unique(y_train)
class_weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
class_weight_dict = {int(k): float(v) for k, v in zip(classes, class_weights)}

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
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

model.fit(X_train, y_train,
          validation_data=(X_val, y_val),
          epochs=20,
          batch_size=32,
          class_weight=class_weight_dict,
          callbacks=[csv_logger])

model_save_path = os.path.join(base_dir, 'models', 'mobileNetV2.keras')
model.save(model_save_path)
print(f"Modèle sauvegardé à : {model_save_path}")
