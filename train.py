import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

# =========================
# LOAD DATASET
# =========================

dataset = tf.keras.utils.image_dataset_from_directory(
    "dataset",
    image_size=(128, 128),
    batch_size=32
)

# =========================
# CLASS NAMES
# =========================

class_names = dataset.class_names
print("Classes:", class_names)

# =========================
# NORMALIZE DATA
# =========================

normalization_layer = layers.Rescaling(1./255)

dataset = dataset.map(
    lambda x, y: (normalization_layer(x), y)
)

# =========================
# BUILD CNN MODEL
# =========================

model = models.Sequential([

    layers.Conv2D(32, (3,3), activation='relu', input_shape=(128,128,3)),
    layers.MaxPooling2D(),

    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(),

    layers.Conv2D(128, (3,3), activation='relu'),
    layers.MaxPooling2D(),

    layers.Flatten(),

    layers.Dense(128, activation='relu'),

    layers.Dense(1, activation='sigmoid')
])

# =========================
# COMPILE MODEL
# =========================

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# =========================
# TRAIN MODEL
# =========================

history = model.fit(
    dataset,
    epochs=10
)

# =========================
# SAVE MODEL
# =========================

model.save("mask_model.h5")

print("Model saved!")

# =========================
# PLOT ACCURACY
# =========================

plt.plot(history.history['accuracy'])
plt.title("Model Accuracy")
plt.ylabel("Accuracy")
plt.xlabel("Epoch")
plt.show()