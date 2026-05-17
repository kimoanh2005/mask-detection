import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt

# =========================
# LOAD MODEL
# =========================

model = tf.keras.models.load_model("mask_model.h5")

# =========================
# LOAD IMAGE
# =========================

img_path = "test_images/test1.jpg"

img = image.load_img(img_path, target_size=(128, 128))

# =========================
# CONVERT IMAGE
# =========================

img_array = image.img_to_array(img)

img_array = img_array / 255.0

img_array = np.expand_dims(img_array, axis=0)

# =========================
# PREDICT
# =========================

prediction = model.predict(img_array)

print("Prediction:", prediction)

# =========================
# RESULT
# =========================

if prediction[0][0] > 0.5:
    result = "No Mask 😐"
else:
    result = "Mask 😷"

print("Result:", result)

# =========================
# SHOW IMAGE
# =========================

plt.imshow(image.load_img(img_path))
plt.title(result)
plt.axis("off")
plt.show()