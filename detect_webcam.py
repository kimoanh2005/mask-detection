import cv2
import numpy as np
import tensorflow as tf

# =========================
# LOAD MODEL
# =========================

model = tf.keras.models.load_model("mask_model.h5")

# =========================
# START WEBCAM
# =========================

cap = cv2.VideoCapture(0)

# =========================
# LOOP CAMERA
# =========================

while True:

    ret, frame = cap.read()

    if not ret:
        break

    # =========================
    # RESIZE FOR MODEL
    # =========================

    img = cv2.resize(frame, (128, 128))

    # =========================
    # NORMALIZE IMAGE
    # =========================

    img = img / 255.0

    img = np.expand_dims(img, axis=0)

    # =========================
    # PREDICT
    # =========================

    prediction = model.predict(img, verbose=0)

    # =========================
    # RESULT
    # =========================

    if prediction[0][0] > 0.5:
        label = "No Mask 😐"
        color = (0, 0, 255)
    else:
        label = "Mask 😷"
        color = (0, 255, 0)

    # =========================
    # DRAW TEXT
    # =========================

    cv2.putText(
        frame,
        label,
        (20, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        color,
        2
    )

    # =========================
    # SHOW FRAME
    # =========================

    cv2.imshow("Mask Detection", frame)

    # =========================
    # PRESS Q TO EXIT
    # =========================

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# =========================
# RELEASE CAMERA
# =========================

cap.release()
cv2.destroyAllWindows()