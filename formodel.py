import pathlib
import numpy as np
import cv2
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

img_width = 180
img_height = 180
batch_size = 64

extracted_faces_dir = pathlib.Path("Extracted Faces")
face_data_dir = pathlib.Path("Face Dataset")

extracted_faces_image_paths = list(extracted_faces_dir.glob('*/*.jpg'))
face_data_image_paths = list(face_data_dir.glob('*/*.jpg'))

images = []
labels = []

for image_path in extracted_faces_image_paths:
    label = 0
    img = cv2.imread(str(image_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (img_width, img_height))
    images.append(img)
    labels.append(label)

for image_path in face_data_image_paths:
    label = 1
    img = cv2.imread(str(image_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (img_width, img_height))
    images.append(img)
    labels.append(label)

images = np.array(images)
labels = np.array(labels)

X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=123)


def create_siamese_model(input_shape):
    input_layer = keras.layers.Input(shape=input_shape)
    base_network = keras.Sequential([
        keras.layers.Conv2D(16, (3, 3), activation='relu'),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Conv2D(32, (3, 3), activation='relu'),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Flatten(),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dense(32)
    ])

    left_input = keras.layers.Input(shape=input_shape)
    left_output = base_network(left_input)

    right_input = keras.layers.Input(shape=input_shape)
    right_output = base_network(right_input)

    L1_distance = keras.layers.Lambda(lambda tensors: keras.backend.abs(tensors[0] - tensors[1]))
    L1_distance_output = L1_distance([left_output, right_output])

    prediction = keras.layers.Dense(1, activation='sigmoid')(L1_distance_output)

    siamese_model = keras.models.Model(inputs=[left_input, right_input], outputs=prediction)

    return siamese_model


batch_size = 64

siamese_model = create_siamese_model((img_height, img_width, 3))
siamese_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

face_detection_epochs = 10
face_detection_history = siamese_model.fit(
    [X_train, X_train],
    y_train,
    validation_data=([X_test, X_test], y_test),
    epochs=face_detection_epochs,
    batch_size=batch_size
)

siamese_model.save('face_detection_model1')
print("Face Detection Model saved!")

num_classes = 2
face_recognition_model = keras.Sequential([
    keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Flatten(),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(num_classes, activation='softmax')
])

face_recognition_model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

face_recognition_epochs = 10
face_recognition_history = face_recognition_model.fit(
    X_train,
    y_train,
    validation_data=(X_test, y_test),
    epochs=face_recognition_epochs,
    batch_size=batch_size
)

face_recognition_model.save('face_recognition_model1')
print("Face Recognition Model saved!")

epochs_range = range(face_detection_epochs)

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, face_detection_history.history['accuracy'], label='Face Detection Training Accuracy')
plt.plot(epochs_range, face_detection_history.history['val_accuracy'], label='Face Detection Validation Accuracy')
plt.plot(epochs_range, face_recognition_history.history['accuracy'], label='Face Recognition Training Accuracy')
plt.plot(epochs_range, face_recognition_history.history['val_accuracy'], label='Face Recognition Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, face_detection_history.history['loss'], label='Face Detection Training Loss')
plt.plot(epochs_range, face_detection_history.history['val_loss'], label='Face Detection Validation Loss')
plt.plot(epochs_range, face_recognition_history.history['loss'], label='Face Recognition Training Loss')
plt.plot(epochs_range, face_recognition_history.history['val_loss'], label='Face Recognition Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')

plt.show()
