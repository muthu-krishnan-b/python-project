# Import required libraries
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix

# Set dataset paths
train_dir = r"C:\Users\91786\PycharmProjects\pythonProject3\chest-xray-pneumonia\chest_xray\train"
test_dir = r"C:\Users\91786\PycharmProjects\pythonProject3\chest-xray-pneumonia\chest_xray\test"
val_dir = r"C:\Users\91786\PycharmProjects\pythonProject3\chest-xray-pneumonia\chest_xray\val"


# Verify dataset paths
for directory in [train_dir, test_dir, val_dir]:
    if not os.path.exists(directory):
        raise FileNotFoundError(f"Directory not found: {directory}")

# Image Augmentation and Preprocessing
train_datagen = ImageDataGenerator(rescale=1.0 / 255,
                                   rotation_range=20,
                                   zoom_range=0.2,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   horizontal_flip=True)

test_val_datagen = ImageDataGenerator(rescale=1.0 / 255)

# Create data generators
train_generator = train_datagen.flow_from_directory(train_dir,
                                                    target_size=(150, 150),
                                                    batch_size=32,
                                                    class_mode='binary')

val_generator = test_val_datagen.flow_from_directory(val_dir,
                                                     target_size=(150, 150),
                                                     batch_size=32,
                                                     class_mode='binary')

test_generator = test_val_datagen.flow_from_directory(test_dir,
                                                      target_size=(150, 150),
                                                      batch_size=32,
                                                      class_mode='binary',
                                                      shuffle=False)

# CNN Model
cnn_model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')  # Binary classification
])

cnn_model.compile(optimizer=Adam(learning_rate=0.0001),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

# Train CNN Model
cnn_history = cnn_model.fit(train_generator,
                            validation_data=val_generator,
                            epochs=20)

# ResNet50 Pre-trained Model
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(150, 150, 3))

# Freeze base layers
for layer in base_model.layers:
    layer.trainable = False

# Add custom layers
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
output = Dense(1, activation='sigmoid')(x)

resnet_model = Model(inputs=base_model.input, outputs=output)

resnet_model.compile(optimizer=Adam(learning_rate=0.0001),
                     loss='binary_crossentropy',
                     metrics=['accuracy'])

# Train ResNet50 Model
resnet_history = resnet_model.fit(train_generator,
                                  validation_data=val_generator,
                                  epochs=20)

# Model Evaluation
cnn_loss, cnn_accuracy = cnn_model.evaluate(test_generator)
resnet_loss, resnet_accuracy = resnet_model.evaluate(test_generator)

print(f"CNN Accuracy: {cnn_accuracy * 100:.2f}%")
print(f"ResNet50 Accuracy: {resnet_accuracy * 100:.2f}%")

# Predictions
cnn_predictions = cnn_model.predict(test_generator)
resnet_predictions = resnet_model.predict(test_generator)

cnn_pred_classes = (cnn_predictions > 0.5).astype("int32")
resnet_pred_classes = (resnet_predictions > 0.5).astype("int32")

# Confusion Matrix - ResNet50
cm = confusion_matrix(test_generator.classes, resnet_pred_classes)
plt.figure(figsize=(6, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Normal', 'Pneumonia'], yticklabels=['Normal', 'Pneumonia'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix - ResNet50')
plt.show()

# Classification Report
print("ResNet50 Classification Report:")
print(classification_report(test_generator.classes, resnet_pred_classes, target_names=['Normal', 'Pneumonia']))

# Plot CNN Training Results
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(cnn_history.history['accuracy'], label='Train Accuracy')
plt.plot(cnn_history.history['val_accuracy'], label='Validation Accuracy')
plt.title('CNN Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(cnn_history.history['loss'], label='Train Loss')
plt.plot(cnn_history.history['val_loss'], label='Validation Loss')
plt.title('CNN Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

# Plot ResNet50 Training Results
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(resnet_history.history['accuracy'], label='Train Accuracy')
plt.plot(resnet_history.history['val_accuracy'], label='Validation Accuracy')
plt.title('ResNet50 Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(resnet_history.history['loss'], label='Train Loss')
plt.plot(resnet_history.history['val_loss'], label='Validation Loss')
plt.title('ResNet50 Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()
