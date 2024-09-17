import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.callbacks import EarlyStopping

imgDirPath = 'Data Collection'

batchSize = 32
imgHeight = 32
imgWidth = 32

train_ds = keras.preprocessing.image_dataset_from_directory(
    imgDirPath,
    class_names=['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y'],
    color_mode='grayscale',
    batch_size=batchSize,
    image_size=(imgHeight, imgWidth),
    shuffle=True,
    seed=100,
    validation_split=0.3,
    subset="training"
)

val_ds = keras.preprocessing.image_dataset_from_directory(
    imgDirPath,
    class_names=['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y'],
    color_mode='grayscale',
    batch_size=batchSize,
    image_size=(imgHeight, imgWidth),
    shuffle=True,
    seed=100,
    validation_split=0.3,
    subset="validation"
)

model = tf.keras.Sequential([
    # Augment data
    layers.RandomRotation(0.3),
    layers.RandomZoom(0.1),
    layers.RandomContrast(0.1),

    layers.Rescaling(1./255),

    # Convolutional layers
    layers.Conv2D(32, (5, 5), activation='relu', input_shape=(32, 32, 1)),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    # Flatten and fully connected layers
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),

    # Output layer
    layers.Dense(24, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss=tf.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=['sparse_categorical_accuracy']
)

early_stopping = EarlyStopping(monitor='val_loss', patience=3)
history = model.fit(train_ds, epochs=10, validation_data=val_ds, callbacks=[early_stopping])
model.save('oms_model.keras')
