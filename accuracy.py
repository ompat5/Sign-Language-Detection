import tensorflow as tf
from tensorflow import keras

imgDirPath = 'Data Collection'

model = tf.keras.models.load_model('oms_model.keras')

test_ds = keras.preprocessing.image_dataset_from_directory(
    imgDirPath,
    class_names=['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y'],
    color_mode='grayscale',
    batch_size=32,
    image_size=(32, 32),
    shuffle=False,
    seed=100
)

test_loss, test_accuracy = model.evaluate(test_ds)

print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
