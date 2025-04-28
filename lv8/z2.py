import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from keras.models import load_model
from tensorflow.keras.utils import to_categorical



# Učitaj model
model = load_model("model.keras")

# Učitaj MNIST skup podataka
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()


print('Train: X=%s, y=%s' % (x_train.shape, y_train.shape))
print('Test: X=%s, y=%s' % (x_test.shape, y_test.shape))


x_train_s = x_train.astype("float32") / 255
x_test_s = x_test.astype("float32") / 255

# slike trebaju biti (28, 28, 1)
x_train_s = x_train_s.reshape((-1, 784))
x_test_s = x_test_s.reshape((-1, 784))

print("x_train shape:", x_train_s.shape)
print(x_train_s.shape[0], "train samples")
print(x_test_s.shape[0], "test samples")


# pretvori labele
y_train_s = keras.utils.to_categorical(y_train, 10)
y_test_s = keras.utils.to_categorical(y_test, 10)


predictions = model.predict(x_test_s)


y_pred_classes = np.argmax(predictions, axis=1)
y_true_classes = np.argmax(y_test_s, axis=1)

incorrect = [i for i, y in enumerate(y_pred_classes) if y != y_true_classes[i]]

print(len(incorrect))
plt.figure(figsize=(10, 5))
for i in range(5):
    index = incorrect[i]
    plt.subplot(1, 5, i + 1)
    plt.imshow(x_test[index])
    true_label = np.argmax(y_test_s[index])
    predicted_label = np.argmax(predictions[index])
    plt.title(f'True: {true_label}\nPred: {predicted_label}')

plt.tight_layout()
plt.show()
