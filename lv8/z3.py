import numpy as np
from tensorflow.keras.models import load_model
from tensorflow import keras
from PIL import Image
import matplotlib.pyplot as plt

model=load_model('model.keras')

img = Image.open("test.png").convert('L')

img = img.resize((28, 28))                 
img_array = np.array(img).astype('float32') / 255 
img_array = np.expand_dims(img_array, axis=-1)  
img_array = np.expand_dims(img_array, axis=0)   

prediction = model.predict(img_array)
predicted_class = np.argmax(prediction)

plt.imshow(img_array[0, :, :, 0], cmap='gray')
plt.title(f"Predicted: {predicted_class}")
plt.axis('off')
plt.show()