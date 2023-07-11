from tensorflow.keras.models import load_model
import tensorflow as tf
import os
import cv2
import imghdr
import numpy as np
from matplotlib import pyplot as plt


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from tensorflow.keras.metrics import Precision, Recall, CategoricalAccuracy


# Avoid OOM errors by setting GPU Memory Consumption Growth
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
print(tf.config.list_physical_devices('GPU'))


# REmoving dody images
data_dir = 'data'
image_exts = ['jpeg', 'jpg', 'bmp', 'png']
for image_class in os.listdir(data_dir):
    for image in os.listdir(os.path.join(data_dir, image_class)):
        image_path = os.path.join(data_dir, image_class, image)
        try:
            img = cv2.imread(image_path)
            tip = imghdr.what(image_path)
            if tip not in image_exts:
                print('Image not in ext list {}'.format(image_path))
                os.remove(image_path)
        except Exception as e:
            print('Issue with image {}'.format(image_path))
            # os.remove(image_path)


# creating dataset for training
data = tf.keras.utils.image_dataset_from_directory(
    'data', label_mode="categorical",)
data_iterator = data.as_numpy_iterator()
batch = data_iterator.next()
print(batch[1])
class_names = data.class_names
print("Class names are: ", class_names)
fig, ax = plt.subplots(ncols=12, figsize=(20, 20))
for idx, img in enumerate(batch[0][:12]):
    ax[idx].imshow(img.astype(int))
    ax[idx].title.set_text(class_names[np.argmax(batch[1][idx])])


# scaling data
data = data.map(lambda x, y: (x/255, y))
data.as_numpy_iterator().next()
print(len(data))


# splitting data
train_size = int(len(data)*.7)
val_size = int(len(data)*.2)
test_size = int(len(data)*.1)+1

print(f"training: {train_size},validation: {val_size},tesing: {test_size}")
train = data.take(train_size)
val = data.skip(train_size).take(val_size)
test = data.skip(train_size+val_size).take(test_size)

# building deep learning model:
model = Sequential()
model.add(Conv2D(16, (3, 3), 1, activation='relu', input_shape=(256, 256, 3)))
model.add(MaxPooling2D())
model.add(Conv2D(32, (3, 3), 1, activation='relu'))
model.add(MaxPooling2D())
model.add(Conv2D(16, (3, 3), 1, activation='relu'))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(8, activation='softmax'))

model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss=tf.losses.categorical_crossentropy, metrics=['accuracy'])
print(model.summary())

# training model
logdir = 'logs'
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
hist = model.fit(train, epochs=32, validation_data=val,
                 callbacks=[tensorboard_callback])

# Ploting performance
ig = plt.figure()
plt.plot(hist.history['loss'], color='teal', label='loss')
plt.plot(hist.history['val_loss'], color='orange', label='val_loss')
fig.suptitle('Loss', fontsize=20)
plt.legend(loc="upper left")
plt.show()

fig = plt.figure()
plt.plot(hist.history['accuracy'], color='teal', label='accuracy')
plt.plot(hist.history['val_accuracy'], color='orange', label='val_accuracy')
fig.suptitle('Accuracy', fontsize=20)
plt.legend(loc="upper left")
plt.show()

# evalutaing model
pre = Precision()
re = Recall()
acc = CategoricalAccuracy()
for batch in test.as_numpy_iterator():
    X, y = batch
    yhat = model.predict(X)
    pre.update_state(y, yhat)
    re.update_state(y, yhat)
    acc.update_state(y, yhat)

print(
    f"Precision: {pre.result()}, Recall: {re.result()}, Accuracy :{acc.result()}")

# Saving model
model.save(os.path.join('models', 'imageclassifier.h5'))

# Loading model
new_model = load_model(os.path.join('models', 'imageclassifier.h5'))
class_names = ['cave', 'city', 'desert', 'forest',
               'glacier', 'mountain', 'ocean', 'waterfall']

# Testing data:

# 1
img = cv2.imread("data/cave/images (100).jpeg")
resize = tf.image.resize(img, (256, 256))
plt.imshow(resize.numpy().astype(int))
plt.show()
yhat = new_model.predict(np.expand_dims(resize/255, 0))
print(
    f"Predicted class: {class_names[np.argmax(yhat,axis=1)[0]]}, confidence: {np.max(yhat)}")

# 2
img = cv2.imread("data/waterfall/images (49).jpeg")
resize = tf.image.resize(img, (256, 256))
plt.imshow(resize.numpy().astype(int))
plt.show()
yhat = new_model.predict(np.expand_dims(resize/255, 0))
print(
    f"Predicted class: {class_names[np.argmax(yhat,axis=1)[0]]}, confidence: {np.max(yhat)}")
