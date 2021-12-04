import tensorflow as tf
from tensorflow.keras import layers,models
import matplotlib.pyplot as plt
from tensorflow.python.keras.layers.normalization_v2 import BatchNormalization
import numpy as np

# Constants
BATCH_SIZE = 16
IMAGE_SIZE = 60
Channel = 1
Epochs = 50

dataset = tf.keras.preprocessing.image_dataset_from_directory(
    "dataset/COVID-19_Radiography_Dataset",shuffle = True,image_size =(IMAGE_SIZE,IMAGE_SIZE),batch_size= BATCH_SIZE,color_mode='grayscale'
)
class_names = dataset.class_names
'''print(class_names)
plt.figure(figsize=(10,10))
for image_batch,labels_batch in dataset.take(1):
    print(image_batch.shape)
    print(labels_batch)
    print(labels_batch.numpy())
    for i in range(12):
        ax = plt.subplot(3,4,i+1)
        plt.imshow(image_batch[i].numpy().astype("uint8"))
        plt.title(class_names[labels_batch[i].numpy()])
        plt.axis("off")'''

def get_dataset_partition(dataset,train_split = 0.8,val_split = 0.1,test_split = 0.1,shuffle = True,shuffle_size = 10000):
    assert(train_split+val_split+test_split)==1
    data_size = len(dataset)
    if shuffle:
        dataset = dataset.shuffle(shuffle_size,seed=12)
    train_size = int(train_split*data_size)
    val_size = int(val_split*data_size)

    train_data = dataset.take(train_size)
    val_data = dataset.skip(train_size).take(val_size)
    test_data = dataset.skip(train_size).skip(val_size)

    return train_data,val_data,test_data

train_data, val_data, test_data = get_dataset_partition(dataset,train_split=0.8,val_split=0.1,test_split=0.1,shuffle=True)

train_data = train_data.cache().shuffle(1000).prefetch(buffer_size = tf.data.AUTOTUNE)
val_data = val_data.cache().shuffle(1000).prefetch(buffer_size = tf.data.AUTOTUNE)
test_data = test_data.cache().shuffle(1000).prefetch(buffer_size = tf.data.AUTOTUNE)

resize_rescale = tf.keras.Sequential([layers.experimental.preprocessing.Resizing(IMAGE_SIZE,IMAGE_SIZE),layers.experimental.preprocessing.Rescaling(1.0/255)])
augmentation = tf.keras.Sequential([layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),layers.experimental.preprocessing.RandomRotation(0.2),layers.experimental.preprocessing.RandomZoom(0.2)])

input_shape = (BATCH_SIZE,IMAGE_SIZE,IMAGE_SIZE,1)
n_class = 4
model = models.Sequential(
    [resize_rescale,
    augmentation,
    layers.Conv2D(16,kernel_size=(3,3),activation='relu',input_shape=input_shape,padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2,2)),
    layers.Dropout(0.2),
    layers.Conv2D(64,kernel_size=(3,3),activation='relu',padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2,2)),
    layers.Dropout(0.2),
    layers.Conv2D(128,kernel_size=(3,3),activation='relu',padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2,2)),
    layers.Dropout(0.2),
    layers.Conv2D(64,kernel_size=(3,3),activation='relu',padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2,2)),
    layers.Flatten(),
    layers.Dense(64,activation='relu'),
    layers.Dense(n_class,activation='softmax')]
)
model.build(input_shape=input_shape)

model.summary()

model.compile(
    optimizer='rmsprop',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=['accuracy']
)

history = model.fit(train_data,batch_size= BATCH_SIZE,validation_data = val_data,verbose = 1,epochs = Epochs)

score = model.evaluate(test_data)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(range(Epochs), acc, label='Training Accuracy')
plt.plot(range(Epochs), val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(range(Epochs), loss, label='Training Loss')
plt.plot(range(Epochs), val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()


model.save("C:/my-files/programming/covid-19-lungs-radiography/models")

model.save("C:/my-files/programming/covid-19-lungs-radiography/covid-19-lungs.h5")