import tensorflow as tf
from tensorflow.keras import models, layers

# Loading image dataset using tf.keras.preprocessing.image_dataset_from_directory() function
# specifying the directory where the dataset is located, a shuffle of the data,
# an image size of 256x256 and the batch size is 32
image_size = 256
batch_size = 32
channels = 3
dataset = tf.keras.preprocessing.image_dataset_from_directory("dataset", shuffle=True,
                                                              image_size=(image_size, image_size),
                                                              batch_size=batch_size)
class_names = dataset.class_names


# Splitting the dataset into train, validation and test sets
def get_dataset_partition_tf(ds, train_split=0.8, val_split=0.1, test_split=0.1, shuffle=True, shuffle_size=10000):
    ds_size = len(ds)
    if shuffle:
        ds = ds.shuffle(shuffle_size, seed=12)

    train_size = int(train_split * ds_size)
    val_size = int(val_split * ds_size)

    train_ds = ds.take(train_size)
    val_ds = ds.skip(train_size).take(val_size)
    test_ds = ds.skip(train_size).skip(val_size)

    return train_ds, val_ds, test_ds


# Assigning the train, validation and test sets
train_ds, val_ds, test_ds = get_dataset_partition_tf(dataset)

# optimize the split datasets
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
val_ds = val_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
test_ds = test_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)

# Preprocessing the dataset
resize_and_rescale = tf.keras.Sequential([layers.experimental.preprocessing.Resizing(image_size, image_size),
                                          layers.experimental.preprocessing.Rescaling(1.0 / 255)])
data_augmentation = tf.keras.Sequential([layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
                                         layers.experimental.preprocessing.RandomRotation(0.2)])

# building model
input_shape = (batch_size, image_size, image_size, channels)
n_classes = 3

model = models.Sequential([
    resize_and_rescale,  # Preprocessing step to resize and rescale images
    layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape), # 32 filters, 3x3 kernel size, relu activation
    layers.MaxPooling2D((2, 2)), # Max pooling layer
    layers.Conv2D(64, kernel_size=(3, 3), activation='relu'), # 64 filters, 3x3 kernel size, relu activation
    layers.MaxPooling2D((2, 2)), # Max pooling layer
    layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(), # flatten layer
    layers.Dense(64, activation='relu'),  # 64 neurons, relu activation
    layers.Dense(n_classes, activation='softmax'), # output layer with n_classes neurons, softmax activation
])

model.build(input_shape=input_shape)

model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=['accuracy'] # accuracy metric
)
history = model.fit(
    train_ds,
    batch_size=batch_size,
    validation_data=val_ds,
    verbose=1,
    epochs=50,
)

model.save("potatoes.h5")
