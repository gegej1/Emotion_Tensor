# adapted from https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html
# for a high level explanation of this, check out https://github.com/SchoolofAI-Vancouver/learn_image_classification_2/blob/master/keras_colab_demo/Image_Classification_II_Demo.ipynb
# for a more detailed explanation, look at the original source at the top
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense
from tensorflow.keras import backend as K

# dimensions of our images
img_width, img_height = 64, 64

# data dirs
train_data_dir = '../../data/train'
validation_data_dir = '../../data/validate'

# basic training params - 更新为实际数据量
nb_train_samples = 953  # 322 + 631 = 953 (训练集总数)
nb_validation_samples = 468  # 232 + 236 = 468 (验证集总数)
epochs = 20  # 增加训练轮数以提高准确率
batch_size = 32  # 增加批次大小以加快训练

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

# this is the augmentation configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator(rescale=1. / 255)

# 打印类别信息
print("训练数据生成器初始化中...")

# generate training data
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')

# generate validation data
validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')

print("开始训练模型...")
# fit model
history = model.fit(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples // batch_size)

# 打印训练结果
print(f"训练完成！最终训练准确率: {history.history['accuracy'][-1]:.4f}")
if 'val_accuracy' in history.history:
    print(f"验证准确率: {history.history['val_accuracy'][-1]:.4f}")

# save model
print("保存模型...")
model.save('distraction_model.hdf5')
print("模型已保存为 'distraction_model.hdf5'")