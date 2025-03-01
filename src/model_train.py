import numpy as np
from get_data import read_params
from keras_preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Model
import os
import argparse
import matplotlib.pyplot as plt


def train_model(config_file):
    config = read_params(config_file)
    if config['model']['trainable']:
        img_size = tuple(config['model']['image_size'])
        train_set = config['model']['train_path']
        test_set = config['model']['test_path']
        num_classes = config['load_data']['num_classes']
        batch_size = config['model']['batch_size']
        epochs = config['model']['epochs']
        loss = config['model']['loss']
        optimizer = config['model']['optimizer']
        metrics = config['model']['metrics']
        sav_dir = config['model_dir']
        rescale = float(config['img_augment']['rescale'])

        vgg16 = VGG16(input_shape=img_size + (3,), weights='imagenet', include_top=False)
        for layer in vgg16.layers:
            layer.trainable = False

        op = Flatten()(vgg16.output)
        prediction = Dense(num_classes, activation='softmax')(op)
        model = Model(inputs=vgg16.input, outputs=prediction)

        model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

        train_datagen = ImageDataGenerator(rescale=rescale,
                                           shear_range=config['img_augment']['shear_range'],
                                           zoom_range=config['img_augment']['zoom_range'],
                                           horizontal_flip=config['img_augment']['horizontal_flip'],
                                           vertical_flip=config['img_augment']['vertical_flip'],
                                           rotation_range=90,
                                           width_shift_range=0.1,
                                           height_shift_range=0.1)

        test_datagen = ImageDataGenerator(rescale=rescale)

        train_generator = train_datagen.flow_from_directory(train_set,
                                                            target_size=img_size,
                                                            batch_size=batch_size,
                                                            class_mode="categorical")

        test_generator = test_datagen.flow_from_directory(test_set,
                                                          target_size=img_size,
                                                          batch_size=batch_size,
                                                          class_mode="categorical")

        history = model.fit(train_generator,
                            epochs=epochs,
                            validation_data=test_generator,
                            steps_per_epoch=len(train_generator),
                            validation_steps=len(test_generator))

        plt.plot(history.history['loss'], label='train loss')
        plt.plot(history.history['val_loss'], label='val loss')
        plt.plot(history.history['accuracy'], label='train acc')
        plt.plot(history.history['val_accuracy'], label='val acc')
        plt.legend()
        plt.savefig(os.path.join(sav_dir, 'training_history.png'))

        model.save(os.path.join(sav_dir, 'trained.h5'))
        print("Model Saved Successfully....!")
    else:
        print("Model is not trainable")


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--config', default='params.yaml')
    passed_args = args.parse_args()
    train_model(config_file=passed_args.config)
