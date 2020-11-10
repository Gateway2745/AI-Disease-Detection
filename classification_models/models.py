import keras
from keras import applications
from keras.models import Model, Sequential
from keras.layers import Dense, Dropout, Flatten, Input
from keras.layers import Conv2D, MaxPooling2D
from keras import regularizers
import numpy as np

input_shape = (200,350,3)
input_tensor = Input(shape=input_shape)

def get_model(num, print_model=False, reg=False, reg_type=None, num_classes=2):
    """
    reg_type is only in accesssed if reg is True
    reg_type is a dictionary with keys
        dropout: [<rate>, ...]
        l2: [<rate>, ...]
    """
    model = Sequential()

    if num == 1:
        if reg:
            model.add(Conv2D(16, kernel_size=(3, 3),
                         activation='relu',
                         input_shape=input_shape,
                         activity_regularizer=regularizers.l2(reg_type['l2'][0]),))
            model.add(Dropout(reg_type['dropout'][0]))
            model.add(Conv2D(4, (5, 5),
                        activation='relu',
                        activity_regularizer=regularizers.l2(reg_type['l2'][1]),))
            model.add(Dropout(reg_type['dropout'][1]))
        else:
            model.add(Conv2D(16, kernel_size=(3, 3),
                         activation='relu',
                         input_shape=input_shape))
            model.add(Conv2D(4, (5, 5),
                        activation='relu'),)

        model.add(MaxPooling2D(pool_size=(7, 7)))
        model.add(Flatten())
        model.add(Dense(num_classes, activation='softmax'))

    elif num == 2:
        if reg:
            model.add(Conv2D(4, kernel_size=(3, 3),
                         activation='relu',
                         input_shape=input_shape,
                         activity_regularizer=regularizers.l2(reg_type['l2'][0]),))
            model.add(MaxPooling2D(pool_size=(5, 5)))
            model.add(Dropout(reg_type['dropout'][0]))

            model.add(Conv2D(16, kernel_size=(3, 3),
                         activation='relu',
                         input_shape=input_shape,
                         activity_regularizer=regularizers.l2(reg_type['l2'][1]),))
            model.add(MaxPooling2D(pool_size=(5, 5)))
            model.add(Dropout(reg_type['dropout'][1]))

            model.add(Conv2D(1, kernel_size=(3, 3),
                         activation='relu',
                         input_shape=input_shape,
                         activity_regularizer=regularizers.l2(reg_type['l2'][2]),))
            model.add(Dropout(reg_type['dropout'][2]))

        else:
            model.add(Conv2D(4, kernel_size=(3, 3),
                         activation='relu',
                         input_shape=input_shape))
            model.add(MaxPooling2D(pool_size=(5, 5)))

            model.add(Conv2D(16, kernel_size=(3, 3),
                         activation='relu',
                         input_shape=input_shape))
            model.add(MaxPooling2D(pool_size=(5, 5)))
            model.add(Conv2D(1, kernel_size=(3, 3),
                         activation='relu',
                         input_shape=input_shape))
        model.add(Flatten())
        model.add(Dense(num_classes, activation='softmax'))

    elif num == 3:
        if reg:
            model.add(Conv2D(4, kernel_size=(3, 3),
                         activation='relu',
                         input_shape=input_shape,
                         activity_regularizer=regularizers.l2(reg_type['l2'][0]),))
            model.add(MaxPooling2D(pool_size=(3, 3)))
            model.add(Dropout(reg_type['dropout'][0]))

            model.add(Conv2D(16, kernel_size=(3, 3), strides=(1,1),
                         activation='relu',
                         input_shape=input_shape,
                         activity_regularizer=regularizers.l2(reg_type['l2'][1]),))
            model.add(MaxPooling2D(pool_size=(3, 3)))
            model.add(Dropout(reg_type['dropout'][1]))

            model.add(Conv2D(1, kernel_size=(3, 3), strides=(1,1),
                         activation='relu',
                         input_shape=input_shape,
                         activity_regularizer=regularizers.l2(reg_type['l2'][2]),))
            model.add(Dropout(reg_type['dropout'][2]))
        else:
            model.add(Conv2D(4, kernel_size=(3, 3),
                         activation='relu',
                         input_shape=input_shape))
            model.add(MaxPooling2D(pool_size=(3, 3)))

            model.add(Conv2D(16, kernel_size=(3, 3), strides=(1,1),
                         activation='relu',
                         input_shape=input_shape))
            model.add(MaxPooling2D(pool_size=(3, 3)))
            model.add(Conv2D(1, kernel_size=(3, 3), strides=(1,1),
                         activation='relu',
                         input_shape=input_shape))
        model.add(Flatten())
        model.add(Dense(num_classes, activation='softmax'))

    elif num == 4: # VGG Model
        base_model = applications.VGG16(weights='imagenet',
                                       include_top=False,
                                       input_tensor=input_tensor)

        layer_dict = dict([(layer.name, layer) for layer in base_model.layers])
        num = 6
        x = layer_dict['block2_pool'].output
        x = Conv2D(filters=4, kernel_size=(1, 1), activation='relu')(x)
        x = MaxPooling2D(pool_size=(5, 5))(x)
        x = Flatten()(x)
        x = Dense(32, activation='relu')(x)
        if reg:
            num +=1
            x = Dropout(reg_type['dropout'][0])(x)
        x = Dense(num_classes, activation='softmax')(x)
        model = Model(base_model.input, x)
        for layer in model.layers[:-num]:
            layer.trainable = False

    elif num == 5: # Resnet 50 v2
        base_model = applications.ResNet50V2(weights='imagenet',
                                       include_top=False,
                                       input_tensor=input_tensor)

        layer_dict = dict([(layer.name, layer) for layer in base_model.layers])
        num = 5
        x = layer_dict['post_relu'].output
        x = Conv2D(filters=8, kernel_size=(1, 1), activation='relu')(x)
        x = MaxPooling2D(pool_size=(5, 5))(x)
        if reg:
            num +=1
            x = Dropout(reg_type['dropout'][0])(x)
        x = Flatten()(x)
        x = Dense(num_classes, activation='softmax')(x)
        model = Model(base_model.input, x)
        for layer in model.layers[:-num]:
            layer.trainable = False

    elif num == 6: # MobileNet v2
        base_model = applications.MobileNetV2(weights='imagenet',
                                       include_top=False,
                                       input_tensor=input_tensor)

        layer_dict = dict([(layer.name, layer) for layer in base_model.layers])
        num = 5
        x = layer_dict['out_relu'].output
        x = Conv2D(filters=16, kernel_size=(1, 1), activation='relu')(x)
        x = MaxPooling2D(pool_size=(5, 5))(x)
        if reg:
            num +=1
            x = Dropout(reg_type['dropout'][0])(x)
        x = Flatten()(x)
        x = Dense(num_classes, activation='softmax')(x)
        model = Model(base_model.input, x)
        for layer in model.layers[:-num]:
            layer.trainable = False

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adam(),
                  metrics=['accuracy'],)
    if print_model:
        print(model.summary())
    return model
