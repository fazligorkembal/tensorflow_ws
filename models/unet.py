import tensorflow as tf

class Unet:
    def __init__(self, input_shape=(512, 512, 3), num_classes=1):
        self.num_classes = num_classes
        self.input_shape = input_shape

    def get_model(self, inputs, training=False):
        x = tf.keras.layers.Conv2D(32, 3, strides=2, padding='same')(inputs)
        x = tf.keras.layers.BatchNormalization()(x, training=training)
        x = tf.keras.layers.ReLU()(x)

        previous_block_activation = x

        for filters in [64, 128, 256]:
            x = tf.keras.layers.Activation('relu')(x)
            x = tf.keras.layers.SeparableConv2D(filters, 3, padding='same')(x)
            x = tf.keras.layers.BatchNormalization()(x, training=training)

            x = tf.keras.layers.Activation('relu')(x)
            x = tf.keras.layers.SeparableConv2D(filters, 3, padding='same')(x)
            x = tf.keras.layers.BatchNormalization()(x, training=training)

            x = tf.keras.layers.MaxPool2D(3, strides=2, padding='same')(x)

            residual = tf.keras.layers.Conv2D(filters, 1, strides=2, padding='same')(previous_block_activation)

            x = tf.keras.layers.add([x, residual])
            previous_block_activation = x

        for filters in [256, 128, 64, 32]:
            x = tf.keras.layers.Activation('relu')(x)
            x = tf.keras.layers.Conv2DTranspose(filters, 3, padding='same')(x)
            x = tf.keras.layers.BatchNormalization()(x, training=training)

            x = tf.keras.layers.Activation('relu')(x)
            x = tf.keras.layers.Conv2DTranspose(filters, 3, padding='same')(x)
            x = tf.keras.layers.BatchNormalization()(x, training=training)

            x = tf.keras.layers.UpSampling2D(2)(x)

            residual = tf.keras.layers.UpSampling2D(2)(previous_block_activation)
            residual = tf.keras.layers.Conv2D(filters, 1, padding='same')(residual)

            x = tf.keras.layers.add([x, residual])
            previous_block_activation = x

        outputs = tf.keras.layers.Conv2D(self.num_classes, 3, activation='sigmoid', padding='same')(x)

        return outputs
  
        
if __name__ == "__main__":
    import numpy as np
    
    input_layer = tf.keras.layers.Input(shape=(512, 512, 3))
    model = Unet().get_model(input_layer)
    model = tf.keras.Model(inputs=input_layer, outputs=model)
    model.summary()
    
    