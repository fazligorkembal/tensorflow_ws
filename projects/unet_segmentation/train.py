import sys
sys.path.append('/home/user/Documents/tensorflow_ws/models')
sys.path.append('/home/user/Documents/tensorflow_ws/datagenerators')

import tensorflow as tf
from unet import Unet
from segmentation_dataloader import SegmentationDataloader
tf.config.experimental_run_functions_eagerly(True)

if __name__ == "__main__":
    input_layer = tf.keras.layers.Input(shape=(512, 512, 3))
    model = Unet().get_model(input_layer)
    model = tf.keras.Model(inputs=input_layer, outputs=model)
    model.summary()

    train_dataloader = SegmentationDataloader("/home/user/Documents/tensorflow_ws/data/peoples/PNGImages", "/home/user/Documents/tensorflow_ws/data/peoples/PedMasksOld", 4, shuffle=True)
    test_dataloader = SegmentationDataloader("/home/user/Documents/tensorflow_ws/data/peoples/PNGImages", "/home/user/Documents/tensorflow_ws/data/peoples/PedMasksOld", 4, shuffle=True)

    loss_object = tf.keras.losses.BinaryCrossentropy()
    optimizer = tf.keras.optimizers.Adam(lr=1e-4)

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.BinaryAccuracy(name='train_accuracy')

    test_loss = tf.keras.metrics.Mean(name='test_loss')
    test_accuracy = tf.keras.metrics.BinaryAccuracy(name='test_accuracy')
    
    @tf.function
    def train_step(images, labels):
        with tf.GradientTape() as tape:
            predictions = model(images, training=True)
            loss_value = loss_object(labels, predictions)
            
        gradients = tape.gradient(loss_value, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        
        train_loss(loss_value)
        train_accuracy(labels, predictions)
        

        #print(predictions.shape, predictions.dtype, tf.reduce_min(predictions), tf.reduce_max(predictions))        
        #print(labels.shape, labels.dtype, tf.reduce_min(labels), tf.reduce_max(labels))

    @tf.function
    def test_step(images, labels):
        predictions = model(images, training=False)
        loss_value = loss_object(labels, predictions)

        test_loss(loss_value)
        test_accuracy(labels, predictions)

    EPOCHS = 100
    batch_size = 4

    max_train_batch_index = len(train_dataloader.image_paths) // batch_size
    max_test_batch_index = len(test_dataloader.image_paths) // batch_size
    accuracy_holder = 0
    for epoch in range(EPOCHS):
        train_loss.reset_states()
        train_accuracy.reset_states()
        test_loss.reset_states()
        test_accuracy.reset_states()

        for batch_index in range(max_train_batch_index):
            imgs, msks = train_dataloader.__getitem__(batch_index)
            train_step(imgs, msks)
        
        for batch_index in range(max_test_batch_index):
            imgs, msks = test_dataloader.__getitem__(batch_index)
            test_step(imgs, msks)

        template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
        print(template.format(epoch+1,
                            train_loss.result(),
                            train_accuracy.result()*100,
                            test_loss.result(),
                            test_accuracy.result()*100))
        if train_accuracy.result() > accuracy_holder:
            model.save_weights("weights.h5")
            accuracy_holder = train_accuracy.result()
            print("Saved weights")
        