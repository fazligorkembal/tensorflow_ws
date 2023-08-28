import sys
sys.path.append('/home/user/Documents/tensorflow_ws/datagenerators')
sys.path.append('/home/user/Documents/tensorflow_ws/models')

import tensorflow as tf
from unet import Unet
from segmentation_dataloader import SegmentationDataloader
tf.config.experimental_run_functions_eagerly(True)


if __name__ == "__main__":
    # load Weights
    weight_path = "/home/user/Documents/tensorflow_ws/weights.h5"
    input = tf.keras.layers.Input(shape=(512, 512, 3))
    model = Unet().get_model(input)
    model = tf.keras.Model(inputs=input, outputs=model)
    model.load_weights(weight_path)
    
    # load dataloader
    test_dataloader = SegmentationDataloader("/home/user/Documents/tensorflow_ws/data/peoples/PNGImages", "/home/user/Documents/tensorflow_ws/data/peoples/PedMasksOld", 4, shuffle=True)
    import numpy as np
    import cv2

    images = test_dataloader[0][0]

    predict = model.predict(images)
    predict = np.where(predict > 0.5, 1, 0) * 255
    predict = predict.astype(np.uint8)

    #tf to numpy
    images = images.numpy()
    images = images * 255
    images = images.astype(np.uint8)

    for i in range(len(images)):
        image = cv2.cvtColor(images[i], cv2.COLOR_RGB2BGR)
        cv2.imshow("image", image)
        cv2.imshow("predict", predict[i])
        cv2.waitKey(0)
        
