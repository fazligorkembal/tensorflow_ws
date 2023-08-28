import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

class SegmentationDataloader(tf.keras.utils.Sequence):
    def __init__(self, image_path, mask_path, batch_size, shuffle=True):
        self.image_path = image_path
        self.mask_path = mask_path

        self.batch_size = batch_size
        self.shuffle = shuffle
        
        self.check_data()
        self.on_epoch_end()
        
    def __len__(self):
        return int(len(self.image_paths) / self.batch_size)


    def __getitem__(self, index):
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        image_paths_temp = [self.image_paths[k] for k in indexes]
        mask_paths_temp = [self.mask_paths[k] for k in indexes]


        X, y = self.__data_generation(image_paths_temp, mask_paths_temp)
        return X, y


    def on_epoch_end(self):
        self.indexes = np.arange(len(self.image_paths))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, image_paths_temp, mask_paths_temp):
        X = np.empty((self.batch_size, 512, 512, 3), dtype=np.float32)
        y = np.empty((self.batch_size, 512, 512, 1), dtype=np.float32)

        for i, (image_path, mask_path) in enumerate(zip(image_paths_temp, mask_paths_temp)):
            img = tf.keras.preprocessing.image.load_img(image_path, target_size=(512, 512), color_mode="rgb")
            mask = tf.keras.preprocessing.image.load_img(mask_path, target_size=(512, 512), color_mode="grayscale")

            img = tf.keras.preprocessing.image.img_to_array(img)
            mask = tf.keras.preprocessing.image.img_to_array(mask)

            mask[mask > 0] = 1

            X[i,] = img.astype(np.float32) / 255.
            y[i,] = mask.astype(np.float32)

        X = tf.convert_to_tensor(X, dtype=tf.float32)
        y = tf.convert_to_tensor(y, dtype=tf.float32)
        return X, y
            

    def get_all_file_path_from_folder(self, root_dir):
        file_paths = []
        for root, dirs, files in os.walk(root_dir):
            for file in files:
                file_paths.append(os.path.join(root, file))
        return sorted(file_paths)

    def check_data(self):
        self.image_paths = self.get_all_file_path_from_folder(self.image_path)
        self.mask_paths = self.get_all_file_path_from_folder(self.mask_path)

        if len(self.image_paths) != len(self.mask_paths):
            raise ValueError("Number of images and masks are not equal")
        else:
            print("Check passed")
            print("Data size: ", len(self.image_paths))

if __name__ == "__main__":
    import cv2

    image_folder = "/home/user/Documents/tensorflow_ws/data/peoples/PNGImages"
    mask_folder = "/home/user/Documents/tensorflow_ws/data/peoples/PedMasksOld"

    dataloader = SegmentationDataloader(image_folder, mask_folder, 5, shuffle=True)
    imgs, msks = dataloader.__getitem__(0)

    for index in range(len(imgs)):
        img = imgs[index]
        mask = msks[index]

        cv2.imshow("img", img)
        cv2.imshow("mask", mask * 255)

        cv2.waitKey(0)


    
    
    