import h5py
import numpy as np

def load_data(train_dataset_path, test_dataset_path):
    train_dataset = h5py.File(train_dataset_path,"r")
    train_set_x = np.array(train_dataset["train_set_x"][:])
    train_set_y = np.array(train_dataset["train_set_y"][:])
    test_dataset = h5py.File(test_dataset_path,"r")
    test_set_x = np.array(test_dataset["test_set_x"][:])
    test_set_y = np.array(test_dataset["test_set_y"][:])

    train_set_y = train_set_y.reshape(1, train_set_y.shape[0])
    test_set_y = test_set_y.reshape(1, test_set_y.shape[0])

    return train_set_x, train_set_y, test_set_x, test_set_y

def image_to_column_vector(image: np.ndarray):
    column_vector_image = image.reshape(image.shape[0], -1).T
    return column_vector_image