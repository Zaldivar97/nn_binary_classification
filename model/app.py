from model.data import data_loader
from model.core.entrypoint import nn_model


def run(train_dataset_path, test_dataset_path):
    train_set_x, train_set_y, test_set_x, test_set_y = data_loader.load_data(
                                                train_dataset_path, test_dataset_path)

    train_x_transformed = data_loader.image_to_column_vector(train_set_x)
    test_x_transformed = data_loader.image_to_column_vector(test_set_x)

    layer_dims = [train_x_transformed.shape[0], 15, 6, 4, 1]
    trained_parameters = nn_model(train_x_transformed, train_set_y, layer_dims, epocs=5000)
    print("DONE")