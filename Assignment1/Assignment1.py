import pickle
import numpy as np
import matplotlib.pyplot as plt


def display_cifar_10_examples(
    cifar_dir: str = "./Datasets/cifar-10-batches-py", ni: int = 5
) -> None:
    # Load a batch of training data
    with open(cifar_dir + "/data_batch_1", "rb") as fo:
        dict = pickle.load(fo, encoding="bytes")
    # Extract the image data and cast to float from the dict dictionary
    X = dict[b"data"].astype(np.float64) / 255.0
    X = X.transpose()
    nn = X.shape[1]
    # Reshape each image from a column vector to a 3d array
    X_im = X.reshape((32, 32, 3, nn), order="F")
    X_im = np.transpose(X_im, (1, 0, 2, 3))
    # Display the first 5 images
    fig, axs = plt.subplots(1, 5, figsize=(10, 5))
    for i in range(ni):
        axs[i].imshow(X_im[:, :, :, i])
        axs[i].axis("off")
    plt.show()


def load_batch(filename: str) -> tuple:
    """
    Retrieves the dataset and converts it to torch tensor

    Args:
        filename (str): File path to the Dataset
    Returns:
        X (torch.tensor): Tensor of dimensions
        Y (torch.tensor): Tensor of dimensions
        y (torch.tensor): Tensor of dimensions
    """
    X = 0
    Y = 0
    y = 0

    return X, Y, y


if __name__ == "__main__":
    # Visualizing dataset
    display_cifar_10_examples()
