from keras.models import load_model
import numpy as np

np.set_printoptions(threshold=np.nan)
model = load_model("MNIST_model.h5")


def predict(test_array):
    """
    Predicts the number from the Bitmap
    :param test_array: Bitmap of hand written number (Normalized)
    :return: Probability in array
    """
    return model.predict(test_array)


#if __name__ == "__main__":
    #pass
