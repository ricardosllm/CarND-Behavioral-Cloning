import os
import random
import numpy as np
import matplotlib as plt
import skimage.transform as sktransform


cameras            = ['left', 'center', 'right']
cameras_correction = [0.25, 0.0, -0.25]
default_top_ratio  = 0.375
default_bot_tatio  = 0.125

img_w = 128
img_h = 32
img_c = 3


def preprocess(image, top_ratio=default_top_ratio, bot_ratio=default_bot_tatio):
    """
    Image preprosessing:
    - Crops image bottom and top
    - Normalizes pixel data
    - resizes images 32x128
    """
    top    = int(top_ratio * image.shape[0])
    bottom = int(bot_ratio * image.shape[0])
    image  = sktransform.resize(image[top:-bottom, :], (img_h, img_w, img_c))
    return image


def crop_image(image):
    """
    Randomly shift up and down while preprocessing
    """
    delta          = .05
    rand_top_ratio = random.uniform(default_top_ratio - delta,
                                    default_top_ratio + delta)
    rand_bot_ratio = random.uniform(default_bot_tatio - delta,
                                    default_bot_tatio + delta)
    image = preprocess(image, top_ratio=rand_top_ratio, bot_ratio=rand_bot_ratio)

    return image


def add_random_shadow(image):
    """
    Add random shadow as a vertical slice of image
    """
    h, w     = image.shape[0], image.shape[1]
    [x1, x2] = np.random.choice(w, 2, replace=False)
    k        = h / (x2 - x1)
    b        = - k * x1
    for i in range(h):
        c = int((i - b) / k)
        image[i, :c, :] = (image[i, :c, :] * .5).astype(np.int32)

    return image


def read_image_and_angle(root_path, data, camera, index):
    """
    Read frame image and work out steering angle
    """
    image_path = os.path.join(root_path,
                              data[cameras[camera]].values[index].strip())
    image      = plt.image.imread(image_path)
    angle      = data.steering.values[index] + cameras_correction[camera]

    return image, angle


def augment_data(x, y, data, root_path, batch_i):
    """
    Read in and preprocess a batch of images
    """
    for i in batch_i:
        # Randomly select camera
        camera       = np.random.randint(len(cameras))
        image, angle = read_image_and_angle(root_path, data, camera, i)

        image = add_random_shadow(image)
        image = crop_image(image)

        # Append to batch
        x = np.append(x, [image], axis=0)
        y = np.append(y, [angle])

    return x, y


def flip_images(x, y):
    """
    Randomly flip half of images in the batch
    """
    flip_i    = random.sample(range(x.shape[0]), int(x.shape[0] / 2))
    x[flip_i] = x[flip_i, :, ::-1, :]
    y[flip_i] = -y[flip_i]

    return x, y


def generate_samples(data, root_path, batch_size=128):
    """
    Returns a Keras generator with the batches of data
    Applies data augmentation
    """
    while True:
        # Generate random batch of indices
        indices = np.random.permutation(data.count()[0])

        for batch in range(0, len(indices), batch_size):
            batch_i = indices[batch:(batch + batch_size)]

            x = np.empty([0, img_h, img_w, img_c], dtype=np.float32)
            y = np.empty([0], dtype=np.float32)

            x, y = augment_data(x, y, data, root_path, batch_i)
            x, y = flip_images(x, y)

            yield (x, y)
