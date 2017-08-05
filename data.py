import numpy as np
import skimage.transform as sktransform
import random
import matplotlib.image as mpimg
import os


cameras            = ['left', 'center', 'right']
cameras_correction = [0.25, 0.0, -0.25]
default_top_ratio  = 0.375
default_bot_tatio  = 0.125

img_w = 128
img_h = 32
img_c = 3

def preprocess(image, top_ratio=default_top_ratio, bottom_ratio=0.125):
    """
    Image preprosessing:
    - Crops image bottom and top
    - Normalizes pixel data
    - resizes images 32x128
    """
    top    = int(top_ratio * image.shape[0])
    bottom = int(bottom_ratio * image.shape[0])
    image  = sktransform.resize(image[top:-bottom, :], (img_h, img_w, img_c))
    return image

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
            # Yieleded arrays
            x = np.empty([0, img_h, img_w, img_c], dtype=np.float32)
            y = np.empty([0], dtype=np.float32)
            # Read in and preprocess a batch of images
            for i in batch_i:
                # Randomly select camera
                camera = np.random.randint(len(cameras))

                # Read frame image and work out steering angle
                image_path = os.path.join(root_path, data[cameras[camera]].values[i].strip())
                image      = mpimg.imread(image_path)
                angle      = data.steering.values[i] + cameras_correction[camera]

                # Add random shadow as a vertical slice of image
                h, w     = image.shape[0], image.shape[1]
                [x1, x2] = np.random.choice(w, 2, replace=False)
                k        = h / (x2 - x1)
                b        = - k * x1
                for i in range(h):
                    c = int((i - b) / k)
                    image[i, :c, :] = (image[i, :c, :] * .5).astype(np.int32)

                # Randomly shift up and down while preprocessing
                delta          = .05
                rand_top_ratio = random.uniform(default_top_ratio - delta,
                                                default_top_ratio + delta)
                rand_bot_ratio = random.uniform(default_bot_tatio - delta,
                                                default_bot_tatio + delta)

                image = preprocess(image, top_ratio=rand_top_ratio, bottom_ratio=rand_bot_ratio)

                # Append to batch
                x = np.append(x, [image], axis=0)
                y = np.append(y, [angle])

            # Randomly flip half of images in the batch
            flip_i    = random.sample(range(x.shape[0]), int(x.shape[0] / 2))
            x[flip_i] = x[flip_i, :, ::-1, :]
            y[flip_i] = -y[flip_i]

            # Yield data
            yield (x, y)

