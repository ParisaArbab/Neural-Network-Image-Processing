import numpy as np
import PIL.Image
import tensorflow as tf
import matplotlib.pyplot as plt

def download_and_extract_model(model_url, data_dir):
    local_zip_file = tf.keras.utils.get_file(fname=model_url.split('/')[-1],
                                             origin=model_url, 
                                             extract=True, 
                                             cache_dir=data_dir, 
                                             cache_subdir='models')
    return local_zip_file

def load_model():
    # Load a pre-trained model, such as InceptionV3
    model = tf.keras.applications.InceptionV3(include_top=False, weights='imagenet')
    return model

def deepdream(model, img, steps, step_size):
    # Preprocess the image
    img = tf.keras.applications.inception_v3.preprocess_input(img)

    for step in range(steps):
        with tf.GradientTape() as tape:
            tape.watch(img)
            activations = model(img)
            loss = tf.reduce_mean(activations)

        gradients = tape.gradient(loss, img)
        gradients /= tf.math.reduce_std(gradients) + 1e-8
        img = img + gradients * step_size
        img = tf.clip_by_value(img, -1, 1)

    return img

def main():
    # Step 1 - Download and extract the model (assuming this is handled by a separate function)
    model_url = 'https://storage.googleapis.com/download.tensorflow.org/models/inception5h.zip'
    data_dir = './'
    download_and_extract_model(model_url, data_dir)

    # Step 2 - Load the model (assuming this is handled by a separate function)
    model = load_model()

    # Step 3 - DeepDream
    # Load your image
    image_path = 'C:/Career/HonoluluForest.jpeg'
    original_img = PIL.Image.open(image_path)
    original_img = np.array(original_img)

    # Convert to float32 numpy array, scale to range [0, 1], and add a batch dimension
    img = np.array(original_img).astype(np.float32) / 255.0
    img_tensor = tf.convert_to_tensor(img, dtype=tf.float32)
    img_tensor = img_tensor[tf.newaxis, ...]  # Add a batch dimension

    # Apply DeepDream
    steps = 100  # Number of gradient ascent steps
    step_size = 0.01  # Step size for each ascent
    dreamed_img_tensor = deepdream(model, img_tensor, steps, step_size)

    # Convert back to image and show
    dreamed_img = np.squeeze(dreamed_img_tensor, axis=0)  # Remove batch dimension
    dreamed_img = ((dreamed_img * 0.5) + 0.5) * 255  # Rescale to [0, 255]
    dreamed_img = np.array(dreamed_img).astype(np.uint8)
    plt.imshow(dreamed_img)
    plt.show()

if __name__ == '__main__':
    main()