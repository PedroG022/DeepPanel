import silence_tensorflow.auto 
import tensorflow as tf
import numpy as np
import cv2
import os

from utils import count_files_in_folder, files_in_folder, labeled_prediction_to_image, map_prediction_to_mask, IMAGE_SIZE
from metrics import iou_coef, dice_coef, border_acc, background_acc, content_acc
from DeepPanelExtractor import extract_panels
from tensorflow import keras

# ===========================
# Modified methods
# ===========================

def parse_image(img_path):
    image = tf.io.read_file(img_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.convert_image_dtype(image, tf.uint8)

    return image

def load_images_from_folder(folder, shuffle=True):
    files = tf.data.Dataset.list_files(folder + "*.jpg", shuffle=shuffle)
    return files.map(parse_image)

def normalize(input_image):
    input_image = tf.cast(input_image, tf.float32) / 255.0
    return input_image

def load_image_test(datapoint):
    input_image = tf.image.resize_with_pad(
        datapoint, target_height=IMAGE_SIZE, target_width=IMAGE_SIZE)
    input_image = normalize(input_image)

    return input_image

# ===========================
# "Generic" methods
# =========================== 

def load_dataset(processed_images, input_path):
    TESTING_BATCH_SIZE = count_files_in_folder(input_path)
    test = processed_images.map(load_image_test)
    test_dataset = test.batch(TESTING_BATCH_SIZE)
    return test_dataset

def write_mask_output(labeled_predictions, output_path):
    predicted_index = 0

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for predicted_result in labeled_predictions:
        prediction_as_image = labeled_prediction_to_image(predicted_result)
        prediction_as_image.save(f"{output_path}{predicted_index:03d}.jpg")
        prediction_as_image.close()
        print(f"    - Image with index {predicted_index} saved.")
        predicted_index += 1

def label_predictions(predictions):
    predicted_images_number = len(predictions)

    labeled_predictions = []

    for image_index in range(predicted_images_number):
        prediction = predictions[image_index]
        prediction = np.squeeze(prediction)
        predicted_mask = map_prediction_to_mask(prediction)
        labeled_predictions.append(predicted_mask)

    return labeled_predictions

def make_labels_and_output(predictions, output_path):
    print(f" - Prediction finished for {len(predictions)} images.")
    print(f" - Transforming predictions into labeled values...")

    labeled_predictions = label_predictions(predictions)

    print(f" - Saving mask images into {output_path} folder...")
    write_mask_output(labeled_predictions)

# ===========================
# Main methods
# =========================== 

def load_model(model_path):
    tf.random.set_seed(11)

    custom_objects = {
        "border_acc": border_acc,
        "background_acc": background_acc,
        "content_acc": content_acc,
        "iou_coef": iou_coef,
        "dice_coef": dice_coef
    }

    model = keras.models.load_model(model_path, custom_objects=custom_objects)
    return model

def predict(model, dataset):
    test_dataset = load_dataset(dataset)
    predictions = model.predict(test_dataset)

    return predictions

def process_default(processed_images, model, generate_masks, panels_output_folder):
    print(" - Loading saved model...")

    model = load_model(model)

    print(" - Prediction started...")

    predictions = predict(model, processed_images)

    if generate_masks:
        make_labels_and_output(predictions)

    cut_panels(predictions, images_set(), panels_output_folder)
    

# ===========================
# TFLite methods
# =========================== 

def load_tflite_model(model_path):
    # Load the TFLite model and allocate tensors.
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    return interpreter, input_details, output_details

def predict_tflite(interpreter, input_data, input_details, output_details, num_images):
    predictions = []

    for image_idx in range(0, num_images):
        img = np.expand_dims(input_data[image_idx, :, :, :], axis=0)
        interpreter.set_tensor(input_details[0]['index'], img)
        interpreter.invoke()

        output_data = interpreter.get_tensor(output_details[0]['index'])
        predictions.append(output_data)

    return predictions

def process_tflite(processed_images, model, generate_masks, panels_output_folder, input_folder):
    print(" - Loading saved model...")

    interpreter, input_details, output_details = load_tflite_model(model)
    test_dataset = load_dataset(processed_images, input_folder)

    for images in test_dataset:
        pass

    images = images.numpy()
    input_data = images
    num_images = images.shape[0]

    print(" - Prediction started...")

    predictions = predict_tflite(interpreter=interpreter, input_data=input_data,
                                 input_details=input_details, output_details=output_details, num_images=num_images)

    if generate_masks:
        make_labels_and_output(predictions)

    cut_panels(predictions, images_set(input_folder), panels_output_folder)

def images_set(input_path):
    image_paths = files_in_folder(input_path)

    images = {}

    for path in image_paths:
        image = cv2.imread(os.path.join(input_path, path))
        images[path] = image

    return images


def cut_panels(predictions, images, panels_output_path):
    for i, prediction in enumerate(predictions):
        current_key = [k for k in images.keys()][i]
        current_image = images[current_key]
        original_name = current_key[0: current_key.index('.')]
        
        print(f" - Working in '{current_key}'...")

        img_width = current_image.shape[1]
        img_height = current_image.shape[0]

        panels = extract_panels(prediction, original_width=img_width, original_height=img_height).panels

        print("    - Found %d panel(s)." % len(panels))

        for j, panel in enumerate(panels):
            if not os.path.exists(panels_output_path):
                os.makedirs(panels_output_path)

            name = f'./{panels_output_path}{original_name}-{j}.jpg'

            left = int(panel.left)
            width = int(panel.width)
            top = int(panel.top)
            height = int(panel.height)

            cropped_image = current_image[top:height, left:width]
            cv2.imwrite(name, cropped_image)

def processFolder(model, input_folder, panels_output_folder, masks_output_folder = False):
    print(f" - Loading and processing images for folder {os.path.dirname(input_folder)} ")

    processed_images = load_images_from_folder(input_folder, shuffle=False)

    if model == 0:
        process_tflite(processed_images, './model.tflite', masks_output_folder, panels_output_folder, input_folder)
    else:
        process_default(processed_images, './model', masks_output_folder, panels_output_folder)