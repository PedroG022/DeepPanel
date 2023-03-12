import silence_tensorflow.auto
import tensorflow as tf
from tensorflow import keras
import os

from metrics import iou_coef, dice_coef, border_acc, background_acc, content_acc
from utils import count_files_in_folder, labeled_prediction_to_image, map_prediction_to_mask, IMAGE_SIZE

INPUT_PATH = "./dataset/test/raw/"
OUTPUT_PATH = "./output/"


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


def predict(model, dataset):
    TESTING_BATCH_SIZE = count_files_in_folder(INPUT_PATH)
    test = dataset.map(load_image_test)
    test_dataset = test.batch(TESTING_BATCH_SIZE)
    predictions = model.predict(test_dataset)

    return predictions


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


def write_output(labeled_predictions):
    predicted_index = 0

    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)

    for predicted_result in labeled_predictions:
        prediction_as_image = labeled_prediction_to_image(predicted_result)
        prediction_as_image.save(f"{OUTPUT_PATH}{predicted_index:03d}.jpg")
        prediction_as_image.close()
        print(f"    - Image with index {predicted_index} saved.")
        predicted_index += 1


def label_predictions(predictions):
    predicted_images_number = len(predictions)

    labeled_predictions = []

    for image_index in range(predicted_images_number):
        prediction = predictions[image_index]
        predicted_mask = map_prediction_to_mask(prediction)
        labeled_predictions.append(predicted_mask)

    return labeled_predictions


if __name__ == "__main__":
    print(" - Loading saved model and dataset")

    model = load_model("./model")
    dataset = load_images_from_folder(INPUT_PATH, shuffle=False)

    print(f" - Test data loaded for {len(dataset)} images")
    print(" - Prediction started")

    predictions = predict(model, dataset)

    print(f" - Prediction finished for {len(predictions)} images")
    print(f" - Transforming predictions into labeled values.")

    labeled_predictions = label_predictions(predictions)

    print(f" - Saving labeled images into {OUTPUT_PATH} folder")
    write_output(labeled_predictions)
