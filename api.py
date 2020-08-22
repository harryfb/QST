from flask import Flask, request
import json

from recognition.objectRec import ObjectInference
from recognition.expiryRec import ExpiryInference


app = Flask(__name__)


@app.route('/object-detection', methods=['POST'])
def object_detection():
    """
    POST ENDPOINT: Runs the object recognition algorithm
    on a given image. Can output predictions for one or
    more objects per image.

    Returns:
        json: A structure containing analysis of the image
    """
    assert request.path == '/object-detection'
    assert request.method == 'POST'

    objectInference = ObjectInference()
    image_bytes = request.files['image'].read()

    predictions = objectInference.run_inference(image_bytes)
    predictions = objectInference.final_predictions(predictions)

    return json.dumps(objectInference.final_predictions(predictions))


@app.route('/expiry-detection', methods=['POST'])
def expiry_detection():
    """
    POST ENDPOINT: Runs the expiry date detection algorithm
    on a given image. Limited to a single object

    Returns:
        json: A structure containing analysis of the image
    """
    assert request.path == '/expiry-detection'
    assert request.method == 'POST'
    prediction = {}

    image_bytes = request.files['image'].read()

    expiryInference = ExpiryInference()

    datetime = expiryInference.run_inference(None, image_bytes)
    datetime = str(datetime)

    if datetime:
        prediction['expiryDate'] = datetime
    else:
        prediction['expiryDate'] = None

    return json.dumps(prediction)


@app.route('/detection', methods=['POST'])
def detection():
    """
    POST ENDPOINT: Runs the object recognition & expiry date detection
    algorithms on a single given image. Largest detected bounding box
    chosen for expiry detection due to single object limitation

    Returns:
        json: A structure containing analysis of the image
    """
    assert request.path == '/detection'
    assert request.method == 'POST'

    inference = ObjectInference()
    expiryInference = ExpiryInference()

    image_bytes = request.files['image'].read()

    predictions = inference.run_inference(image_bytes)
    predictions = inference.final_predictions(predictions)

    prediction = inference.get_largest_item(predictions)

    if prediction:
        predicted_class = prediction['predicted_class']
    else:
        predicted_class = None

    datetime = expiryInference.run_inference(predicted_class, image_bytes)
    datetime = str(datetime)

    if datetime:
        prediction['expiry_date'] = datetime
    else:
        prediction['expiry_date'] = None

    return json.dumps(prediction)


if __name__ == '__main__':
    app.run()

    # from glob import glob
    # import csv
    # import time
    # import io
    #
    # objectInference = ObjectInference()
    # expiryInference = ExpiryInference(expiry_days)
    #
    # with open('../expiry_detector_outputs/expiry_h/results.csv', 'w', newline='') as csvfile:
    #     result_writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    #     result_writer.writerow(
    #         ['Filename', 'Class', 'Number', 'Inference_time', 'Detection'])
    #
    # paths = glob('../expiry_detector_outputs/expiry_h/*.JPG')
    #
    # for index, path in enumerate(paths):
    #     name = path.split('/').pop()
    #     cat = name.split('_')[0]
    #     num = name.split('_').pop().split('.')[0]
    #
    #     print(f"[{index + 1}/{len(paths)}] {int(index * 100 / len(paths))}%  Analysing {name}...")
    #
    #     image_file = io.open(path, 'rb')
    #     image_bytes = image_file.read()
    #
    #     t_start = time.perf_counter()
    #
    #     predictions = objectInference.run_inference(image_bytes)
    #     predictions = objectInference.final_predictions(predictions)
    #     prediction = objectInference.get_largest_item(predictions)
    #
    #     if prediction:
    #         predicted_class = prediction['predicted_class']
    #     else:
    #         predicted_class = None
    #
    #     date = expiryInference.run_inference(predicted_class, image_bytes)
    #     print(date)
    #
    #     t_finish = time.perf_counter()
    #     t = t_finish - t_start
    #
    #     print(f"Writing to file...")
    #
    #     with open('../expiry_detector_outputs/expiry_h/results.csv', 'a', newline='') as csvfile:
    #         result_writer = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    #         result_writer.writerow([name, cat, num, t, date])
