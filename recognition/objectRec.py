import itertools
import json
import time
import string
import cv2
import numpy as np
from Levenshtein import distance
from shapely.geometry.polygon import Polygon
from shapely.errors import TopologicalError
from copy import deepcopy
from detectron2.config import get_cfg

from detectors.objectDetect import ObjectDetector
from detectors.textDetect import TextDetector
from detectron2.utils.logger import setup_logger

CONFIG_FILE = "resources/mask_rcnn_X_101_32x8d_FPN_3x.yaml"
WEIGHTS = "resources/model_final.pth"
CONFIDENCE_THRESHOLD = 0.4
DICTIONARY_FILE = "resources/keywords.json"
DEVICE = "cpu"

CLASSES = ["cereal",
           "condiment",
           "bread",
           "pasta",
           "milk",
           "egg",
           "tinned tomatoes",
           "butter",
           "baked beans",
           "soup"]


class ObjectInference:
    def __init__(self):
        """
        Constructor for ExpiryDetector class
        """
        num_classes = len(CLASSES)
        self.logger = setup_logger(name="recognition")

        cfg = self._setup_config(num_classes, CONFIG_FILE, WEIGHTS, CONFIDENCE_THRESHOLD, DEVICE)
        self.logger.info(f"Setting up config file \n {cfg}")

        self.object_detector = ObjectDetector(cfg)
        self.text_detector = TextDetector()

        self.object_detector.register_metadata(CLASSES)

        self.logger.info("Loading dictionary file")
        with open(DICTIONARY_FILE) as file:
            self.keywords = json.load(file)

    @staticmethod
    def _setup_config(num_classes, config_file, weights, confidence_thres, device):
        """

        Args:
            num_classes:
            config_file:
            weights:
            confidence_thres:
            device:

        Returns:

        """
        # load config from file and constants
        cfg = get_cfg()
        cfg.merge_from_file(config_file)
        cfg.MODEL.WEIGHTS = weights
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
        cfg.MODEL.DEVICE = device

        # Set score_threshold for builtin models
        cfg.MODEL.RETINANET.SCORE_THRESH_TEST = confidence_thres
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = confidence_thres
        cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = confidence_thres
        cfg.freeze()
        return cfg

    def run_inference(self, image_bytes):
        """

        Args:
            image_bytes:

        Returns:

        """
        np_arr = np.frombuffer(image_bytes, np.uint8)
        image_np = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        # # Convert to bgr np array
        # image_rgb = np.array(Image.open(image_file).convert('RGB'))
        # image_np = image_rgb[..., ::-1].copy()

        self.logger.info("Running detectron2 recognition...")
        t_start = time.perf_counter()
        objects = self.object_detector.get_pred_list(image_np)
        t_finish = time.perf_counter()
        obj_t = t_finish - t_start
        self.logger.info(f"Inference time: {obj_t:0.4f} seconds")

        self.logger.info("Calling google vision API...")
        t_start = time.perf_counter()
        texts = self.text_detector.get_text(image_bytes)

        # Locate detected text inside each detected object
        self.logger.info("Analysing textual content")

        # Reduce detections of the same object to highest scoring prediction
        reduced = self._reduce_multi_pred(objects)
        data = self._analyse_text(reduced, texts)
        t_finish = time.perf_counter()
        text_t = t_finish - t_start
        self.logger.info(f"Lexical analysis time: {text_t:0.4f} seconds")

        return data

    def _analyse_text(self, detected_objects, detected_texts):
        """
        Localises text within detection masks and generates a class prediction

        Args:
            detected_objects (dict):
            detected_texts (dict):

        Returns:

        """
        ret = []

        for detected_object in detected_objects:
            pack = detected_object.copy()
            del pack['mask']

            detected_words = self._locate_words(detected_object['mask'], detected_texts)

            cleaned_words = []
            for detected_word in detected_words:
                if detected_word.isalpha():
                    cleaned_words.append(detected_word)

                elif detected_word.isnumeric():
                    continue

                elif '-' in detected_word:
                    split_word = detected_word.split('-')

                    for word in split_word:
                        if word != '':
                            cleaned_words.append(word)
                else:
                    word = ""

                    for char in detected_word:
                        if char not in string.punctuation:
                            word += char
                    cleaned_words.append(word)

            pack['matched_words'] = self._match_in_dict(cleaned_words, self.keywords)

            textual_pred = self._textual_prediction(pack['matched_words'])

            pack['text_class'] = textual_pred['class']
            pack['text_score'] = textual_pred['score']
            pack['text_distance'] = textual_pred['distance']

            ret.append(pack)
        return ret

    @staticmethod
    def get_largest_item(data):
        """
        Returns the detection with the largest area

        Args:
            data list(dict): List of detections (result from recognition)

        Returns:
            dict: Detection with the highest area
        """
        if len(data) > 1:
            largest_area = 0
            largest_item = None

            for item in data:
                x1 = item['bbox'][0]
                y1 = item['bbox'][1]
                x2 = item['bbox'][2]
                y2 = item['bbox'][3]

                area = Polygon([[x1, y1], [x1, y2], [x2, y2], [x2, y1]]).area

                if area > largest_area:
                    largest_area = area
                    largest_item = item
            return largest_item
        elif len(data) == 1:
            return data[0]
        else:
            return None

    @staticmethod
    def _locate_words(obj_mask, texts):
        """
        Find which textual items are located inside a given mask

        Args:
            obj_mask (list): A list of x, y coordinates, dictating the location of a mask
            texts (dict):

        Returns (list): All text contained inside the bounds of a mask

        """
        ret = []
        item_poly = Polygon(obj_mask)

        for text in texts:
            text_poly = Polygon(texts[text])

            if item_poly.contains(text_poly):
                ret.append(text)
        return ret

    @staticmethod
    def _reduce_multi_pred(predictions):
        """

        Args:
            predictions:

        Returns:

        """
        reduce = predictions.copy()

        for ((index_a, a), (index_b, b)) in itertools.combinations(enumerate(predictions), 2):
            a_polygon = Polygon(a['mask'])
            b_polygon = Polygon(b['mask'])

            try:
                intersection = a_polygon.intersection(b_polygon).area
            except TopologicalError:
                continue

            smallest = min(a_polygon.area, b_polygon.area)
            overlap = intersection / smallest

            if overlap > 0.8:
                if a['object_score'] > b['object_score']:
                    reduce[index_b] = None
                else:
                    reduce[index_a] = None

        ret = [x for x in reduce if x]
        return ret

    @staticmethod
    def _match_in_dict(words, keywords):
        """
        Searches for word matches in a keyword dictionary

        Args:
            words (list): A list of textual items found inside a detection mask
            keywords (dict): Keyword dictionary in the format {"class": ["word1", "word2"]}

        Returns (dict): A data structure containing a list of matched words and

        """
        matches = {}

        for word in words:
            for category, items in keywords.items():
                for item in items:
                    if distance(item, word) <= 1:
                        try:
                            matches[category].append(item)
                        except KeyError:
                            matches[category] = [item]

        return matches

    @staticmethod
    def _textual_prediction(matched_text):
        """


        Args:
            matched_text:

        Returns:

        """
        highest_score = {
            "score": 0,
            "class": None,
            "matches": 0,
            "distance": 0
        }

        # If no matched words then return dict
        if not matched_text:
            return highest_score

        # Count matches for each category
        num_matches = {key: len(values) for key, values in matched_text.items()}
        # Count total number of matches across all categories
        total_matches = sum(num_matches.values())

        # Get the highest matching category and corresponding match count
        first = max(num_matches.items(), key=lambda k: k[1])
        first = list(first)

        # Calculate distance from second matching category
        if len(num_matches) > 1:
            del num_matches[first[0]]
            second = max(num_matches.items(), key=lambda k: k[1])
            dist = first[1] - second[1]
        else:
            dist = first[1]

        highest_score['distance'] = dist

        highest_score['class'] = first[0]
        highest_score['score'] = first[1] / total_matches

        highest_score['total'] = total_matches
        highest_score['matches'] = matched_text[first[0]]

        return highest_score

    @staticmethod
    def final_predictions(data):
        """

        Args:
            data:

        Returns:

        """
        prediction_data = deepcopy(data)

        for prediction in prediction_data:
            object_score = prediction['object_score']
            text_score = prediction['text_score']
            text_dist = prediction['text_distance']

            object_class = prediction['object_class']
            text_class = prediction['text_class']

            if object_score > 0:
                if text_dist > 2:
                    prediction['predicted_class'] = text_class
                    prediction['predicted_score'] = text_score
                else:
                    prediction['predicted_class'] = object_class
                    prediction['predicted_score'] = object_score

            elif (object_score <= 0) and text_score:
                prediction['predicted_class'] = text_class
                prediction['predicted_score'] = text_score

            else:
                prediction['predicted_class'] = None
                prediction['predicted_score'] = 0

        return prediction_data
