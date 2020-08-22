import cv2
import numpy as np

from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer, ColorMode, GenericMask
from detectron2.utils.logger import setup_logger


class ObjectDetector:
    def __init__(self, cfg, logging=True):
        """
        Constructor for the ObjectDetector class, initialising configuration and logging

        Args:
            cfg (CfgNode): Detectron2 config file
            logging (bool): Enables/disables logging
        """
        if logging:
            setup_logger(name="fvcore")
            self.logger = setup_logger()

        self.cpu_device = "cpu"
        self.instance_mode = ColorMode.IMAGE
        self.predictor = DefaultPredictor(cfg)
        self.logging = logging
        self.classes = None
        self.metadata = None
        self.predictions = None
        self.image = None

    def register_metadata(self, classes, name="custom_train"):
        """
        Saves an annotations file to the metadata catalog and assigns a list of classes

        Args:
            classes (list): List of class labels                    TODO: Can this be derived from the coco json file?
            name (string): Name of the new metadata catalog entry
        """
        self.classes = classes
        MetadataCatalog.get(name).set(evaluator_type="coco", **{})
        self.metadata = MetadataCatalog.get(name)
        self.metadata.thing_classes = classes

    def get_pred_dict(self, image):
        """
        Gets predicted classes for an image. Predictions are output in a dictionary
        and grouped by class, i.e. class label as dict key

        Args:
            image (np.array): an image of shape (H, W, C) (in BGR order).

        Returns:
            output (dict): Predictions in the format {class1: [bbox1, bbox2, bbox3]}
        """
        ret = {}
        assert self.metadata, "No metadata found, please run register_metadata."
        self.image = image

        self.predictions = self.predictor(image)

        boxes = self.predictions['instances'].pred_boxes.tensor.cpu().numpy().tolist()
        classes = self.predictions['instances'].pred_classes.cpu().numpy().tolist()
        scores = self.predictions['instances'].scores.cpu().numpy().tolist()

        masks = np.asarray(self.predictions['instances'].pred_masks)
        height = image.shape[0]
        width = image.shape[1]

        masks = self._bitmask_to_polygons(masks, width, height)

        for index, cls in enumerate(classes):
            # Retrieve textual label
            label = self.metadata.thing_classes[cls]

            # Package result into a dictionary
            pack = {
                "score": scores[index],
                "bbox": boxes[index],
                "mask": masks[index]
            }

            # Add objects as instances of each label
            if label in ret.keys():
                ret[label].append(pack)
            else:
                ret[label] = [pack]

        return ret

    def get_pred_list(self, image):
        """
        Gets predicted classes for an image. Predictions are output in a list
        where each item is a dictionary containing the class label and bbox coords

        Args:
            image (np.array): an image of shape (H, W, C) (in BGR order).

        Returns:
            output (list): Predictions in the format [{class: class1, bbox: x1, y1, x2, y2}]
        """
        ret = []
        assert self.metadata, "No metadata found, please run register_metadata."
        self.image = image

        # Get predictions for the image
        self.predictions = self.predictor(image)

        # Decompose prediction object
        boxes = self.predictions['instances'].pred_boxes.tensor.cpu().numpy().tolist()
        classes = self.predictions['instances'].pred_classes.cpu().numpy().tolist()
        scores = self.predictions['instances'].scores.cpu().numpy().tolist()
        masks = np.asarray(self.predictions['instances'].pred_masks)

        height = image.shape[0]
        width = image.shape[1]

        # Convert the masks to polygon masks
        masks = self._bitmask_to_polygons(masks, width, height)

        for index, cls in enumerate(classes):
            # Retrieve textual label
            label = self.metadata.thing_classes[cls]

            # Package result into a dictionary
            pack = {
                "object_score": scores[index],
                "object_class": label,
                "bbox": boxes[index],
                "mask": masks[index]
            }

            # Add object to list
            ret.append(pack)
        return ret

    def display_result(self):
        """
        Used to visualise the predicted classes and masks on the image

        """
        vis_output = self._visualise_predictions()

        cv2.namedWindow("Detection result", cv2.WINDOW_NORMAL)
        cv2.imshow("Detection result", vis_output.get_image()[:, :, ::-1])

        if cv2.waitKey(0) == 32:
            return

    def save_result(self, path):
        """
        Saves the visualised result to a file

        Args:
            path (string): The full path to save the file (inc. filename and type extension)

        Returns:
            Bool: Boolean indicating whether the image saved successful
        """
        vis_output = self._visualise_predictions()

        img = vis_output.get_image()[:, :, ::-1]

        try:
            cv2.imwrite(path, img)
        except Exception:
            return False

        return True

    def _visualise_predictions(self):
        """
        Draws the predictions and masks onto the input image

        Returns:
            VisImage: A detectron2 object containing the image with drawn predictions
        """
        assert self.metadata, "No metadata found, please run register_metadata."
        assert self.predictions, "No predictions found, please run get_predictions."

        image = self.image[:, :, ::-1]
        visualizer = Visualizer(image, self.metadata, instance_mode=self.instance_mode)
        ret = visualizer.draw_instance_predictions(self.predictions["instances"].to("cpu"))

        return ret

    def _bitmask_to_polygons(self, masks, width, height):
        """
        Converts masks from bitmasks to polygon masks using built in detectron2
        object types.

        Args:
            masks (np.ndarray): An array of bitmasks
            width (int): The width of the image (in px)
            height (int): The height of the image (in px)

        Returns:
            list: A list polygon masks
        """
        masks = [GenericMask(x, height, width) for x in masks]
        temp_mask = []
        ret = []

        # Convert mask to a generic mask if not already
        for x in masks:
            if isinstance(x, GenericMask):
                temp_mask.append(x)
            else:
                temp_mask.append(GenericMask(x, height, width))

        # Get the polygon object and reshape from a 1x1 to a 1x2 array
        for mask in temp_mask:
            mask_poly = mask.polygons[0]
            mask_poly = mask_poly.reshape(-1, 2).tolist()
            ret.append(mask_poly)

        return ret
