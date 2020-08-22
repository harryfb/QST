import io
from google.cloud import vision


class TextDetector:
    def __init__(self):
        """
        Constructor for TextDetector class.
        """
        self.client = vision.ImageAnnotatorClient()

    def get_text(self, content):
        """
        Sends an image to the Google Vision API for OCR

        Args:
            content (bytes): An image stored as a bytes object

        Returns:
            list (dict): Returns detected words in a list of
            dictionaries formatted as {text: [bbox]}
        """
        image = vision.types.Image(content=content)
        response = self.client.text_detection(image=image)
        texts = response.text_annotations
        results = {}

        if response.error.message:
            raise Exception('Error encounterd when calling Google Vision API'
                            .format(response.error.message))

        # Convert the results object into dictionary format
        for index, text in enumerate(texts):
            if index == 0:
                continue

            bbox = []
            for point in text.bounding_poly.vertices:
                bbox.append((point.x, point.y))

            results[text.description.lower()] = bbox
        return results


if __name__ == '__main__':
    textDetector = TextDetector()

    path = '../images/test1.jpg'

    with io.open(path, 'rb') as image_file:
        image = image_file.read()

    print(textDetector.get_text(image).keys())

