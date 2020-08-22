import io
import csv
import time
import dateparser
from glob import glob

from detectors.textDetect import TextDetector
from detectors.expiryDetect import ExpiryDetector

years = 256
months = 31
expiry_days = {
    'milk': 1 * months,
    'egg': 1.5 * months,
    'cereal': 2 * years,
    'condiment': 4 * years,
    'bread': 2 * years,
    'butter': 6 * months,
    'tinned tomatoes': 5 * years,
    'soup': 5 * years,
    'baked beans': 5 * years,
    'pasta': 3 * years
}


class ExpiryInference:
    def __init__(self):
        """
        Constructor for ExpiryDetector class
        """
        self.expiryDetector = ExpiryDetector(30, 30)
        self.textDetector = TextDetector()

    def run_inference(self, path, image_bytes, predicted_class=None):
        """
        Top level inference returning expiry detection prediction

        Args:
            image_bytes (bytes): An input image
            predicted_class (string): Optional class to narrow down predictions
                                      valid predictions

        Returns:
            datetime: An expiration date prediction
        """
        exif_capture = self.expiryDetector.get_capture_date(path)

        # Run textual recognition and split any pre-process strings in list
        texts = self.textDetector.get_text(image_bytes)
        texts = self.expiryDetector.decompose_texts_list(texts)

        # Search list of words for month keyword matches and return indexes
        detected_months = self.expiryDetector.find_month(texts)

        # Month name/s were detected, attempt to decode each into a date
        if detected_months:
            month_first_proposals = self.expiryDetector.month_first_search(texts, detected_months)
        else:
            month_first_proposals = None

        # No successful matches so look for deliminated strings within list
        year_first_proposals = self.expiryDetector.year_first_search(texts)

        dates = self.reduce_candidates(month_first_proposals, year_first_proposals)

        # Make a single date prediction from the list of predictions
        if exif_capture:
            threshold = dateparser.parse(str(exif_capture))
        else:
            threshold = self.expiryDetector.current_date

        return self.make_prediction(dates, threshold, predicted_class)

    @staticmethod
    def reduce_candidates(month_first_proposals, year_first_proposals):
        """
        Reduces the two lists of proposals found by searching for month
        strings and valid years, to a single list of proposals

        Args:
            month_first_proposals (list): Proposals from a month-first search
            year_first_proposals (list): Proposals from a year-first search

        Returns:
            list: A list of proposals as datetime objects
        """
        # If both month and year first proposals, return any exact matches from
        # month first list, else choose year first list
        if month_first_proposals and year_first_proposals:
            ret = [item[0] for item in month_first_proposals if item[1] == 0]
            if ret:
                return ret
            else:
                return year_first_proposals
        elif month_first_proposals:
            ret = [item[0] for item in month_first_proposals]
        elif year_first_proposals:
            ret = year_first_proposals
        else:
            ret = []

        return ret

    @staticmethod
    def make_prediction(dates, date_threshold, predicted_class=None):
        """
        Makes a final prediction from a list of proposed expiration
        dates.

        Args:
            dates (list): A list of proposed date values as datetime objects
            date_threshold (datetime): Date that a result is valid from e.g.
                                       current date, image capture date
            predicted_class (str): Optional class to narrow down predictions
                                   valid predictions

        Returns:
            datetime: A single expiration date prediction
        """
        if dates and len(dates) > 1:
            if predicted_class:
                class_threshold = expiry_days[predicted_class]

                dates = [date for date in dates if (date - date_threshold).days <= class_threshold]
            else:
                dates = [date for date in dates if (date - date_threshold).days > 0]

            # Highest remaining date assumed to be the expiration date
            if dates:
                date = max(dates)
            else:
                date = "No future expiration date found"
        elif dates and len(dates) == 1:
            date = dates[0]
        else:
            date = "No expiration date found"

        return date


if __name__ == "__main__":
    expiryInference = ExpiryInference()

    with open('../../expiry_detector_outputs/expiry_h/results.csv', 'w', newline='') as csvfile:
        result_writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        result_writer.writerow(
            ['Filename', 'Class', 'Number', 'Inference_time', 'Detection'])

    paths = glob('../../expiry_detector_outputs/expiry_h/*.JPG')

    for index, path in enumerate(paths):
        name = path.split('/').pop()
        cat = name.split('_')[0]
        num = name.split('_').pop().split('.')[0]

        print(f"[{index + 1}/{len(paths)}] {int(index * 100 / len(paths))}%  Analysing {name}...")

        image_file = io.open(path, 'rb')
        image_bytes = image_file.read()

        t_start = time.perf_counter()

        date = expiryInference.run_inference(cat, image_bytes)
        print(date)

        t_finish = time.perf_counter()
        t = t_finish - t_start

        print(f"Writing to file...")

        with open('../../expiry_detector_outputs/expiry_h/results.csv', 'a', newline='') as csvfile:
            result_writer = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            result_writer.writerow([name, cat, num, t, date])
