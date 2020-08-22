import unittest
from datetime import datetime

from recognition.expiryRec import ExpiryInference


class TestExpiryRecognition(unittest.TestCase):

    def setUp(self):
        self.expiryInference = ExpiryInference()

    def test_reduce_candidates_month_first(self):
        """
        Test month-first predictions only
        """
        month_first_proposals = [(datetime(2020, 9, 20), 1), (datetime(2025, 2, 10), 1), (datetime(2013, 3, 17), 1)]
        year_first_proposals = None

        result = self.expiryInference.reduce_candidates(month_first_proposals, year_first_proposals)
        expected = [item[0] for item in month_first_proposals]
        self.assertListEqual(result, expected)

    def test_reduce_candidates_year_first(self):
        """
        Test year-first predictions only
        """
        month_first_proposals = None
        year_first_proposals = [datetime(2020, 9, 20), datetime(2025, 2, 10), datetime(2013, 3, 17)]

        result = self.expiryInference.reduce_candidates(month_first_proposals, year_first_proposals)
        self.assertListEqual(result, year_first_proposals)

    def test_reduce_candidates_both_dist1(self):
        """
        Test when both month-first & year-first predictions
        month-first dists = 1 so should not be chosen
        """
        month_first_proposals = [(datetime(2023, 3, 21), 1), (datetime(2019, 2, 10), 1), (datetime(2000, 3, 1), 1)]
        year_first_proposals = [datetime(2020, 9, 20), datetime(2025, 2, 10), datetime(2013, 3, 17)]

        result = self.expiryInference.reduce_candidates(month_first_proposals, year_first_proposals)
        self.assertListEqual(result, year_first_proposals)

    def test_reduce_candidates_both_dist0(self):
        """
        Test when both month-first & year-first predictions
        month-first dists = 0 so should be chosen
        """
        month_first_proposals = [(datetime(2023, 3, 21), 0), (datetime(2019, 2, 10), 0)]
        year_first_proposals = [datetime(2020, 9, 20), datetime(2025, 2, 10), datetime(2013, 3, 17)]

        result = self.expiryInference.reduce_candidates(month_first_proposals, year_first_proposals)
        expected = [item[0] for item in month_first_proposals]

        self.assertListEqual(result, expected)

    def test_make_prediction_no_dates(self):
        """
        Test final prediction when no dates found
        """
        current_date = self.expiryInference.expiryDetector.current_date

        result = self.expiryInference.make_prediction(None, current_date, None)
        self.assertEqual(result, "No expiration date found")

    def test_make_prediction_no_future_dates(self):
        """
        Test final prediction when no dates found in the future
        """
        current_date = self.expiryInference.expiryDetector.current_date
        dates = [datetime(2010, 9, 20), datetime(2018, 10, 20), datetime(2013, 3, 17)]

        result = self.expiryInference.make_prediction(dates, current_date, None)
        self.assertEqual(result, "No future expiration date found")

    def test_make_prediction_no_class(self):
        """
        Test final prediction when no class is specified
        """
        current_date = self.expiryInference.expiryDetector.current_date
        dates = [datetime(2020, 9, 1), datetime(2020, 10, 20), datetime(2013, 3, 17)]

        result = self.expiryInference.make_prediction(dates, current_date, None)
        expected = datetime(2020, 10, 20)

        self.assertGreaterEqual(result, current_date)
        self.assertEqual(result, expected)

    def test_make_prediction_class(self):
        """
        Test final prediction when a class has been specified
        """
        current_date = self.expiryInference.expiryDetector.current_date

        dates = [datetime(2020, 9, 1), datetime(2020, 10, 20), datetime(2013, 3, 17)]
        category = "milk"

        result = self.expiryInference.make_prediction(dates, current_date, category)
        expected = datetime(2020, 9, 1)

        self.assertGreaterEqual(result, current_date)
        self.assertEqual(result, expected)


if __name__ == '__main__':
    unittest.main(verbosity=2)
