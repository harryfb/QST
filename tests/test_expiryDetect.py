import unittest
from datetime import datetime

from detectors.expiryDetect import ExpiryDetector


class TestExpiryRecognition(unittest.TestCase):

    def setUp(self):
        self.expiryDetector = ExpiryDetector(30, 30)

    def test_texts_decomposition(self):
        """
        Test decomposition of raw text detection output
        Splits strings using delims, textual month matches and alphanums
        """
        texts = {
            'test': [0, 10, 10, 0],
            '20juntt': [0, 10, 10, 0],
            '43,f,': [0, 10, 10, 0],
            '3word33': [0, 10, 10, 0],
            'whole': [0, 10, 10, 0]
        }
        expected = ['test', '20', 'jun', 'tt', '43', 'f', '3', 'word', '33', 'whole']
        result = self.expiryDetector.decompose_texts_list(texts)

        self.assertEqual(expected, result)

    def test_find_month_exact_match(self):
        """
        Test discovery of month keywords
        """
        words = ['test', '20', 'jun', 'tt', '3', 'december', '33', 'whole', 'novenber']

        expected = [(2, 'jun', 0), (5, 'december', 0)]
        result = self.expiryDetector.find_month(words)

        self.assertEqual(expected, result)

    def test_find_month_close_match(self):
        """
        Test no exact matches and distance filter
        """
        words = ['test', '20', 'jin', 'tt', '3', 'dcember', 'matzh', '33', 'whole']

        expected = [(2, 'jan', 1), (2, 'jun', 1), (5, 'december', 1)]
        result = self.expiryDetector.find_month(words)

        self.assertEqual(expected, result)

    def test_month_first_search(self):
        """
        Test month first date search, using the month as the root
        """
        words = ['test', '20', 'jun', '20', 'tt', '3', 'december', '23', 'whole', 'novenber']
        detected_months = [(2, 'jun', 0), (6, 'december', 0)]

        expected = [(datetime(2020, 6, 20), 0), (datetime(2023, 12, 3), 0)]
        result = self.expiryDetector.month_first_search(words, detected_months)

        self.assertEqual(expected, result)

    def test_month_first_search_b2b_dates(self):
        """
        Test month first date search, where there are back to back month strings
        """
        words = ['test', '20', 'jun', '30', 'jul', '3', 'december', '23', 'whole', 'novenber']
        detected_months = [(2, 'jun', 0), (4, 'jul', 0), (6, 'december', 0)]

        expected = [(datetime(2020, 6, 20), 0), (datetime(2020, 7, 30), 0), (datetime(2023, 12, 3), 0)]
        result = self.expiryDetector.month_first_search(words, detected_months)
        self.assertEqual(expected, result)

        words = ['test', '20', 'jun', '30', 'jul', 'december', '23', 'whole', 'novenber']
        detected_months = [(2, 'jun', 0), (4, 'jul', 0), (5, 'december', 0)]

        expected = [(datetime(2020, 6, 20), 0), (datetime(2020, 7, 30), 0), (datetime(2023, 12, 1), 0)]
        result = self.expiryDetector.month_first_search(words, detected_months)
        self.assertEqual(expected, result)

    def test_month_first_search_only_month(self):
        """
        Test month first date search when there are no other surrounding date components
        """
        words = ['test', 'jun', 'tte', 'juaddl', 'dgs', 'ddd', 'whole', 'novenber']
        detected_months = [(2, 'jun', 0)]

        expected = None
        result = self.expiryDetector.month_first_search(words, detected_months)
        self.assertEqual(expected, result)

    def test_year_first_search(self):
        """
        Test year first search, using the year number as the root
        """
        words = ['test', 'gfd', '29', '12', '20', 'ddd', '1', '21']

        expected = [datetime(2020, 12, 29), datetime(2021, 1, 1)]
        result = self.expiryDetector.year_first_search(words)

        self.assertEqual(expected, result)

    def test_year_first_search_b2b_dates(self):
        """
        Test year first search where dates are back to back
        """
        words = ['test', 'gfd', '29', '12', '20', '12', '1', '21', '4', '20']

        expected = [datetime(2020, 12, 29), datetime(2021, 1, 12), datetime(2020, 4, 21)]
        result = self.expiryDetector.year_first_search(words)

        self.assertEqual(expected, result)

    def test_year_first_search_only_year(self):
        """
        Test year first search when no other date components surround year
        """
        words = ['test', 'gfd', '2029', '203']

        expected = None
        result = self.expiryDetector.year_first_search(words)

        self.assertEqual(expected, result)

    def test_convert_year_1val(self):
        """
        Test conversion of a single digit year
        """
        year = '1'
        expected = None
        result = self.expiryDetector.convert_year(year)

        self.assertEqual(expected, result)

    def test_convert_year_2val(self):
        """
        Test conversion of a two digit year
        """
        year = '10'
        expected = '2010'
        result = self.expiryDetector.convert_year(year)

        self.assertEqual(expected, result)

        year = '82'
        expected = '1982'
        result = self.expiryDetector.convert_year(year)

        self.assertEqual(expected, result)

    def test_convert_year_3val(self):
        """
        Test conversion of a three digit year
        """
        year = '020'
        expected = '2020'
        result = self.expiryDetector.convert_year(year)

        self.assertEqual(expected, result)

    def test_convert_year_4val(self):
        """
        Test conversion of a four digit year
        """
        year = '2021'
        expected = '2021'
        result = self.expiryDetector.convert_year(year)

        self.assertEqual(expected, result)

    def test_year_is_valid(self):
        """
        Test year validation
        """
        year = '2021'
        result = self.expiryDetector.year_is_valid(year)
        self.assertTrue(result)

        year = '1800'
        result = self.expiryDetector.year_is_valid(year)
        self.assertFalse(result)

        year = '2200'
        result = self.expiryDetector.year_is_valid(year)
        self.assertFalse(result)

        year = '2f00'
        result = self.expiryDetector.year_is_valid(year)
        self.assertFalse(result)

        year = '20'
        result = self.expiryDetector.year_is_valid(year)
        self.assertTrue(result)

    def test_month_is_valid(self):
        """
        Test month validation
        """
        month = 'apr'
        result = self.expiryDetector.month_is_valid(month)
        self.assertTrue(result)

        month = 'october'
        result = self.expiryDetector.month_is_valid(month)
        self.assertTrue(result)

        month = '1'
        result = self.expiryDetector.month_is_valid(month)
        self.assertTrue(result)

        month = '12'
        result = self.expiryDetector.month_is_valid(month)
        self.assertTrue(result)

        month = 'apr3'
        result = self.expiryDetector.month_is_valid(month)
        self.assertFalse(result)

        month = 'apsr'
        result = self.expiryDetector.month_is_valid(month)
        self.assertFalse(result)

        month = '13'
        result = self.expiryDetector.month_is_valid(month)
        self.assertFalse(result)

    def test_day_is_valid(self):
        """
        Test day validation
        """
        month = '1'
        result = self.expiryDetector.day_is_valid(month)
        self.assertTrue(result)

        month = '31'
        result = self.expiryDetector.day_is_valid(month)
        self.assertTrue(result)

        month = '32'
        result = self.expiryDetector.day_is_valid(month)
        self.assertFalse(result)

        month = 'gsa'
        result = self.expiryDetector.day_is_valid(month)
        self.assertFalse(result)

        month = 'a2psr'
        result = self.expiryDetector.day_is_valid(month)
        self.assertFalse(result)


if __name__ == '__main__':
    unittest.main(verbosity=2)
