import re
import calendar
import exifread
import dateparser
from datetime import datetime
from Levenshtein import distance


class ExpiryDetector:
    def __init__(self, filter_low, filter_high):
        """
        Constructor for ExpiryDetector class

        Args:
            filter_low (int): Lower limit of the year filter - used to validate year values
            filter_high (int): Upper limit of the year filter - used to validate year values
        """
        self.current_date = datetime.now()
        self.filter_low = filter_low
        self.filter_high = filter_high

    def decompose_texts_list(self, texts):
        """
        Function used to pre-process a list of detected texts.
        Separates alphanumeric strings into alpha and num components
        Further splits strings containing matching deliminators
        Further splits strings containing matching month names

        Args:
            texts (dict): A dictionary containing detection & bbox {detected_word1: bbox1, detected_word2: bbox2}

        Returns
            list: The resulting list of words post split
        """
        month_names = self._get_month_list()
        ret = []

        for text in texts:
            alnum_split = []

            # Split text into alphabetic and numeric components
            # e.g. '4848jun' becomes '4848', 'jun'
            if text.isalnum() and not (text.isalpha() or text.isnumeric()):
                alnum_split = re.findall(r"[^\W\d_]+|\d+", text)
            else:
                alnum_split.append(text)

            # Split results into parts via deliminators if present
            # e.g. '03.06.19' becomes '03', '06', '19'
            delim_split = []
            for alnum_item in alnum_split:
                temp = re.split('\.|,|-|\/|\\|\|', alnum_item)
                for item in temp:
                    if item != '':
                        delim_split.append(item)

            # Split results into parts via month string matches
            # e.g. 'fjjuneifs' becomes 'fj', 'jun' 'ifs'
            for delim_item in delim_split:
                month_split = None

                for month in month_names:
                    if month in delim_item and (len(delim_item) != len(month) + 1):
                        month_split = delim_item.partition(month)
                        break

                if month_split:
                    for item in month_split:
                        if item != '':
                            ret.append(item)
                else:
                    ret.append(delim_item)
        return ret

    def find_month(self, words):
        """
        Searches a list of textual items for month strings e.g. 'jun' or 'august'
        Strings with 1 character distance from full/abbr months are matched successfully

        Args:
            words (list): A list of candidate strings.

        Returns:
            list (index, month, distance): A list of matches containing the detected
                                           index, month and Levenshtein distance
        """
        ret = []
        lowest_dist = None
        month_names = self._get_month_list()

        for index, word in enumerate(words):

            # Find the Levenshein distance between the current word and each full & abbr month
            for month in month_names:
                dist = distance(month, word.lower())

                if dist <= 1:
                    if not ret:
                        lowest_dist = dist
                        ret.append((index, month, dist))
                        continue

                    ret.append((index, month, dist))

                    # Track the lowest distance for filtering later
                    if dist < lowest_dist:
                        lowest_dist = dist

        # Remove all predictions bar the ones with lowest distance value
        ret[:] = [item for item in ret if item[2] <= lowest_dist]

        return ret

    def month_first_search(self, words, detected_months):
        """
        Identifies candidate dates, using a textual month as the root
        Capable of detecting day/month/year, day/month & month/year formats

        Args:
            words (list): A list of detected texts
            detected_months (list(index, month, distance)): A list of matches containing the detected
                                                            index, month and Levenshtein distance

        Returns:
            list: A list of proposed date candidates
        """

        dates = []
        for index, detected_month in enumerate(detected_months):
            month_index = detected_month[0]
            month = detected_month[1]
            lev_distance = detected_month[2]

            # Get the index of the next detected month used to avoid reading
            # the month/day value of the next date as the year
            try:
                next_month_index = detected_months[index + 1][0]
            except IndexError:
                next_month_index = None

            day = year = None

            # Assume a day value exists at the previous index
            try:
                proposed_day = words[month_index - 1]
            except IndexError:
                proposed_day = ''

            # Assume a year value exists at the next index
            try:
                proposed_year = words[month_index + 1]
            except IndexError:
                proposed_year = ''

            # Make sure proposals are valid
            if self.day_is_valid(proposed_day):
                day = proposed_day
            if self.year_is_valid(proposed_year):
                year = self.convert_year(proposed_year)

            # If the next detected month item is 2 positions to the right and
            # the middle item is a valid month value, remove the year proposal
            # e.g. ['20','jun', '21', 'jul'] => 20/06 rather than 20/06/2021
            if next_month_index and (next_month_index - 1 is month_index + 1):
                suspect_proposal = words[month_index + 1]

                if self.day_is_valid(suspect_proposal):
                    year = ''

            # Attempt to parse the full date
            if day and month and year:
                valid_date = dateparser.parse(f"{day} {month} {year}", settings={'DATE_ORDER': 'DMY'})
            elif day and month:
                valid_date = dateparser.parse(f"{day} {month}", settings={'DATE_ORDER': 'DMY'})
            elif month and year:
                valid_date = dateparser.parse(f"01/{month}/{year}", settings={'DATE_ORDER': 'DMY'})
            else:
                valid_date = None

            if valid_date:
                dates.append((valid_date, lev_distance))

        if len(dates) == 0:
            return None
        else:
            return dates

    def year_first_search(self, words):
        """
        Identifies candidate dates, using a numerical year values as the root
        Capable of detecting day/month/year, day/month & month/year formats

        Args:
            words (list): A list of detected texts

        Returns:
            list: A list of proposed date candidates
        """
        current_year = self.current_date.year
        dates = []

        for index, word in enumerate(words):
            # Attempt to convert to year value from 2-digit to 4-digit style
            try:
                year_str = self.convert_year(word)
                year = int(year_str)
            except ValueError:
                continue
            except TypeError:
                continue

            # TODO: Try without this as repeated later on
            if current_year - 20 <= year <= current_year + 20:
                day = month = None
                proposed_month = ''
                proposed_day = ''

                # Assume the month value exists at the index before the year
                try:
                    if index - 1 >= 0:
                        proposed_month = words[index - 1]
                except IndexError:
                    continue

                # Assume the day value exists two indexes before the year
                try:
                    if index - 2 >= 0:
                        proposed_day = words[index - 2]
                except IndexError:
                    pass

                # Filter out fully numeric day values more than 3 chars long
                if proposed_day.isnumeric() and len(proposed_day) > 3:
                    proposed_day = ''

                # If final day char is numeric, crop the last two chars. Remove any non numerical chars
                elif proposed_day and proposed_day[-1].isnumeric():
                    proposed_day = re.sub("[^0-9]", "", proposed_day[-2:])

                # Check if the proposals are valid
                if self.month_is_valid(proposed_month):
                    month = proposed_month

                if self.day_is_valid(proposed_day):
                    day = proposed_day

                # Attempt to parse the full date
                if day and month and year:
                    valid_date = dateparser.parse(f"{day} {month} {year}", settings={'DATE_ORDER': 'DMY'})
                elif month and year:
                    valid_date = dateparser.parse(f"01/{month}/{year}", settings={'DATE_ORDER': 'DMY'})
                else:
                    valid_date = None

                # If a valid date was found, store it with the index of the year component
                if valid_date:
                    dates.append((valid_date, index))

        year_indexes = [item[1] for item in dates]
        copy = year_indexes[:]

        # Remove cases where month is detected as the year
        for index, value in enumerate(copy):
            try:
                if (value + 1) == year_indexes[index + 1]:
                    del dates[index]
            except IndexError:
                pass

        if dates:
            return [item[0] for item in dates]
        else:
            return None

    def convert_year(self, short):
        """
        Converts a 2-digit (short) year value into a 4-digit value

        Args:
            short (str): A year value in short, 2-digit, format i.e. '19' meaning '2019'

        Returns:
            str: The converted year in 4-digit format
        """
        # Return input if input is already 4-characters
        if len(short) == 2:
            pass
        elif len(short) == 3 and short[0] == '0':
            short = short[-2:]
        elif len(short) == 4:
            return short
        else:
            return None

        current_short = str(self.current_date.year)[2:]
        diff = int(short) - int(current_short)

        # Choose the correct 100 year designation based on the current year
        if diff > 50:
            year = str(self.current_date.year - 100)[:2] + short
        elif diff < -50:
            year = str(self.current_date.year + 100)[:2] + short
        else:
            year = str(self.current_date.year)[:2] + short
        return year

    def year_is_valid(self, year):
        """
        Checks whether a given year value is valid

        Args:
            year (str): String representation of a year value (alphabetic or numeric)

        Returns:
            bool: Boolean value indicating the validity of a given year value
        """
        # Year value must be numeric
        if not year.isnumeric():
            return False

        # Convert the year value 4-digit value
        year = self.convert_year(year)

        if not year:
            return False

        year = int(year)

        # Ensures the date is from a realistic year (+-current year)
        if (self.current_date.year - self.filter_low) <= year <= (self.current_date.year + self.filter_high):
            return True
        else:
            return False

    def month_is_valid(self, month):
        """
        Checks whether a given month value is valid

        Args:
            month (str): String representation of a month value (alphabetic or numeric)

        Returns:
            bool: Boolean value indicating the validity of a given month value
        """

        # If alphabetic, check if exists in list of month names
        if month.isalpha():
            month_names = self._get_month_list()

            if month in month_names:
                return True

        # If numeric, check if between 1 and 12
        elif month.isnumeric():
            month = int(month)

            if 1 <= month <= 12:
                return True

        return False

    def day_is_valid(self, day):
        """
        Checks whether a given day value is valid

        Args:
            day (str): String representation of a day value (alphabetic or numeric)

        Returns:
            bool: Boolean value indicating the validity of a given day value
        """
        if day.isnumeric():
            day = int(day)

            if 0 < day < 32:
                return True

        return False

    @staticmethod
    def get_capture_date(path):
        """
        Gets an image's capture datetime from its EXIF tag data

        Args:
            path (str): The path to the image

        Returns:
            datetime: The datetime object representing the image capture datetime
        """
        try:
            with open(path, 'rb') as fh:
                tags = exifread.process_file(fh, stop_tag="EXIF DateTimeOriginal")
                capture_date = str(tags["EXIF DateTimeOriginal"])

                capture_date_split = capture_date.split(' ')[0]
                capture_date_delim = capture_date_split.replace(':', '/')
                ret = dateparser.parse(capture_date_delim)
        except KeyError:
            ret = None
        return ret

    @staticmethod
    def _get_month_list():
        """
        Compiles a list of full and abbreviated month names

        Returns:
            list: A lis of full and abbreviated month names
        """
        month_names = []

        for month_index in range(1, 13):
            month_name = calendar.month_name[month_index]
            month_abbr = calendar.month_abbr[month_index]

            month_name = month_name.lower()
            month_abbr = month_abbr.lower()

            month_names.append(month_name)
            month_names.append(month_abbr)
        return month_names
