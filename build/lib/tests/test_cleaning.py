import unittest

from cleaning.cleaning import *


class TestCleaningMethods(unittest.TestCase):
    def test_remove_punctuation(self):
        text = "This is a test! Can you remove the punctuation?"
        expected_output = "This is a test Can you remove the punctuation"
        assert remove_punctuation(text) == expected_output

    def test_remove_numbers(self):
        text = "This is a test! Can you remove the 123 numbers?"
        expected_output = "This is a test! Can you remove the  numbers?"
        assert remove_numbers(text) == expected_output

    def test_remove_whitespace(self):
        text = "This is a test!    Can you remove the extra whitespace?"
        expected_output = "This is a test! Can you remove the extra whitespace?"
        assert remove_whitespace(text) == expected_output

    def test_lowercase(self):
        text = "This Is A Test! Can You Convert It To Lowercase?"
        expected_output = "this is a test! can you convert it to lowercase?"
        assert lowercase(text) == expected_output

    def test_remove_stopwords(self):
        text = "This is a test! Can you remove the stopwords?"
        stopwords = ['remove', 'the']
        expected_output = "This is a test! Can you stopwords?"
        assert remove_stopwords(text, stopwords) == expected_output

    def test_remove_accented_characters(self):
        text = "This is a test! Can you remove the accented characters?"
        expected_output = "This is a test! Can you remove the accented characters?"
        assert remove_accented_characters(text) == expected_output

    def test_remove_special_characters(self):
        text = "This is a test! Can you remove the special characters?"
        expected_output = "This is a test Can you remove the special characters"
        assert remove_special_characters(text) == expected_output

    def test_remove_html_tags(self):
        text = "This is a test! Can you <b>remove</b> the HTML tags?"
        expected_output = "This is a test! Can you remove the HTML tags?"
        assert remove_html_tags(text) == expected_output


if __name__ == '__main__':
    unittest.main()
