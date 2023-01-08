from cleaning import *


def test_remove_punctuation():
    text = "This is a test! Can you remove the punctuation?"
    expected_output = "This is a test Can you remove the punctuation"
    assert remove_punctuation(text) == expected_output


def test_remove_numbers():
    text = "This is a test! Can you remove the 123 numbers?"
    expected_output = "This is a test! Can you remove the  numbers?"
    assert remove_numbers(text) == expected_output


def test_remove_whitespace():
    text = "This is a test!    Can you remove the extra whitespace?"
    expected_output = "This is a test! Can you remove the extra whitespace?"
    assert remove_whitespace(text) == expected_output


def test_lowercase():
    text = "This Is A Test! Can You Convert It To Lowercase?"
    expected_output = "this is a test! can you convert it to lowercase?"
    assert lowercase(text) == expected_output


def test_general_clean():
    text = "This Is A Test! Can You Clean It Up? 123"
    expected_output = "this is a test! can you clean it up?"
    assert general_clean(text) == expected_output


def test_remove_stopwords():
    text = "This is a test! Can you remove the stopwords?"
    stopwords = ['remove', 'the']
    expected_output = "This is a test! Can you stopwords?"
    assert remove_stopwords(text, stopwords) == expected_output


def test_lemmatize():
    text = "This is a test! Can you lemmatize the words?"
    expected_output = "This be a test! Can you lemmatize the word?"
    assert lemmatize(text, language='english') == expected_output


def test_stem():
    text = "This is a test! Can you stem the words?"
    expected_output = "This is a test! Can you stem the word?"
    assert stem(text, language='english') == expected_output


def test_remove_accented_characters():
    text = "This is a test! Can you remove the accented characters?"
    expected_output = "This is a test! Can you remove the accented characters?"
    assert remove_accented_characters(text) == expected_output


def test_remove_special_characters():
    text = "This is a test! Can you remove the special characters?"
    expected_output = "This is a test Can you remove the special characters"
    assert remove_special_characters(text) == expected_output


def test_remove_html_tags():
    text = "This is a test! Can you <b>remove</b> the HTML tags?"
    expected_output = "This is a test! Can you remove the HTML tags?"
    assert remove_html_tags(text) == expected_output
