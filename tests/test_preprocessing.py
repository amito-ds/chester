import unittest

from preprocessing.preprocessing_func import tokenize, pos_tag, stem, lemmatize


class TestProcessingMethods(unittest.TestCase):

    def test_tokenize(self):
        text = "This is a sample sentence showing how tokenization works."
        expected_tokens = ['This', 'is', 'a', 'sample', 'sentence', 'showing', 'how', 'tokenization', 'works', '.']
        self.assertEqual(tokenize(text), expected_tokens)

    def test_pos_tag(self):
        text = "This is a sample sentence showing how POS tagging works."
        expected_pos_tags = [('This', 'DT'), ('is', 'VBZ'), ('a', 'DT'), ('sample', 'JJ'), ('sentence', 'NN'),
                             ('showing', 'VBG'), ('how', 'WRB'), ('POS', 'NNP'), ('tagging', 'VBG'), ('works', 'NNS'),
                             ('.', '.')]

        self.assertEqual(pos_tag(text), expected_pos_tags)


class TestStemming(unittest.TestCase):
    def test_stemming(self):
        words = ['run', 'running', 'ran', 'runs', 'easily', 'fairly']
        expected_stems = ['run', 'run', 'ran', 'run', 'easili', 'fairli']
        self.assertEqual(stem(words), expected_stems)

    def test_stemming_case_insensitivity(self):
        words = ['Run', 'Running', 'Ran', 'Runs', 'Easily', 'Fairly']
        expected_stems = ['run', 'run', 'ran', 'run', 'easili', 'fairli']
        self.assertEqual(stem(words), expected_stems)

    def test_base_form_words(self):
        words = ['run', 'easily', 'fairly']
        expected_stems = ['run', 'easili', 'fairli']
        self.assertEqual(stem(words), expected_stems)

    def test_lemmatization(self):
        text = ['run', 'running', 'ran', 'runs', 'easily', 'fairly']
        expected_lemmas = ['run', 'run', 'ran', 'run', 'easily', 'fairly']
        self.assertEqual(lemmatize(text), expected_lemmas)

    def test_lemmatization_case_insensitivity(self):
        text = ['Run', 'Running', 'Ran', 'Runs', 'Easily', 'Fairly']
        expected_lemmas = ['Run', 'Running', 'Ran', 'Runs', 'Easily', 'Fairly']
        self.assertEqual(lemmatize(text), expected_lemmas)

    if __name__ == '__main__':
        unittest.main()
