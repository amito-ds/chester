import unittest


class TestCleanText(unittest.TestCase):
    def test_clean_text(self):
        # Create a sample DataFrame
        df = pd.DataFrame(
            {'text': ['Hello, world!', '<p>This is some html</p>', '1.23', 'An apple a day keeps the doctor away!',
                      'I am a very complex sentence with lots of words and special characters!',
                      'I am another complex sentence with lots of words and numbers 1234567890!',
                      'I am a complex sentence with lots of html tags <br>, <p>, and <h1>!']})

        # Clean the text column
        df['text'] = df['text'].apply(clean_text, lowercase=True, remove_punctuation=True, remove_digits=True,
                                      remove_html=True, remove_stopwords=True, stem_words=True, lemmatize_words=True)

        # Assert that the text was cleaned correctly
        self.assertEqual(df.iloc[0]['text'], 'hello world')
        self.assertEqual(df.iloc[1]['text'], 'p some html')
        self.assertEqual(df.iloc[2]['text'], '')
        self.assertEqual(df.iloc[3]['text'], 'appl day keep doctor')
        self.assertEqual(df.iloc[4]['text'], 'complex sentenc lot word special character')
        self.assertEqual(df.iloc[5]['text'], 'another complex sentenc lot word number')
        self.assertEqual(df.iloc[6]['text'], 'complex sentenc lot html tag br p h1')


# Run the test
unittest.main(argv=[''], exit=False)
