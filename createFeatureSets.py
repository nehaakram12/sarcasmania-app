import numpy as np
from textblob import TextBlob
import nltk
import string
import exp_replace
import random


class CreateFeatureSet:
    # Read the data from numpy files into arrays
    def __init__(self):
        self.featuresets = []
        self.classes = ["SARCASTIC", "REGULAR"]

    def extract_feature_of_sentence(self, sen):
        # type: (object) -> object
        """
        This method extracts features of a single sentence.
        We have following list of features being extracted.
        1. Full sentence Polarity
        2. Full sentence Subjectivity
        3. Half sentence Polarity (1/2 and 2/2)
        4. Half sentence Subjectivity (1/2 and 2/2)
        5. Difference between polarities of two halves
        6. Third sentence Polarity (1/3, 2/3 and 3/3)
        7. Third sentence Subjectivity (1/3, 2/3 and 3/3)
        8. Difference between max and min polarity of the thirds.
        9. Fourth sentence Polarity (1/4, 2/4, 3/4 and 4/4)
        10. Fourth sentence Subjectivity (1/4, 2/4, 3/4 and 4/4)
        11. Difference between max and min polarities of the fourths.

        Like this we extract 23 features of a single sentence.
        :param sen:
        :return:
        """
        features = []

        # Tokenize the sentence and then convert everything to lower case.
        tokens = nltk.word_tokenize(exp_replace.replace_emo(str(sen)))
        tokens = [(t.lower()) for t in tokens]

        # Extract features of full sentence.
        fullBlob = TextBlob(self.join_tokens(tokens))
        features.append(fullBlob.sentiment.polarity)
        features.append(fullBlob.sentiment.subjectivity)

        # Extract features of halves.
        size = len(tokens) // 2
        parts = []
        i = 0
        while i <= len(tokens):
            if i == size:
                parts.append(tokens[i:])
                break
            else:
                parts.append(tokens[i:i + size])
                i += size
        for x in range(0, len(parts)):
            part = parts[x]
            halfBlob = TextBlob(self.join_tokens(part))
            features.append(halfBlob.sentiment.polarity)
            features.append(halfBlob.sentiment.subjectivity)
        features.append(np.abs(features[-2] - features[-4]))

        # Extract features of thirds.
        size = len(tokens) // 3
        parts = []
        i = 0
        while i <= len(tokens):
            if i == 2 * size:
                parts.append(tokens[i:])
                break
            else:
                parts.append(tokens[i:i + size])
                i += size

        ma = -2
        mi = 2
        for x in range(0, len(parts)):
            part = parts[x]
            thirdsBlob = TextBlob(self.join_tokens(part))
            pol = thirdsBlob.sentiment.polarity
            sub = thirdsBlob.sentiment.subjectivity
            if pol > ma:
                ma = pol
            if pol < mi:
                mi = pol
            features.append(pol)
            features.append(sub)
        features.append(np.abs(ma - mi))

        # Extract features of fourths.
        size = len(tokens) // 4
        parts = []
        i = 0
        while i <= len(tokens):
            if i == 3 * size:
                parts.append(tokens[i:])
                break
            else:
                parts.append(tokens[i:i + size])
                i += size
        ma = -2
        mi = 2
        for x in range(0, len(parts)):
            part = parts[x]
            fourths_blob = TextBlob(self.join_tokens(part))
            pol = fourths_blob.sentiment.polarity
            sub = fourths_blob.sentiment.subjectivity
            if pol > ma:
                ma = pol
            if pol < mi:
                mi = pol
            features.append(pol)
            features.append(sub)
        features.append(np.abs(ma - mi))

        return features

    def join_tokens(self, t):
        """
        This method joins tokes into a single text avoiding punctuations and
        special characters as required by the textblob api.
        :param t:
        :return:
        """
        s = ""
        for i in t:
            if i not in string.punctuation and not i.startswith("'"):
                s += (" " + i)
        return s.strip()


if __name__ == '__main__':
    CreateFeatureSet().extract_features()
