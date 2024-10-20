import re

import nltk.corpus
from nltk.tokenize import word_tokenize
nltk.download('stopwords')

class Token:
    def __init__(self, token: str, pos: int):
        self.token = token
        self.pos = pos

    def __str__(self):
        return f'{self.token, self.pos}'


class TokenStream:
    def __init__(self, tokens: list[str]):
        self.tokens = tokens
        self.pos = -1

    def next(self) -> Token:
        """
        Get the next token in the stream
        :return: Token: the next token in the stream
        """
        self.pos += 1
        return Token(self.tokens[self.pos], self.pos)

    def peek(self) -> Token:
        """
        Peek at the next token in the stream
        :return: Token: the next token in the stream
        """
        return Token(self.tokens[self.pos + 1], self.pos + 1)

    def has_next(self) -> bool:
        """
        Check if there are more tokens in the stream
        :return: bool: True if there are more tokens, False otherwise
        """
        return self.pos + 1 < len(self.tokens)


class Tokenizer:

    @staticmethod
    def _remove_stop_words(tokens: list[str]) -> list[str]:
        """
        Remove stop words from the list of tokens
        :param tokens: The list of tokens to remove stop words from
        :return: list[str]: The list of tokens with stop words removed
        """
        stop_words = set(nltk.corpus.stopwords.words('english'))
        return [token for token in tokens if token not in stop_words]

    @staticmethod
    def tokenize(input, file_input=True, lower_case=True, remove_stop_words=True, stemming=True,
                 remove_punctuation_marks=True, unknown_character_removal=True) -> TokenStream:
        """
        Tokenize the text in the file at the given path
        :param input: The path to the file to tokenize
        :param file_input: Whether the input is a file path or a string
        :param lower_case: Whether to convert the tokens to lowercase
        :param remove_stop_words: Whether to remove stop words from the tokens
        :param stemming: Whether to apply stemming to the tokens
        :param remove_punctuation_marks: Whether to remove punctuation marks from the tokens
        :param unknown_character_removal: Whether to remove unknown characters from the tokens
        :return: TokenStream: A stream of tokens from the file
        """
        if file_input:
            with open(input, 'r', encoding='utf-8') as file:
                text = file.read()
        else:
            text = input
        # Preprocess text
        # split words connected by capitals (e.g. cityOfLondon -> [city, Of, London])
        text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
        if lower_case:
            text = text.lower()
        if remove_punctuation_marks:
            text = ''.join([char for char in text if char not in ['.', ',', '!', '?', ':', ';', '"', "'"]])
        if unknown_character_removal:
            text = ''.join([char for char in text if char.isascii() or char.isspace()])
        tokens = word_tokenize(text)
        if remove_stop_words:
            tokens = Tokenizer._remove_stop_words(tokens)
        if stemming:
            stemmer = nltk.stem.PorterStemmer()
            tokens = [stemmer.stem(token) for token in tokens]
        return TokenStream(tokens)

if __name__ == '__main__':
    stream = Tokenizer.tokenize('crunchPasta, i like crunchyFood! I like different types, roads and huggers?, DCRF', file_input=False)
    while stream.has_next():
        print(stream.next())