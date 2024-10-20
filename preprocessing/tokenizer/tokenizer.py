import nltk.corpus
from nltk.tokenize import word_tokenize


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
    def tokenize(file_path, lower_case=True, remove_stop_words=True, stemming=True,
                 remove_punctuation_marks=True) -> TokenStream:
        """
        Tokenize the text in the file at the given path
        :param file_path: The path to the file to tokenize
        :param lower_case: Whether to convert the tokens to lowercase
        :param remove_stop_words: Whether to remove stop words from the tokens
        :param stemming: Whether to apply stemming to the tokens
        :param remove_punctuation_marks: Whether to remove punctuation marks from the tokens
        :return: TokenStream: A stream of tokens from the file
        """
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
            if lower_case:
                text = text.lower()
        tokens = word_tokenize(text)
        if remove_stop_words:
            tokens = Tokenizer._remove_stop_words(tokens)
        if stemming:
            stemmer = nltk.stem.PorterStemmer()
            tokens = [stemmer.stem(token) for token in tokens]
        if remove_punctuation_marks:
            tokens = [token for token in tokens if token not in ['.', ',', '!', '?', ';', ':', '"', "'"]]
        return TokenStream(tokens)


if __name__ == "__main__":
    stream = Tokenizer.tokenize('test.txt')
    while stream.has_next():
        print(stream.next())
