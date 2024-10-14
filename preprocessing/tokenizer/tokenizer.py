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
    def tokenize(file_path):
        """
        Tokenize the text in the file at the given path
        :param file_path: The path to the file to tokenize
        :return: TokenStream: A stream of tokens from the file
        """
        with open(file_path, 'r') as file:
            text = file.read()
        return TokenStream(word_tokenize(text))
