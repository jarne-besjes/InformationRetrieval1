import numpy as np
import math

class documentVectorizer():
    def __init__(self):
        pass

    def tf_idf_weight(self, tf: int, df: int, N: int) -> float:
        tf_weight = 1 + math.log10(tf)
        idf_weight = math.log10(N/df)
        return tf_weight * idf_weight