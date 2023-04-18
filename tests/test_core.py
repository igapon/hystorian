import numpy as np
import unittest


class TestFileHandlerClass(unittest.TestCase):
    def generate_empty_file(self):
        d1 = np.random.random(size=(1000, 1000))
