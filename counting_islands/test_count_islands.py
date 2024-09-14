import unittest
from count_islands import count_islands


class TestNumIslandsBFS(unittest.TestCase):

    def test_first_example(self):
        """Test first example."""
        matrix = [
            [0, 1, 0],
            [0, 0, 0],
            [0, 1, 1]
        ]
        self.assertEqual(count_islands(matrix, 3, 3), 2)

    def test_second_example(self):
        """Test second example."""
        matrix = [
            [0, 0, 0, 1],
            [0, 0, 1, 0],
            [0, 1, 0, 0]
        ]
        self.assertEqual(count_islands(matrix, 3, 4), 3)

    def test_third_example(self):
        """Test third example."""
        matrix = [
            [0, 0, 0, 1],
            [0, 0, 1, 1],
            [0, 1, 0, 1]
        ]
        self.assertEqual(count_islands(matrix, 3, 3), 2)

    def test_single_island(self):
        """Test a single island in a 3x3 grid."""
        matrix = [
            [0, 1, 0],
            [0, 1, 0],
            [0, 0, 0]
        ]
        self.assertEqual(count_islands(matrix, 3, 3), 1)

    def test_large_island(self):
        """Test a 5x5 grid where the entire land is one big island."""
        matrix = [
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1]
        ]
        self.assertEqual(count_islands(matrix, 5, 5), 1)

    def test_no_island(self):
        """Test a 3x3 grid with no islands (all ocean)."""
        matrix = [
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0]
        ]
        self.assertEqual(count_islands(matrix, 3, 3), 0)

    def test_single_cell_island(self):
        """Test a 1x1 grid with a single land cell."""
        matrix = [[1]]
        self.assertEqual(count_islands(matrix, 1, 1), 1)

    def test_single_cell_ocean(self):
        """Test a 1x1 grid with a single ocean cell."""
        matrix = [[0]]
        self.assertEqual(count_islands(matrix, 1, 1), 0)


if __name__ == '__main__':
    unittest.main()
