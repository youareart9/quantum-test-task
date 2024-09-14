from collections import deque
import sys
from typing import List, Tuple


def perform_bfs(
        matrix: List[List[int]],
        rows: int,
        cols: int,
        start_row: int,
        start_col: int
) -> None:
    """
    Perform Breadth-First Search (BFS) to explore an island and mark all its parts as visited.

    :param matrix: The grid representing the map where 1 is land and 0 is water.
    :param rows: Number of rows in the grid.
    :param cols: Number of columns in the grid.
    :param start_row: The starting row position for BFS.
    :param start_col: The starting column position for BFS.
    :return: None. The matrix is modified in place to mark visited cells.
    """
    queue = deque([(start_row, start_col)])
    matrix[start_row][start_col] = 0  # Mark the current land as visited by setting it to 0
    while queue:
        row, col = queue.popleft()
        # Explore all four directions (up, down, left, right)
        for dir_row, dir_col in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            new_row, new_col = row + dir_row, col + dir_col
            if 0 <= new_row < rows and 0 <= new_col < cols and matrix[new_row][new_col] == 1:
                matrix[new_row][new_col] = 0  # Mark as visited
                queue.append((new_row, new_col))


def count_islands(
        matrix: List[List[int]],
        rows: int,
        cols: int
) -> int:
    """
    Calculate the number of islands in a grid using Breadth-First Search (BFS).

    :param matrix: The grid representing the map where 1 is land and 0 is water.
    :param rows: Number of rows in the grid.
    :param cols: Number of columns in the grid.
    :return: The number of islands found in the grid.
    """
    island_count = 0

    # Iterate through the entire grid
    for row in range(rows):
        for col in range(cols):
            if matrix[row][col] == 1:  # Found a new island
                island_count += 1
                perform_bfs(matrix, rows, cols, row, col)  # Perform BFS to mark the entire island
    return island_count


def parse_input() -> Tuple[int, int, List[List[int]]]:
    """
    Parse and validate the input from the command line.

    :return: A tuple containing:
             - The number of rows in the grid (int).
             - The number of columns in the grid (int).
             - The grid (matrix) as a list of lists of integers (List[List[int]]).
    :raises ValueError: If input does not meet expected format or constraints.
    """
    input_lines = sys.stdin.read().splitlines()

    if len(input_lines) == 0:
        raise ValueError("No input provided.")

    # First line contains the dimensions
    try:
        rows, cols = map(int, input_lines[0].split())
        if rows <= 0 or cols <= 0:
            raise ValueError("Dimensions must be positive integers.")
    except ValueError:
        raise ValueError("First line must contain two integers separated by a space.")

    matrix = []
    for line in input_lines[1:]:
        try:
            row = list(map(int, line.split()))
            if len(row) != cols:
                raise ValueError("Row length does not match number of columns.")
            matrix.append(row)
        except ValueError:
            raise ValueError("Matrix rows must contain only integers separated by spaces.")

    if len(matrix) != rows:
        raise ValueError("Number of rows does not match the number specified.")

    return rows, cols, matrix


def main():
    """
    Main function to read input, execute BFS, and print the number of islands.
    """
    # Parse the input
    rows, cols, matrix = parse_input()

    # Calculate the number of islands using BFS
    result = count_islands(matrix, rows, cols)

    # Print the result (number of islands)
    print(result)


if __name__ == '__main__':
    main()
