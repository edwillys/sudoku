from src.sudoku_engine import Sudoku
from .dataset import *
from os import path as osp


def test_grid(n=3) -> None:
    grid = [
        [-1] for _ in range(n ** 2)
        for _ in range(n ** 2)
    ]
    sdk = Sudoku(n, grid=grid)
    assert (sdk.order == n)


def test_grid4() -> None:
    test_grid(4)


def test_validity() -> None:
    sdk = Sudoku(3, grid=valid_grid3)
    assert (sdk.verify())
    sdk = Sudoku(3, grid=invalid_grid3_1)
    assert (not sdk.verify())
    sdk = Sudoku(3, grid=invalid_grid3_2)
    assert (not sdk.verify())


def test_generate() -> None:
    sdk = Sudoku(3)
    sdk.generate()
    assert (sdk.verify())


def test_equality() -> None:
    sdk1 = Sudoku(3, grid=valid_grid3)
    sdk2 = Sudoku(3, grid=valid_grid3)
    assert (sdk1 == sdk2)


def test_dataset(n_tests=100) -> None:
    curr_dir = osp.dirname(__file__)
    with open(osp.join(curr_dir, 'sudoku.csv'), 'r') as fin:
        fin.readline()  # skip header
        for _ in range(n_tests):
            line = fin.readline()
            line = line.replace('\n', '')
            [quiz, solution] = line.split(',')
            sdk_quiz = Sudoku(3, grid=quiz)
            assert (sdk_quiz.solve())
            sdk_solution = Sudoku(3, grid=solution)
            assert (sdk_quiz == sdk_solution)


if __name__ == "__main__":
    import pytest
    pytest.main()
