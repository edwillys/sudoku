from src.sudoku import Sudoku
from dataset import *
from os import path as osp

def test_grid(n = 3):
    grid = [-1] * (n ** 4)
    sdk = Sudoku(grid = grid)
    assert(sdk.order == n)

def test_grid4():
    test_grid(4)

def test_validity():
    sdk = Sudoku(grid = valid_grid3)
    assert(sdk.verify())
    sdk = Sudoku(grid = invalid_grid3_1)
    assert(not sdk.verify())
    sdk = Sudoku(grid = invalid_grid3_2)
    assert(not sdk.verify())

def test_generate():
    sdk = Sudoku()
    sdk.generate()
    assert(sdk.verify())

def test_equality():
    sdk1 = Sudoku(grid = valid_grid3)
    sdk2 = Sudoku(grid = valid_grid3)
    assert(sdk1 == sdk2)

def test_dataset(n_tests = 10):
    curr_dir = osp.dirname(__file__)
    with open(osp.join(curr_dir, 'sudoku.csv'), 'r') as fin:
        fin.readline() # skip header
        for _ in range(n_tests):
            line = fin.readline()
            line.replace('\n','')
            [quiz, solution] = line.split(',')
            sdk_quiz = Sudoku(grid = quiz)
            sdk_quiz.solve()
            sdk_solution = Sudoku(grid = solution)
            assert(sdk_quiz == sdk_solution)

if __name__ == "__main__":
    import pytest
    pytest.main()
    