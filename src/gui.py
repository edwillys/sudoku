
from PyQt5 import QtGui
from PyQt5.QtGui import QFont, QPainter, QPen, QCursor
from PyQt5.QtCore import Qt, QRectF, QLineF, QSize, pyqtSlot
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, \
    QBoxLayout, QGridLayout, QPushButton, QWidget, QMenu, QSizePolicy, QSpacerItem, \
    QMenuBar, QAction
import sys
from functools import partial
import numpy as np


class SudokuInsertNumberMenu(QMenu):
    def __init__(self, parent: QPushButton, button_size: int = 70, dimension: int = 3, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dimension = dimension

        grid = QGridLayout()
        self.buttons = [QPushButton(str(i))
                        for i in range(self.dimension ** 2)]

        for i, btn in enumerate(self.buttons):
            btn.setFixedSize(QSize(button_size, button_size))
            grid.addWidget(btn, i // dimension, i % dimension)
            btn.clicked.connect(partial(self.onBtnClicked, i, parent))

        grid.setSpacing(0)
        self.setLayout(grid)

    @ pyqtSlot()
    def onBtnClicked(self, number: int, parent: QPushButton):
        parent.setText(str(number))
        self.close()


class SudokuWidget(QWidget):
    def __init__(self, parent, dimension: int = 3):
        super().__init__(parent)
        self.setMinimumSize(400, 400)
        self.puzzle = self.generate(dimension)
        self.solution = self.solve()
        self.mainLayout = QBoxLayout(QBoxLayout.LeftToRight, self)
        self.grid = SquareBtnGrid(self.puzzle)
        self.mainLayout.addItem(QSpacerItem(0, 0))
        self.mainLayout.addWidget(self.grid)
        self.mainLayout.addItem(QSpacerItem(0, 0))
        self.setLayout(self.mainLayout)

    def setVals(self, vals, disable: bool):
        self.grid.setVals(vals, disable)

    def setDimension(self, dimension: int):
        self.puzzle = self.generate(dimension)
        self.solution = self.solve()
        self.setVals(self.puzzle, True)

    def resizeEvent(self, a0: QtGui.QResizeEvent) -> None:
        w = a0.size().width()
        h = a0.size().height()
        if w > h:  # too wide
            self.mainLayout.setDirection(QBoxLayout.LeftToRight)
            widget_stretch = h
            outer_stretch = (w - widget_stretch) / 2
        else:  # too tall
            self.mainLayout.setDirection(QBoxLayout.TopToBottom)
            widget_stretch = w
            outer_stretch = (h - widget_stretch) / 2
        self.mainLayout.setStretch(0, int(outer_stretch))
        self.mainLayout.setStretch(1, int(widget_stretch))
        self.mainLayout.setStretch(2, int(outer_stretch))

    def reset(self):
        self.grid.reset(self.puzzle)

    def showSolution(self):
        self.setVals(self.solution, False)

    def check(self):
        vals = self.grid.getVals()
        inds_wrong = np.argwhere((vals != self.solution) & (vals > 0))
        inds_right = np.argwhere((vals == self.solution) & (self.puzzle < 0))
        self.grid.setBtnStylesheetAt(inds_wrong, "background-color: red")
        self.grid.setBtnStylesheetAt(inds_right, "background-color: green")

    def uncheck(self):
        inds = np.argwhere(self.puzzle < 0)
        self.grid.setBtnStylesheetAt(inds, "")

    def solve(self):
        if len(self.puzzle) == 9:
            solution = [
                [8, 6, 4, 3, 7, 1, 2, 5, 9],
                [3, 2, 5, 8, 4, 9, 7, 6, 1],
                [9, 7, 1, 2, 6, 5, 8, 4, 3],
                [4, 3, 6, 1, 9, 2, 5, 8, 7],
                [1, 9, 8, 6, 5, 7, 4, 3, 2],
                [2, 5, 7, 4, 8, 3, 9, 1, 6],
                [6, 8, 9, 7, 3, 4, 1, 2, 5],
                [7, 1, 3, 5, 2, 8, 6, 9, 4],
                [5, 4, 2, 9, 1, 6, 3, 7, 8]
            ]
        else:
            solution = [
                [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            ]
        return np.array(solution)

    def generate(self, dimension: int):
        vals = []
        if dimension == 3:
            vals = [
                [-1, -1, 4, 3, -1, -1, 2, -1, 9],
                [-1, -1, 5, -1, -1, 9, -1, -1, 1],
                [-1, 7, -1, -1, 6, -1, -1, 4, 3],
                [-1, -1, 6, -1, -1, 2, -1, 8, 7],
                [1, 9, -1, -1, -1, 7, 4, -1, -1],
                [-1, 5, -1, -1, 8, 3, -1, -1, -1],
                [6, -1, -1, -1, -1, -1, 1, -1, 5],
                [-1, -1, 3, 5, -1, 8, 6, 9, -1],
                [-1, 4, 2, 9, 1, -1, 3, -1, -1],
            ]
        elif dimension == 4:
            vals = [
                [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            ]
        return np.array(vals)


class SquareBtnGrid(QWidget):
    def __init__(self, vals: list[list[int]], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.spacing = 2
        self.gridLayout = QGridLayout()
        self.setVals(vals, True)
        self.gridLayout.setSpacing(self.spacing)
        self.setLayout(self.gridLayout)

    def getVals(self) -> list[list[int]]:
        vals = np.array([
            [-1 for _ in range(self.cols)]
            for _ in range(self.rows)
        ])
        for row, btn_row in enumerate(self.buttons):
            for col, btn in enumerate(btn_row):
                try:
                    vals[row][col] = int(btn.text())
                except:
                    pass

        return vals

    def setBtnStylesheetAt(self, positions, stylesheet: str):
        for row, col in positions:
            self.buttons[row][col].setStyleSheet(stylesheet)

    def setVals(self, vals: list[list[int]], disable: bool):
        new_count = len(vals) ** 2
        cur_count = self.gridLayout.count()

        if new_count != cur_count:
            for i in reversed(range(self.gridLayout.count())):
                self.gridLayout.itemAt(i).widget().setParent(None)
            self.dimension = int(np.sqrt(len(vals)))
            self.cols = self.dimension ** 2
            self.rows = self.cols

            self.buttons = [
                [QPushButton() for _ in range(self.cols)]
                for _ in range(self.rows)
            ]

            for row, btn_row in enumerate(self.buttons):
                for col, btn in enumerate(btn_row):
                    btn.clicked.connect(partial(self.onBtnClicked, btn))
                    # get the buttons to stretch both horizontally and vertically
                    size_policy = QSizePolicy(
                        QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Minimum)
                    size_policy.setVerticalStretch(1)
                    size_policy.setHorizontalStretch(1)
                    btn.setSizePolicy(size_policy)
                    btn.setFont(QFont("Courrier New", 15))
                    self.gridLayout.addWidget(btn, row, col)

        for row, btn_row in enumerate(self.buttons):
            for col, btn in enumerate(btn_row):
                val = vals[row][col]
                # set initial empty text for place holder, so that the resizing doesn't get messed up
                if val > 0:
                    btn.setText(str(val))
                    btn.setEnabled(not disable)
                else:
                    btn.setText(" ")

    @ pyqtSlot()
    def onBtnClicked(self, item: QPushButton):
        menu = SudokuInsertNumberMenu(item, dimension=self.dimension)
        menu.move(QCursor.pos())
        menu.show()

    def paintEvent(self, event):
        qp = QPainter(self)
        pen_normal_line = QPen(Qt.black, 1)
        pen_bold_line = QPen(Qt.black, 3)
        qp.setRenderHints(qp.Antialiasing)

        # draw contour
        x_left = self.buttons[0][0].geometry().left() - self.spacing
        x_right = self.buttons[0][-1].geometry().right() + self.spacing
        y_top = self.buttons[0][0].geometry().top() - self.spacing
        y_bottom = self.buttons[-1][0].geometry().bottom() + self.spacing

        qp.setPen(pen_bold_line)
        qp.drawRect(x_left, y_top, x_right - x_left, y_bottom - y_top)

        for ind in range(self.rows - 1):
            if (ind % self.dimension) == (self.dimension - 1):
                qp.setPen(pen_bold_line)
            else:
                qp.setPen(pen_normal_line)

            y0 = self.buttons[ind][0].geometry().bottom()
            y1 = self.buttons[ind+1][0].geometry().top()
            y_middle = (y0 + y1) / 2 + 1
            qline = QLineF(x_left, y_middle, x_right, y_middle)
            qp.drawLine(qline)

        for ind in range(self.cols - 1):
            if (ind % self.dimension) == (self.dimension - 1):
                qp.setPen(pen_bold_line)
            else:
                qp.setPen(pen_normal_line)

            x0 = self.buttons[0][ind].geometry().right()
            x1 = self.buttons[0][ind+1].geometry().left()
            x_middle = (x0 + x1) / 2 + 1
            qline = QLineF(x_middle, y_top, x_middle, y_bottom)
            qp.drawLine(qline)

    def reset(self, vals: list[list[int]]):
        for row in range(self.rows):
            for col in range(self.cols):
                btn = self.buttons[row][col]
                val = vals[row][col]
                # set initial empty text for place holder, so that the resizing doesn't get messed up
                if val < 0:
                    btn.setText(" ")


class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.dimension = 3

        layout = QVBoxLayout()
        buttons_layout = QHBoxLayout()

        self.sudoku_widget = SudokuWidget(self, self.dimension)
        layout.addWidget(self.sudoku_widget)
        layout.addLayout(buttons_layout)
        widget = QWidget()
        widget.setLayout(layout)
        self.setCentralWidget(widget)
        self.setWindowTitle("Sudoku")

        generate_button = QPushButton('New')
        solve_button = QPushButton('Solve')
        reset_button = QPushButton('Reset')

        generate_button.clicked.connect(self.onGenerate)
        solve_button.clicked.connect(self.sudoku_widget.showSolution)
        reset_button.clicked.connect(self.onReset)

        buttons_layout.addWidget(generate_button)
        buttons_layout.addWidget(solve_button)
        buttons_layout.addWidget(reset_button)

        menubar = QMenuBar()

        menu_file = menubar.addMenu("File")
        action_file_new = QAction("New", self, shortcut="Ctrl+N")
        action_file_new.triggered.connect(self.onGenerate)
        menu_file.addAction(action_file_new)
        action_file_solve = QAction("Solve", self, shortcut="Ctrl+E")
        action_file_solve.triggered.connect(partial(self.sudoku_widget.showSolution))
        menu_file.addAction(action_file_solve)
        action_file_reset = QAction("Reset", self, shortcut="Ctrl+R")
        action_file_reset.triggered.connect(self.onReset)
        menu_file.addAction(action_file_reset)
        action_file_check = QAction("Check", self, shortcut="Ctrl+H")
        action_file_check.triggered.connect(partial(self.sudoku_widget.check))
        menu_file.addAction(action_file_check)
        action_file_ucheck = QAction("Uncheck", self, shortcut="Ctrl+U")
        action_file_ucheck.triggered.connect(
            partial(self.sudoku_widget.uncheck))
        menu_file.addAction(action_file_ucheck)
        menubar.addMenu(menu_file)

        menu_option = menubar.addMenu("Option")
        action_option_tips = QAction(
            "Show Tips", self, shortcut="Ctrl+T", checkable=True)
        menu_option.addAction(action_option_tips)
        menu_option_dimension = menu_option.addMenu("Dimension")
        self.action_dimensions = {
            k: QAction(
                str(k), self, shortcut="Ctrl+{}".format(k), checkable=True)
            for k in [3, 4]
        }

        for k, v in self.action_dimensions.items():
            v.triggered.connect(partial(self.onDimension, k))
            menu_option_dimension.addAction(v)
        self.action_dimensions[self.dimension].setChecked(True)
        menubar.addMenu(menu_option)

        self.setMenuBar(menubar)

    @pyqtSlot()
    def onReset(self):
        self.sudoku_widget.reset()

    @pyqtSlot()
    def onGenerate(self):
        pass

    @pyqtSlot()
    def onDimension(self, dimension: int):
        if dimension != self.dimension:
            for k, v in self.action_dimensions.items():
                v.setChecked(k == dimension)
            self.sudoku_widget.setDimension(dimension)
            self.dimension = dimension


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.resize(700, 800)
    window.show()
    app.exec_()


# Only execute when run as main script (not when imported module)
if __name__ == '__main__':
    main()
