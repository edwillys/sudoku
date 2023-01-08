
from PyQt5 import QtGui
from PyQt5.QtGui import QFont, QPainter, QPen, QCursor
from PyQt5.QtCore import Qt, QRectF, QLineF, QSize, pyqtSlot
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, \
    QBoxLayout, QGridLayout, QPushButton, QWidget, QMenu, QSizePolicy, QSpacerItem, \
    QMenuBar, QAction
import sys
from functools import partial
import numpy as np
from sudoku import Sudoku


class SudokuWidget(QWidget):
    def __init__(self, parent, order: int = 3):
        super().__init__(parent)
        self.setMinimumSize(400, 400)
        self.order = order
        self.puzzle = Sudoku(order)
        self.mainLayout = QBoxLayout(QBoxLayout.LeftToRight, self)
        self.grid = SquareBtnGrid(self)
        # the square button grid in between spacers, so that the spacers can
        # shrink / grow upon resizing, in order to keep the grid square
        self.mainLayout.addItem(QSpacerItem(0, 0))
        self.mainLayout.addWidget(self.grid)
        self.mainLayout.addItem(QSpacerItem(0, 0))
        self.setLayout(self.mainLayout)

        self.generatePuzzle(order)

        self.check_en = True
        self.tips_en = False

    def updateTips(self):
        """
        Updates the puzzle hints to the user
        """
        if self.tips_en:
            tips = self.getTips()
            # don't show tips for the elements that already contain right answers or the ones from the puzzle
            vals = self.grid.getVals()
            inds_right = np.argwhere(vals == self.solution_vals)
            for ind in inds_right:
                # https://numpy.org/doc/stable/user/quickstart.html#indexing-with-arrays-of-indices
                tips[tuple(ind)] = set()
            self.grid.updateToolTips(tips)

    def showTips(self, show: bool):
        """
        Whether to show puzzle hints to the user
        """
        self.tips_en = show
        if show:
            self.updateTips()
        else:
            self.grid.updateToolTips([])

    def resizeEvent(self, a0: QtGui.QResizeEvent) -> None:
        """
        QWidget overwrite in order to keep the main grid square
        To achieve that, we need to set the strecth factor of the spacers
        (elements with index 0 and 2) around the main grid widget (index 1)
        """
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

    def reset(self) -> None:
        """
        Resets the GUI with the original puzzle values
        """
        self.grid.reset(self.puzzle_vals)
        self.resetCheck()

    def showSolution(self) -> None:
        """
        Show the solution of the puzzle
        """
        self.grid.setVals(self.solution_vals)

    def updateCheck(self) -> None:
        """
        Update style of the user values, according to their correectness
        """
        if self.check_en:
            vals = self.grid.getVals()
            inds_wrong = np.argwhere((vals != self.solution_vals) & (vals > 0))
            inds_right = np.argwhere(
                (vals == self.solution_vals) & (self.puzzle_vals < 0))
            self.grid.setBtnStylesheetAt(inds_wrong, "background-color: red")
            self.grid.setBtnStylesheetAt(inds_right, "background-color: green")

    def resetCheck(self):
        inds = np.argwhere(self.puzzle_vals < 0)
        self.grid.setBtnStylesheetAt(inds, "")

    def setCheckEnable(self, enable: bool) -> None:
        """
        De/Activate style hints to the user entry values
        """
        self.check_en = enable
        if enable:
            self.updateCheck()
        else:
            self.resetCheck()

    def getTips(self) -> np.ndarray:
        """
        Interface to get the allowed values for each grid element
        """
        vals = np.array(self.grid.getVals())
        inds_wrong = np.argwhere((vals != self.solution_vals) & (vals > 0))
        for ind in inds_wrong:
            vals[tuple(ind)] = -1

        puzzle_now = Sudoku(self.order, grid=vals)
        tips = [
            [set(puzzle_now.allowed_vals(row, col))
             for col in range(self.order**2)]
            for row in range(self.order**2)
        ]

        return np.array(tips)

    def generatePuzzle(self, order: int) -> None:
        """
        Interface for generating the sudoku puzzle
        """
        self.order = order
        # generate puzzle
        self.puzzle.set_order(order)
        self.puzzle.generate_puzzle()
        self.puzzle_vals = np.array(self.puzzle.get_vals())
        self.grid.setVals(self.puzzle_vals)
        self.grid.enableElements(self.puzzle_vals > 0)
        self.grid.enableElements(self.puzzle_vals <= 0)
        self.resetCheck()
        # get solution
        self.solution = Sudoku(order, grid=self.puzzle_vals)
        self.solution.solve()
        self.solution_vals = np.array(self.solution.get_vals())


class SudokuInsertNumberMenu(QMenu):
    def __init__(self, parent: QPushButton, sudoku_widget: SudokuWidget, button_size: int = 50, order: int = 3, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.order = order
        self.sudoku_widget = sudoku_widget

        grid = QGridLayout()
        self.buttons = [QPushButton(str(i + 1))
                        for i in range(self.order ** 2)]

        for i, btn in enumerate(self.buttons):
            btn.setFixedSize(QSize(button_size, button_size))
            grid.addWidget(btn, i // order, i % order)
            btn.clicked.connect(partial(self.onBtnClicked, i, parent))

        grid.setSpacing(0)
        self.setLayout(grid)

    @ pyqtSlot()
    def onBtnClicked(self, number: int, parent: QPushButton) -> None:
        parent.setText(str(number))
        self.sudoku_widget.updateCheck()
        self.sudoku_widget.updateTips()
        self.close()


class SquareBtnGrid(QWidget):
    def __init__(self, sudoku_widget: SudokuWidget, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sudoku_widget = sudoku_widget
        self.spacing = 2
        self.gridLayout = QGridLayout()
        self.gridLayout.setSpacing(self.spacing)
        self.setLayout(self.gridLayout)

    def getVals(self) -> list[list[int]]:
        """
        TODO: description
        """
        vals = np.array([
            [-1 for _ in range(self.ncols)]
            for _ in range(self.nrows)
        ])
        for row, btn_row in enumerate(self.buttons):
            for col, btn in enumerate(btn_row):
                try:
                    vals[row][col] = int(btn.text())
                except:
                    pass

        return vals

    def setBtnStylesheetAt(self, positions: list[tuple[int, int]], stylesheet: str) -> None:
        """
        TODO: description
        """
        for row, col in positions:
            self.buttons[row][col].setStyleSheet(stylesheet)

    def enableElements(self, vals: list[list[bool]]):
        for row_ind, row in enumerate(vals):
            for col_ind, val in enumerate(row):
                self.buttons[row_ind][col_ind].setEnabled(val)

    def setVals(self, vals: list[list[int]]) -> None:
        """
        TODO: description
        """
        new_count = len(vals) ** 2
        cur_count = self.gridLayout.count()

        if new_count != cur_count:
            for i in reversed(range(self.gridLayout.count())):
                self.gridLayout.itemAt(i).widget().setParent(None)
            self.order = int(np.sqrt(len(vals)))
            self.ncols = self.order ** 2
            self.nrows = self.ncols

            self.buttons = [
                [QPushButton() for _ in range(self.ncols)]
                for _ in range(self.nrows)
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
                else:
                    btn.setText(" ")

    @ pyqtSlot()
    def onBtnClicked(self, item: QPushButton) -> None:
        """
        TODO: description
        """
        menu = SudokuInsertNumberMenu(
            item, self.sudoku_widget, order=self.order)
        menu.move(QCursor.pos())
        menu.show()

    def paintEvent(self, event) -> None:
        """
        TODO: description
        """
        qp = QPainter(self)
        pen_normal_line = QPen(Qt.black, 1)
        pen_bold_line = QPen(Qt.black, 3)
        qp.setRenderHints(qp.Antialiasing)

        # draw contour rectangle
        x_left = self.buttons[0][0].geometry().left() - self.spacing
        x_right = self.buttons[0][-1].geometry().right() + self.spacing
        y_top = self.buttons[0][0].geometry().top() - self.spacing
        y_bottom = self.buttons[-1][0].geometry().bottom() + self.spacing

        qp.setPen(pen_bold_line)
        qp.drawRect(x_left, y_top, x_right - x_left, y_bottom - y_top)

        # draw lines between the elements, with bold lines every `n` rows/columns
        for ind in range(self.nrows - 1):
            if (ind % self.order) == (self.order - 1):
                qp.setPen(pen_bold_line)
            else:
                qp.setPen(pen_normal_line)

            y0 = self.buttons[ind][0].geometry().bottom()
            y1 = self.buttons[ind+1][0].geometry().top()
            y_middle = (y0 + y1) / 2 + 1
            qline = QLineF(x_left, y_middle, x_right, y_middle)
            qp.drawLine(qline)

        for ind in range(self.ncols - 1):
            if (ind % self.order) == (self.order - 1):
                qp.setPen(pen_bold_line)
            else:
                qp.setPen(pen_normal_line)

            x0 = self.buttons[0][ind].geometry().right()
            x1 = self.buttons[0][ind+1].geometry().left()
            x_middle = (x0 + x1) / 2 + 1
            qline = QLineF(x_middle, y_top, x_middle, y_bottom)
            qp.drawLine(qline)

    def reset(self, vals: list[list[int]]) -> None:
        """
        Reset GUI elements with the values from `vals`
        """
        for row in range(self.nrows):
            for col in range(self.ncols):
                btn = self.buttons[row][col]
                val = vals[row][col]
                # set initial empty text for place holder, so that the resizing doesn't get messed up
                if val < 0:
                    btn.setText(" ")

    def updateToolTips(self, tips: list) -> None:
        """
        TODO: description
        """
        if len(tips) == 0:
            for btn_row in self.buttons:
                for btn in btn_row:
                    btn.setToolTip("")
        else:
            for tips_row, btn_row in zip(tips, self.buttons):
                for tip, btn in zip(tips_row, btn_row):
                    if len(tip) > 0:
                        btn.setToolTip(str(tip))
                    else:
                        btn.setToolTip("")


class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.order = 3

        layout = QVBoxLayout()
        buttons_layout = QHBoxLayout()

        self.sudoku_widget = SudokuWidget(self, self.order)
        layout.addWidget(self.sudoku_widget)
        layout.addLayout(buttons_layout)
        widget = QWidget()
        widget.setLayout(layout)
        self.setCentralWidget(widget)
        self.setWindowTitle("Sudoku")

        menubar = QMenuBar()

        menu_file = menubar.addMenu("File")
        action_file_new = QAction("New", self, shortcut="Ctrl+N")
        action_file_new.triggered.connect(
            partial(self.sudoku_widget.generatePuzzle, self.order))
        menu_file.addAction(action_file_new)
        action_file_solve = QAction("Solve", self, shortcut="Ctrl+E")
        action_file_solve.triggered.connect(
            partial(self.sudoku_widget.showSolution))
        menu_file.addAction(action_file_solve)
        action_file_reset = QAction("Reset", self, shortcut="Ctrl+R")
        action_file_reset.triggered.connect(partial(self.sudoku_widget.reset))
        menu_file.addAction(action_file_reset)
        menubar.addMenu(menu_file)

        menu_option = menubar.addMenu("Option")
        action_option_tips = QAction(
            "Show Tips", self, shortcut="Ctrl+T", checkable=True)
        action_option_tips.triggered.connect(self.sudoku_widget.showTips)
        menu_option.addAction(action_option_tips)
        action_file_check = QAction(
            "Check", self, shortcut="Ctrl+H", checkable=True, checked=True)
        action_file_check.triggered.connect(
            partial(self.sudoku_widget.setCheckEnable))
        menu_option.addAction(action_file_check)
        menu_option_order = menu_option.addMenu("Puzzle Order")
        self.action_orders = {
            k: QAction(
                str(k), self, shortcut="Ctrl+{}".format(k), checkable=True)
            for k in [3, 4]
        }

        for k, v in self.action_orders.items():
            v.triggered.connect(partial(self.onOrderChange, k))
            menu_option_order.addAction(v)
        self.action_orders[self.order].setChecked(True)
        menubar.addMenu(menu_option)

        self.setMenuBar(menubar)

    @pyqtSlot()
    def onOrderChange(self, order: int) -> None:
        """
        TODO: description
        """
        if order != self.order:
            for k, v in self.action_orders.items():
                v.setChecked(k == order)
            self.sudoku_widget.setPuzzleOrder(order)
            self.order = order


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.resize(700, 800)
    window.show()
    app.exec_()


# Only execute when run as main script (not when imported module)
if __name__ == '__main__':
    main()
