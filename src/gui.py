
from PyQt5 import QtGui
from PyQt5.QtGui import QFont, QPainter, QPen, QCursor
from PyQt5.QtCore import Qt, QRectF, QLineF, QSize, pyqtSlot
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, \
    QBoxLayout, QGridLayout, QPushButton, QWidget, QMenu, QSizePolicy, QSpacerItem, \
    QMenuBar, QAction
import sys
from functools import partial


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
    def __init__(self, parent, dimension=3):
        super().__init__(parent)
        self.mainLayout = QBoxLayout(QBoxLayout.LeftToRight, self)

        self.sudoku = SudokuGrid(dimension)
        self.mainLayout.addItem(QSpacerItem(0, 0))
        self.mainLayout.addWidget(self.sudoku)
        self.mainLayout.addItem(QSpacerItem(0, 0))
        self.setLayout(self.mainLayout)

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


class SudokuGrid(QWidget):
    def __init__(self, dimension=3, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setMinimumSize(400, 400)
        self.dimension = dimension
        self.cols = dimension * dimension
        self.rows = dimension * dimension

        self.sudoku_elements = [
            [0, 0, 4, 3, 0, 0, 2, 0, 9],
            [0, 0, 5, 0, 0, 9, 0, 0, 1],
            [0, 7, 0, 0, 6, 0, 0, 4, 3],
            [0, 0, 6, 0, 0, 2, 0, 8, 7],
            [1, 9, 0, 0, 0, 7, 4, 0, 0],
            [0, 5, 0, 0, 8, 3, 0, 0, 0],
            [6, 0, 0, 0, 0, 0, 1, 0, 5],
            [0, 0, 3, 5, 0, 8, 6, 9, 0],
            [0, 4, 2, 9, 1, 0, 3, 0, 0],
        ]

        self.gridLayout = QGridLayout()
        self.buttons = [
            [QPushButton() for _ in range(self.cols)]
            for _ in range(self.rows)
        ]

        for btn_row in self.buttons:
            for btn in btn_row:
                btn.clicked.connect(partial(self.onBtnClicked, btn))

        for row in range(self.rows):
            for col in range(self.cols):
                btn = self.buttons[row][col]
                val = self.sudoku_elements[row][col]
                # get the buttons to stretch both horizontally and vertically
                size_policy = QSizePolicy(
                    QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Minimum)
                size_policy.setVerticalStretch(1)
                size_policy.setHorizontalStretch(1)
                btn.setSizePolicy(size_policy)
                btn.setFont(QFont("Courrier New", 15))
                # set initial empty text for place holder, so that the resizing doesn't get messed up
                if val > 0:
                    btn.setText(str(val))
                    btn.setEnabled(False)
                else:
                    btn.setText(" ")

                self.gridLayout.addWidget(btn, row, col)

        self.spacing = 5
        self.gridLayout.setSpacing(self.spacing)
        self.setLayout(self.gridLayout)

    @ pyqtSlot()
    def onBtnClicked(self, item: QPushButton):
        menu = SudokuInsertNumberMenu(item)
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
            if (ind % self.dimension) == 2:
                qp.setPen(pen_bold_line)
            else:
                qp.setPen(pen_normal_line)

            y0 = self.buttons[ind][0].geometry().bottom()
            y1 = self.buttons[ind+1][0].geometry().top()
            y_middle = (y0 + y1) / 2 + 1
            qline = QLineF(x_left, y_middle, x_right, y_middle)
            qp.drawLine(qline)

        for ind in range(self.cols - 1):
            if (ind % self.dimension) == 2:
                qp.setPen(pen_bold_line)
            else:
                qp.setPen(pen_normal_line)

            x0 = self.buttons[0][ind].geometry().right()
            x1 = self.buttons[0][ind+1].geometry().left()
            x_middle = (x0 + x1) / 2 + 1
            qline = QLineF(x_middle, y_top, x_middle, y_bottom)
            qp.drawLine(qline)

        return
        width = self.squareSize * self.cols
        height = self.squareSize * self.rows
        # center the grid
        left = (self.width() - width) / 2
        top = (self.height() - height) / 2
        y = top
        # we need to add 1 to draw the topmost right/bottom lines too
        for ind in range(self.rows + 1):
            if (ind % self.dimension) == 0:
                qp.setPen(pen_bold_line)
            else:
                qp.setPen(pen_normal_line)
            qline = QLineF(left, y, left + width, y)
            qp.drawLine(qline)
            y += self.squareSize
        x = left
        for ind in range(self.cols + 1):
            if (ind % self.dimension) == 0:
                qp.setPen(pen_bold_line)
            else:
                qp.setPen(pen_normal_line)
            qp.drawLine(QLineF(x, top, x, top + height))
            x += self.squareSize

        # set "normal" pen back again
        qp.setPen(pen_normal_line)

        # create a smaller rectangle
        objectSize = self.squareSize * .8
        margin = self.squareSize * .1
        objectRect = QRectF(margin, margin, objectSize, objectSize)

        qp.setBrush(Qt.blue)
        font = QFont("Courrier New")
        font.setPixelSize(int(self.squareSize / 2))
        qp.setFont(font)
        center_offset = self.squareSize / 2 - font.pixelSize() / 2
        for row, row_list in enumerate(self.sudoku_elements):
            for col, el in enumerate(row_list):
                qp.drawText(objectRect.translated(
                    left + col * self.squareSize + center_offset,
                    top + row * self.squareSize + center_offset / 2),
                    str(el)
                )

    def reset(self):
        for row in range(self.rows):
            for col in range(self.cols):
                btn = self.buttons[row][col]
                val = self.sudoku_elements[row][col]
                # set initial empty text for place holder, so that the resizing doesn't get messed up
                if val == 0:
                    btn.setText(" ")


class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        dimension = 3

        layout = QVBoxLayout()
        buttons_layout = QHBoxLayout()

        generate_button = QPushButton('New')
        solve_button = QPushButton('Solve')
        reset_button = QPushButton('Reset')

        generate_button.clicked.connect(self.onGenerate)
        solve_button.clicked.connect(self.onSolve)
        reset_button.clicked.connect(self.onReset)

        buttons_layout.addWidget(generate_button)
        buttons_layout.addWidget(solve_button)
        buttons_layout.addWidget(reset_button)

        self.sudoku_widget = SudokuWidget(self, dimension)
        layout.addWidget(self.sudoku_widget)
        layout.addLayout(buttons_layout)
        widget = QWidget()
        widget.setLayout(layout)
        self.setCentralWidget(widget)
        self.setWindowTitle("Sudoku")

        menubar = QMenuBar()

        menu_file = menubar.addMenu("File")
        action_file_new = QAction("New", self, shortcut="Ctrl+N")
        action_file_new.triggered.connect(self.onGenerate)
        menu_file.addAction(action_file_new)
        action_file_solve = QAction("Solve", self, shortcut="Ctrl+E")
        action_file_solve.triggered.connect(self.onSolve)
        menu_file.addAction(action_file_solve)
        action_file_reset = QAction("Reset", self, shortcut="Ctrl+R")
        action_file_reset.triggered.connect(self.onReset)
        menu_file.addAction(action_file_reset)
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
        self.action_dimensions[dimension].setChecked(True)
        menubar.addMenu(menu_option)

        self.setMenuBar(menubar)

    @pyqtSlot()
    def onReset(self):
        self.sudoku_widget.sudoku.reset()

    @pyqtSlot()
    def onSolve(self):
        pass

    @pyqtSlot()
    def onGenerate(self):
        pass

    @pyqtSlot()
    def onDimension(self, dimension: int):
        for k, v in self.action_dimensions.items():
            v.setChecked(k == dimension)


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.resize(700, 800)
    window.show()
    app.exec_()


# Only execute when run as main script (not when imported module)
if __name__ == '__main__':
    main()
