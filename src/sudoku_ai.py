from PyQt5 import QtCore, QtGui
from PyQt5.QtGui import QFont, QTransform, QPen, QPixmap, QPolygonF, QImage
from PyQt5.QtCore import QPointF, QRectF, pyqtSlot, QLineF
from PyQt5.QtWidgets import QApplication, QMainWindow, QGraphicsItem, QFileDialog, \
    QGraphicsView, QGraphicsScene, QGraphicsEllipseItem, QGraphicsPolygonItem, QGraphicsPixmapItem, \
    QMenuBar, QAction, QGraphicsSceneMouseEvent, QGraphicsLineItem
import sys
from functools import partial
import numpy as np
import typing
import cv2 as cv2
from os import path as osp
from sudoku_ai_model import MyLeNet5
from pathlib import Path
from skimage.segmentation import clear_border
import torch
from torchvision import transforms


class MoveableQuadrilateral(QGraphicsPolygonItem):
    def __init__(self, polygon: QtGui.QPolygonF):
        diameter = 10
        self.quad_points = [QPointF(point) for point in polygon]
        # sort it clock-wise order
        self.quad_points = [
            self.topLeft(),
            self.topRight(),
            self.bottomRight(),
            self.bottomLeft()
        ]
        super().__init__(QPolygonF(self.quad_points))
        self.point_items = [MoveablePoint(
            point.x() - diameter / 2., point.y() - diameter / 2., diameter, diameter, i, self)
            for i, point in enumerate(self.quad_points)]
        # update pen
        self.setPen(self.pen())

    def onPointMove(self, index: int, pos: QPointF):
        poly = self.polygon()
        self.quad_points[index] = pos
        poly[index].setX(pos.x())
        poly[index].setY(pos.y())
        self.setPolygon(poly)

    def setPen(self, pen: typing.Union[QtGui.QPen, QtGui.QColor, QtCore.Qt.GlobalColor, QtGui.QGradient]) -> None:
        pen = QPen(pen)
        pen.setCosmetic(True)
        super().setPen(pen)
        for point in self.point_items:
            point.setPen(pen)
            point.setBrush(pen.color())

    def getPair(self, side: str = 'left'):
        if side == 'left':
            def comp(a, b): return a.x() < b.x()
        else:
            def comp(a, b): return a.x() > b.x()

        pair = []
        points = self.quad_points.copy()
        pt1 = points[0]
        for pt in points[1:]:
            if comp(pt, pt1):
                pt1 = pt
        pair += [pt1]
        points.remove(pt1)

        pt2 = points[0]
        for pt in points[1:]:
            if comp(pt, pt2):
                pt2 = pt
        pair += [pt2]

        return pair

    def topLeft(self) -> QPointF:
        pt1, pt2 = self.getPair('left')
        return pt1 if pt1.y() < pt2.y() else pt2

    def topRight(self) -> QPointF:
        pt1, pt2 = self.getPair('right')
        return pt1 if pt1.y() < pt2.y() else pt2

    def bottomLeft(self) -> QPointF:
        pt1, pt2 = self.getPair('left')
        return pt1 if pt1.y() > pt2.y() else pt2

    def bottomRight(self) -> QPointF:
        pt1, pt2 = self.getPair('right')
        return pt1 if pt1.y() > pt2.y() else pt2


class MoveablePoint(QGraphicsEllipseItem):
    def __init__(self, x: float, y: float, w: float, h: float, index: int, parent: MoveableQuadrilateral):
        super().__init__(x, y, w, h, parent)
        self.prev_x = x
        self.prev_y = y
        self.index = index
        self.parent_polygon = parent
        self.clicked = False

    def mousePressEvent(self, event: 'QGraphicsSceneMouseEvent') -> None:
        self.clicked = True
        event.accept()

    def mouseMoveEvent(self, event: 'QGraphicsSceneMouseEvent') -> None:
        if self.clicked:
            point = event.scenePos()
            new_point = QPointF(point.x() - self.prev_x,
                                point.y() - self.prev_y)
            self.setPos(new_point)
            self.parent_polygon.onPointMove(self.index, point)
        event.accept()

    def mouseReleaseEvent(self, event: 'QGraphicsSceneMouseEvent') -> None:
        self.clicked = False
        self.prev_x = event.pos().x()
        self.prev_y = event.pos().y()
        event.accept()


class SudokuView(QGraphicsView):
    def __init__(self):
        super().__init__()

        self.grid_lines = []
        self.polygons = []
        self.quadrilateral = None
        self.cv_image = None
        self.file_name = ""

        base_path = Path(__file__).parent / Path("../res/DL")
        self.model = MyLeNet5(
            numch_out=16, numch_conv=[32, 64],
            transf=transforms.Normalize(0.1307, 0.3081)
        )
        #self.model.load_state_dict(torch.load(base_path / Path("model_1.pth")))
        self.model.load_state_dict(torch.load(base_path / Path("model_0.pth")))

        scene = QGraphicsScene()

        self.pixmap = QPixmap()
        self.pixmap_item = QGraphicsPixmapItem(self.pixmap)
        scene.addItem(self.pixmap_item)

        self.setScene(scene)
        self.setAcceptDrops(True)

    def dragEnterEvent(self, a0: QtGui.QDragEnterEvent) -> None:
        self.checkAcceptEvent(a0)

    def dragMoveEvent(self, a0: QtGui.QDragMoveEvent) -> None:
        self.checkAcceptEvent(a0)

    def dropEvent(self, event: QtGui.QDropEvent) -> None:
        mimeData = event.mimeData()
        if mimeData.hasUrls():
            localFileName = mimeData.urls()[0].toLocalFile()
            self.file_name = localFileName
            self.setPicture(localFileName)

    def resizeEvent(self, event):
        self.fitInView(self.scene().sceneRect(), QtCore.Qt.KeepAspectRatio)

    def checkAcceptEvent(self, event: QtGui.QDropEvent):
        mimeData = event.mimeData()
        if mimeData.hasText():
            _, ext = osp.splitext(mimeData.text())
            ext = ext.lower()
            if ext in ('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif'):
                event.acceptProposedAction()

    def setPicture(self, file_name: str):
        try:
            self.file_name = file_name
            self.pixmap = QPixmap(file_name)
            self.drawReset()
        except:
            pass

    def removePolygons(self):
        for cnt in self.polygons:
            self.scene().removeItem(cnt)
        self.polygons = []

    def removeQuadrilateral(self):
        if self.quadrilateral is not None:
            self.scene().removeItem(self.quadrilateral)
            self.quadrilateral = None

    def removeGridLines(self):
        for line in self.grid_lines:
            self.scene().removeItem(line)
        self.grid_lines = []

    def fitSceneToPixmap(self, pixmap):
        self.scene().setSceneRect(QRectF(pixmap.rect()))
        self.fitInView(self.scene().sceneRect(), QtCore.Qt.KeepAspectRatio)

    def drawReset(self):
        self.cv_image = cv2.imread(self.file_name)
        self.removeQuadrilateral()
        self.removeGridLines()
        self.removePolygons()
        self.drawPixmap(self.pixmap)

    def drawPixmap(self, pixmap):
        self.area_image = pixmap.height() * pixmap.width()
        self.pixmap_item.setPixmap(pixmap)
        self.fitSceneToPixmap(pixmap)

    def cvMatToQImage(self, inMat: np.ndarray):
        height = inMat.shape[0]
        width = inMat.shape[1]
        dim = inMat.ndim
        if dim == 3:
            return QImage(inMat.data, width, height, 3 * width, QImage.Format_RGB888).rgbSwapped()
        else:
            return QImage(inMat.data, width, height, width, QImage.Format_Grayscale8)

    def cropToContourAndDraw(self):
        if self.quadrilateral:
            top_left = self.quadrilateral.topLeft()
            top_right = self.quadrilateral.topRight()
            bottom_left = self.quadrilateral.bottomLeft()
            bottom_right = self.quadrilateral.bottomRight()
            pts1 = np.float32([
                [top_left.x(), top_left.y()],
                [top_right.x(), top_right.y()],
                [bottom_left.x(), bottom_left.y()],
                [bottom_right.x(), bottom_right.y()],
            ])

            average_width = (
                (top_right.x() - top_left.x()) + (bottom_right.x() - bottom_left.x())) / 2.
            average_heigth = (
                (bottom_left.y() - top_right.y()) + (bottom_left.y() - top_left.y())) / 2.

            average_side = (average_width + average_heigth) / 2.
            pts2 = np.float32([
                [0., 0.],
                [average_side, 0.],
                [0., average_side],
                [average_side, average_side]
            ])

            M = cv2.getPerspectiveTransform(pts1, pts2)
            dsize = (int(average_side), int(average_side))
            self.cv_image = cv2.warpPerspective(self.cv_image, M, dsize)

            pixmap = QPixmap.fromImage(self.cvMatToQImage(self.cv_image))
            self.removeQuadrilateral()
            self.drawPixmap(pixmap)

    def findContours(self, image):
        simple_thresh = False
        constrast_correction = False
        alpha = 1.3
        beta = 20
        # convert the image to grayscale format
        img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        if constrast_correction:
            img_gray = cv2.convertScaleAbs(
                img_gray, alpha=alpha, beta=beta)

        if simple_thresh:
            img_blurred = cv2.GaussianBlur(img_gray, (3, 3), 0)
            _, img_thresh = cv2.threshold(
                img_blurred, 120, 255, cv2.THRESH_BINARY)
        else:
            img_blurred = cv2.GaussianBlur(img_gray, (7, 7), 7)
            img_thresh = cv2.adaptiveThreshold(
                img_blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 2)

        contours, hierarchy = cv2.findContours(
            image=img_thresh, mode=cv2.RETR_LIST, method=cv2.CHAIN_APPROX_NONE)

        print("Number of contours detected:", len(contours))

        return (contours, hierarchy)

    def findPolygons(self, image: np.ndarray, pct_arc=0.001, pct_area=0.,
                     min_points=1, max_points=sys.maxsize, closed=False):
        polygons = []
        if image is not None:
            contours, _ = self.findContours(image)
            polygons = []
            for cnt in contours:
                arc_len = cv2.arcLength(cnt, closed)
                poly = cv2.approxPolyDP(cnt, pct_arc * arc_len, closed)
                if len(poly) >= min_points and len(poly) <= max_points:
                    w, h, _ = image.shape
                    img_area = w * h
                    poly_area = cv2.contourArea(poly)
                    if poly_area >= (pct_area * img_area):
                        # squeeze the array by removing useless dimension
                        poly = [(pt[0][0], pt[0][1]) for pt in poly]
                        polygons += [poly]
            print("Number of polygons detected:", str(len(polygons)))
        return polygons

    def findLines(self, image, th=np.pi/2.):
        # convert the image to grayscale format
        img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        magic_num = 3
        img_blurred = cv2.GaussianBlur(
            img_gray, (magic_num, magic_num), magic_num)
        # img_blurred = cv2.medianBlur(img_gray, magic_num)
        img_thresh = cv2.adaptiveThreshold(
            img_blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, magic_num, 3)

        # Edge detection
        dst = cv2.Canny(img_thresh, 50, 200, None, 7, False)

        # pixmap = QPixmap.fromImage(self.cvMatToQImage(img_thresh))
        # self.drawPixmap(pixmap)

        #  Standard Hough Line Transform
        lines = cv2.HoughLines(dst, 1, th, 200, None, 0, 0)
        lines_polar = []
        if lines is not None:
            for i in range(0, len(lines)):
                rho = lines[i][0][0]
                theta = lines[i][0][1]
                lines_polar += [(rho, theta)]

        lines_polar = sorted(
            lines_polar, key=lambda tup: (tup[1], tup[0]))

        print(f"Number of lines detected: {len(lines_polar)}")

        return lines_polar

    def findGridQuads(self, image):
        w, h = image.shape[:2]
        min_dim = min(image.shape[:2])
        theta_margin = 3. * (np.pi / 180.)
        rho_margin = max(10., min_dim * 0.03)
        # Find vertical and horizontal lines
        lines_polar = self.findLines(image, np.pi/2.)
        # Get rid of almost parallel lines that are too close from each other
        lines_polar = self.linesSingleOut(
            lines_polar, theta_margin, rho_margin)

        # split vertical and horizontal lines
        # they are grouped into similar angles, given the theta margin
        min_angle = np.pi / 2. - theta_margin
        max_angle = np.pi / 2. + theta_margin
        v_lines = [line for line in lines_polar
                   if line[1] <= max_angle and line[1] >= min_angle]
        min_angles = (0., np.pi - theta_margin)
        max_angles = (theta_margin, np.pi)
        h_lines = [line for line in lines_polar
                   if (line[1] <= max_angles[0] and line[1] >= min_angles[0]) or
                      (line[1] <= max_angles[1] and line[1] >= min_angles[1])]
        # calculate the distances between lines
        v_line_dist = np.diff(np.array(v_lines)[:, 0])
        h_line_dist = np.diff(np.array(h_lines)[:, 0])

        # as we are interested in a roughly square grid, we unify
        # one distance for horizontal and one for vertical lines.
        # we take the median because there is still some risk of outliers
        # hanging around, even after singling out the almost parallel lines
        h_delta = np.median(h_line_dist)
        v_delta = np.median(v_line_dist)
        # calculate the number of steps. We could have some
        # extra robustness here as the sudoku grids are always
        # perfect squares. Something in the lines of
        # h_steps = pow(np.sqrt(round(w / h_delta)), 2.)
        # v_steps = pow(np.sqrt(round(h / v_delta)), 2.)
        # However, this doesn't seem to be necessary until now...
        h_steps = int(round(w / h_delta))
        v_steps = int(round(h / v_delta))

        grid_polys = []
        for i in range(int(h_steps)):
            for j in range(int(v_steps)):
                top_left = (i * h_delta, j * v_delta)
                top_right = ((i+1) * h_delta, j * v_delta)
                bottom_left = (i * h_delta, (j+1) * v_delta)
                bottom_right = ((i+1) * h_delta, (j+1) * v_delta)
                grid_polys += [[top_left, top_right,
                                bottom_right, bottom_left]]

        return grid_polys

    def findGridNumbers(self, image):
        polys = self.findGridQuads(self.cv_image)
        # limit the amount of classes for 9x9, 16x16 or 25x25 grids
        if len(polys) == 81:
            class_range = slice(1,10,1)
        elif len(polys) == 256:
            class_range = slice(0,15,1)
        elif len(polys) == 625:
            class_range = slice(1,25,1)
        else:
            class_range = slice(0,len(polys),1)
        # convert the image to grayscale format
        img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        self.model.eval()
        numbers = {}
        with torch.inference_mode():
            for poly in polys:
                tl_x, tl_y = poly[0]
                tr_x, tr_y = poly[1]
                br_x, br_y = poly[2]
                bl_x, bl_y = poly[3]
                w = tr_x - tl_x
                h = bl_y - tl_y
                cx = tl_x + w / 2.
                cy = tl_y + h / 2.
                crop_img = img_gray[int(tl_y):int(bl_y), int(tl_x):int(tr_x)]
                crop_img = cv2.resize(crop_img, self.model.shape_in)
                #crop_img = cv2.GaussianBlur(crop_img,(5,5),0)
                crop_img = cv2.threshold(crop_img, 0, 255,
                                         cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
                crop_img = clear_border(crop_img)
                density = crop_img.sum() / np.prod(self.model.shape_in) / 255.
                print(f"Density = {density * 100}%")
                if density > 0.05:
                    x = crop_img.astype(np.float32) / 255.
                    x = np.array([[x]])
                    res = self.model(torch.from_numpy(x))
                    res = res[0][class_range]
                    num = res.argmax().item() + class_range.start
                    print(f"Guessed number = {num}")
                    cv2.imshow("Cell Thresh", crop_img)
                    cv2.waitKey(0)
                    numbers[(cx, cy)] = num
        return numbers

    def linesSingleOut(self, lines, theta_margin=3.*(np.pi / 180.), rho_margin=10.):
        # Get rid of almost parallel lines that are too close from each other
        i = 0
        while i < len(lines):
            inds_to_delete = []
            base_rho, base_theta = lines[i]
            j = i + 1
            for (rho, theta) in lines[i+1:]:
                # since line_points_polar is sorted by increasing values of theta and rho, we
                # can break the search as soon as we find the first entry that doesn't fit the
                # margins for rho and theta
                if abs(rho - base_rho) > rho_margin or abs(theta - base_theta) > theta_margin:
                    break
                j += 1
            if j > (i + 1):
                inds_to_delete = list(range(i+1, j))
                avg_rho, avg_theta = np.average(lines[i:j], 0)
                lines[i] = (avg_rho, avg_theta)
                for ind in reversed(inds_to_delete):
                    del lines[ind]
            i += 1

        print(f"Singled out {len(lines)} lines")
        return lines

    def linesConvertPolarToPoint(self, lines_polar, scale=10000):
        line_points = []
        for rho, theta in lines_polar:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            pt1 = (int(x0 + scale*(-b)), int(y0 + scale*(a)))
            pt2 = (int(x0 - scale*(-b)), int(y0 - scale*(a)))
            line_points += [(pt1, pt2)]
        return line_points

    def drawGridQuads(self):
        polys = self.findGridQuads(self.cv_image)
        self.drawPolygons(polys)

    def drawGridNumbers(self):
        nums = self.findGridNumbers(self.cv_image)
        self.drawNumbers(nums)

    def drawQuads(self):
        polys = self.findPolygons(self.cv_image, 0.1, 0.001, 4, 4, True)
        self.drawPolygons(polys)

    def drawAllPolygons(self):
        polys = self.findPolygons(self.cv_image)
        self.drawPolygons(polys)

    def drawNumbers(self, nums):
        pass

    def drawPolygons(self, polygons):
        self.removePolygons()
        for points in polygons:
            points = [QPointF(point[0], point[1]) for point in points]
            # polygon
            poly = QGraphicsPolygonItem(QPolygonF(points))
            pen = QPen(QtCore.Qt.green, 2)
            pen.setCosmetic(True)
            poly.setPen(pen)
            self.scene().addItem(poly)
            self.polygons += [poly]

    def drawMainQuadrilateral(self):
        self.removeQuadrilateral()
        main_contour = self.findMainQuadrilateral(self.cv_image)
        # polygon
        self.quadrilateral = MoveableQuadrilateral(main_contour)
        self.quadrilateral.setPen(QPen(QtCore.Qt.green, 5))
        self.scene().addItem(self.quadrilateral)
        opa = 11

    def findMainQuadrilateral(self, image: np.ndarray, pct_area: float = 0.1, pct_arc: float = 0.1):
        if image is not None:
            contours, _ = self.findContours(image)
            height, width, _ = image.shape
            area_image = height * width
            quads = {}
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area >= pct_area * area_image:
                    poly = cv2.approxPolyDP(
                        cnt, pct_arc * cv2.arcLength(cnt, True), True)

                    # Get all quadrilaterals. The contour hierarchy is
                    # defined as follows [Next, Previous, First_Child, Parent]
                    # TODO: should it be strictly equal?
                    if len(poly) >= 4:
                        quads[area] = poly
            print(f"Number of quadrilaterals detected: {len(quads)}")

            # sort keys according to the area, from biggest to smallest
            quads_sorted_keys = sorted(quads)
            quads_sorted_keys.reverse()
            # create Qt points and polygons for each of the contours filtered from above
            points = [
                [QPointF(pt[0][0], pt[0][1]) for pt in quads[key]]
                for key in quads_sorted_keys
            ]
            polygons = [QPolygonF(pts) for pts in points]
            inds_to_delete = set()
            # Remove quarilaterals that contain quadrilateral children
            # We are searching from biggest to smallest - 1, as the smallest cannot contain
            # any of the bigger ones
            for i, poly in enumerate(polygons[:-1]):
                for pts in points[1:]:
                    # A polygon only contains another one if it contains all of its points
                    contains_all = True
                    for pt in pts:
                        if not poly.containsPoint(pt, QtCore.Qt.OddEvenFill):
                            contains_all = False
                            break
                    if contains_all:
                        # found parent polygon, add index for deletion
                        inds_to_delete.add(i)
                        # we break as poly contains at least one quadrilateral child
                        break
            # we reverse the list so that the indices are not shifted during deletion
            for ind in reversed(list(inds_to_delete)):
                del polygons[ind]

            print(f"Out of which {len(polygons)} are single")

        # return the biggest single quadrilateral
        return polygons[0]


class MainWindow(QMainWindow):

    def __init__(self) -> None:
        super().__init__()
        self.setAcceptDrops(True)

        self.sdk_view = SudokuView()

        self.setCentralWidget(self.sdk_view)
        self.setWindowTitle("Sudoku")

        menubar = QMenuBar()

        menu_file = menubar.addMenu("File")
        action_file_open = QAction("Open", self)
        action_file_open.setShortcut("Ctrl+O")
        action_file_open.triggered.connect(self.onOpen)
        menu_file.addAction(action_file_open)
        menubar.addMenu(menu_file)

        menu_edit = menubar.addMenu("Edit")

        action_edit_reset = QAction("Reset", self)
        action_edit_reset.setShortcut("Ctrl+R")
        action_edit_reset.triggered.connect(
            partial(self.sdk_view.drawReset))
        menu_edit.addAction(action_edit_reset)

        action_edit_draw_poly = QAction("Draw Polygons", self)
        action_edit_draw_poly.setShortcut("Ctrl+A")
        action_edit_draw_poly.triggered.connect(
            partial(self.sdk_view.drawAllPolygons))
        menu_edit.addAction(action_edit_draw_poly)

        action_edit_draw_quad = QAction("Draw Quadrilaterals", self)
        action_edit_draw_quad.setShortcut("Ctrl+Q")
        action_edit_draw_quad.triggered.connect(
            partial(self.sdk_view.drawQuads))
        menu_edit.addAction(action_edit_draw_quad)

        action_edit_fc = QAction("Draw Contour", self)
        action_edit_fc.setShortcut("Ctrl+1")
        action_edit_fc.triggered.connect(
            partial(self.sdk_view.drawMainQuadrilateral))
        menu_edit.addAction(action_edit_fc)

        action_edit_crop = QAction("Crop", self)
        action_edit_crop.setShortcut("Ctrl+2")
        action_edit_crop.triggered.connect(
            partial(self.onCrop))
        menu_edit.addAction(action_edit_crop)

        action_edit_draw_grids = QAction("Draw Grid Cells", self)
        action_edit_draw_grids.setShortcut("Ctrl+3")
        action_edit_draw_grids.triggered.connect(
            partial(self.sdk_view.drawGridQuads))
        menu_edit.addAction(action_edit_draw_grids)

        action_edit_draw_nums = QAction("Draw NUmbers", self)
        action_edit_draw_nums.setShortcut("Ctrl+4")
        action_edit_draw_nums.triggered.connect(
            partial(self.sdk_view.drawGridNumbers))
        menu_edit.addAction(action_edit_draw_nums)

        menubar.addMenu(menu_edit)

        self.setMenuBar(menubar)

    def onOpen(self) -> None:
        """
        TODO: description
        """
        fileName = QFileDialog.getOpenFileName(self,
                                               "Open Image", "", "Image Files (*.png *.jpg *.bmp)")
        if fileName:
            self.openFile(fileName[0])

    def onCrop(self):
        self.sdk_view.cropToContourAndDraw()

    def openFile(self, file_name: str):
        self.sdk_view.setPicture(file_name)


def main() -> None:
    app = QApplication(sys.argv)
    window = MainWindow()
    window.resize(700, 800)
    window.show()
    app.exec_()


if __name__ == '__main__':
    main()
