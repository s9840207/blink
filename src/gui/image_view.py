import sys
import time
from collections import namedtuple
from pathlib import Path

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtQuickWidgets, QtWidgets
from PySide6.QtCore import Qt
from PySide6.QtQuick import QQuickWindow, QSGRendererInterface

import blink.image_processing as imp
from blink import mapping

pg.setConfigOption("background", "w")
pg.setConfigOption("foreground", "k")

AOI_WIDTH_STR = "AOI width"
SPOT_DIA_STR = "Spot diameter"
NOISE_DIA_STR = "Noise diameter"
SPOT_BRIGHTNESS_STR = "Spot brightness"
DIST_STR = "Distance threshold"
ValueRange = namedtuple("ValueRange", ("min", "max", "step"))
SPOT_PARAMS_RANGE = {
    SPOT_DIA_STR: ValueRange(1, 99, 2),
    NOISE_DIA_STR: ValueRange(0, 10, 1),
    SPOT_BRIGHTNESS_STR: ValueRange(0, 1000, 1),
    AOI_WIDTH_STR: ValueRange(1, 100, 1),
    DIST_STR: ValueRange(0, 100, 1),
}

PROPERTY_NAME_ROLE = Qt.UserRole + 1


class MyImageView(pg.ImageView):
    coord_get = QtCore.Signal(tuple)
    change_aois_state = QtCore.Signal(str)
    remove_close_aoi = QtCore.Signal()
    remove_empty_aoi = QtCore.Signal()
    remove_occupied_aoi = QtCore.Signal()
    load_aois = QtCore.Signal()
    save_aois = QtCore.Signal()
    frame_changed_notify = QtCore.Signal()
    frame_average_changed = QtCore.Signal(int)
    aois_changed_notify = QtCore.Signal()
    map = QtCore.Signal()
    inverseMap = QtCore.Signal()
    mapping_changed_notify = QtCore.Signal()
    sequence_set = QtCore.Signal()

    def __init__(self):
        super().__init__()
        self.view_box = self.getView()
        self.model = Model()
        self.aois_view = AoisView(self.view_box, self.model)
        self.view_box.setAspectLocked(lock=True)
        self.aois_view.pick_aois.connect(
            self.model.pick_spots, QtCore.Qt.UniqueConnection
        )
        self.aois_view.gaussian_refine.connect(
            self.model.gaussian_refine_aois, QtCore.Qt.UniqueConnection
        )
        self.model.aois_changed.connect(
            self.aois_view.update, QtCore.Qt.UniqueConnection
        )
        self.model.aois_changed.connect(
            self.aois_changed_notify, QtCore.Qt.UniqueConnection
        )
        self.sigTimeChanged.connect(
            self.model.change_current_frame, QtCore.Qt.UniqueConnection
        )
        self.sigTimeChanged.connect(
            self.frame_changed_notify, QtCore.Qt.UniqueConnection
        )
        self.crossHairActive = False
        self.vLine = pg.InfiniteLine(angle=90, movable=False)
        self.hLine = pg.InfiniteLine(angle=0, movable=False)
        self.view_box.addItem(self.vLine, ignoreBounds=True)
        self.view_box.addItem(self.hLine, ignoreBounds=True)
        self.proxy = pg.SignalProxy(
            self.view_box.scene().sigMouseMoved, rateLimit=60, slot=self.mouseMoved
        )
        self.view_box.scene().sigMouseClicked.connect(self.onMouseClicked)
        self.coord_get.connect(self.model.process_new_coord)
        self.change_aois_state.connect(self.model.change_aois_state)
        self.remove_close_aoi.connect(self.model.remove_close_aoi)
        self.remove_empty_aoi.connect(self.model.remove_empty_aoi)
        self.remove_occupied_aoi.connect(self.model.remove_occupied_aoi)
        self.save_aois.connect(self.model.save_aois)
        self.load_aois.connect(self.model.load_aois)
        self.frame_average_changed.connect(self.model.update_frame_average)
        self.model.frame_average_changed.connect(self.onFrameAverageChanged)
        self.map.connect(self.model.map)
        self.inverseMap.connect(self.model.inverse_map)
        self.imageSequence = None

    def onFrameAverageChanged(self):
        self.tVals = np.arange(self.imageSequence.length - self.model.frame_average + 1)
        self.frameTicks.setXVals(self.tVals)
        if len(self.tVals) > 1:
            start = self.tVals.min()
            stop = self.tVals.max() + abs(self.tVals[-1] - self.tVals[0]) * 0.02
        elif len(self.tVals) == 1:
            start = self.tVals[0] - 0.5
            stop = self.tVals[0] + 0.5
        else:
            start = 0
            stop = 1
        for s in [self.timeLine, self.normRgn]:
            s.setBounds([start, stop])
        self.roiClicked()
        self.updateImage()

    def updateImage(self, autoHistogramRange=True):
        """Override the original"""
        # Redraw image on screen
        if self.image is None:
            return

        self.image = self.imageSequence.get_averaged_image(
            self.currentIndex, size=self.model.frame_average
        )
        image = self.normalize(self.image)
        self.imageDisp = image
        self._imageLevels = self.quickMinMax(self.imageDisp)
        self.levelMin = min([level[0] for level in self._imageLevels])
        self.levelMax = max([level[1] for level in self._imageLevels])

        if autoHistogramRange:
            self.ui.histogram.setHistogramRange(self.levelMin, self.levelMax)

        self.ui.roiPlot.show()

        self.imageItem.updateImage(image.T)

    def setSequence(self, image_sequence: imp.ImageSequence):
        self.imageSequence = image_sequence
        image = self.imageSequence.get_averaged_image(
            self.currentIndex, size=self.model.frame_average
        )
        self.view_box.setLimits(
            xMin=0, xMax=image_sequence.width, yMin=0, yMax=image_sequence.height
        )
        self.setImage(image, axes={"x": 1, "y": 0})
        self.axes = {"t": -1, "x": 1, "y": 0, "c": None}
        self.tVals = np.arange(self.imageSequence.length)

        # Set ticks
        self.currentIndex = 0
        self.frameTicks.setXVals(self.tVals)
        self.timeLine.setValue(0)
        if len(self.tVals) > 1:
            start = self.tVals.min()
            stop = self.tVals.max() + abs(self.tVals[-1] - self.tVals[0]) * 0.02
            stop = self.tVals.max()
        elif len(self.tVals) == 1:
            start = self.tVals[0] - 0.5
            stop = self.tVals[0] + 0.5
        else:
            start = 0
            stop = 1
        for s in [self.timeLine, self.normRgn]:
            s.setBounds([start, stop])

        self.updateImage()
        self.autoRange()
        self.roiClicked()
        self.sequence_set.emit()

    def play(self, rate=None):
        """Begin automatically stepping frames forward at the given rate (in fps).
        This can also be accessed by pressing the spacebar."""
        # This function was copied here to match the perf_counter() call in
        # timeout(), otherwise the perf_counter() in this package and imported
        # in the pyqtgraph.imageview.ImageView.py will time with different
        # reference points.
        if rate is None:
            rate = self.fps
        self.playRate = rate

        if rate == 0:
            self.playTimer.stop()
            return

        self.lastPlayTime = time.perf_counter()
        if not self.playTimer.isActive():
            self.playTimer.start(16)

    def evalKeyState(self):
        # This function was copied here to match the perf_counter() call in
        # timeout(), otherwise the perf_counter() in this package and imported
        # in the pyqtgraph.imageview.ImageView.py will time with different
        # reference points.
        if len(self.keysPressed) == 1:
            key = list(self.keysPressed.keys())[0]
            if key == QtCore.Qt.Key.Key_Right:
                self.play(20)
                self.jumpFrames(1)
                # effectively pause playback for 0.2 s
                self.lastPlayTime = time.perf_counter() + 0.2
            elif key == QtCore.Qt.Key.Key_Left:
                self.play(-20)
                self.jumpFrames(-1)
                self.lastPlayTime = time.perf_counter() + 0.2
            elif key == QtCore.Qt.Key.Key_Up:
                self.play(-100)
            elif key == QtCore.Qt.Key.Key_Down:
                self.play(100)
            elif key == QtCore.Qt.Key.Key_PageUp:
                self.play(-1000)
            elif key == QtCore.Qt.Key.Key_PageDown:
                self.play(1000)
        else:
            self.play(0)

    def timeout(self):
        now = time.perf_counter()
        dt = now - self.lastPlayTime
        if dt < 0:
            return
        n = int(self.playRate * dt)
        if n != 0:
            self.lastPlayTime += float(n) / self.playRate
            # The next line was modified for accessing the z length my custom
            # image stack.
            if self.currentIndex + n > self.tVals[-1] + 1:
                self.play(0)
            self.jumpFrames(n)

    def setCurrentIndex(self, ind):
        """Set the currently displayed frame index."""
        # The next line was modified for accessing the z length my custom
        # image stack.
        index = pg.functions.clip_scalar(ind, 0, self.tVals[-1])
        self.ignorePlaying = True
        # Implicitly call timeLineChanged
        self.timeLine.setValue(self.tVals[index])
        self.ignorePlaying = False

    @QtCore.Slot()
    def onPickButtonPressed(self):
        self.aois_view.pick_aois.emit()

    @QtCore.Slot()
    def onFitButtonPressed(self):
        self.aois_view.gaussian_refine.emit()

    @QtCore.Slot()
    def onAddButtonPressed(self):
        self.crossHairActive = True
        self.change_aois_state.emit("add")

    @QtCore.Slot()
    def onRemoveButtonPressed(self):
        self.crossHairActive = True
        self.change_aois_state.emit("remove")

    @QtCore.Slot()
    def onRemoveCloseButtonPressed(self):
        self.remove_close_aoi.emit()

    @QtCore.Slot()
    def loadMapping(self):
        load_path = open_file_path_dialog()
        if load_path is not None:
            load_path = Path(load_path)
            self.model.load_mapping(load_path)
            self.mapping_changed_notify.emit()

    def onMouseClicked(self, event):
        if self.crossHairActive:
            button = event.button()
            if button == Qt.MouseButton.LeftButton:
                point: QtCore.QPointF = self.view_box.mapSceneToView(event.scenePos())
                coord = (point.x(), point.y())
                self.coord_get.emit(coord)
            elif button == Qt.MouseButton.RightButton:
                self.crossHairActive = False
                self.hLine.setValue(0)
                self.vLine.setValue(0)
                self.change_aois_state.emit("idle")

    def mouseMoved(self, evt):
        pos = evt[0]  # using signal proxy turns original arguments into a tuple
        if self.crossHairActive and self.view_box.sceneBoundingRect().contains(pos):
            mousePoint = self.view_box.mapSceneToView(pos)
            self.vLine.setPos(mousePoint.x())
            self.hLine.setPos(mousePoint.y())

    def _getCurrentIndex(self):
        return self.currentIndex

    def _getAoiCount(self):
        return self.model.get_aoi_counts()

    def _getMapMatrix(self):
        return self.model.mapper_string()

    def _getNFrames(self):
        if self.imageSequence is None:
            return 0
        return self.imageSequence.length

    idx = QtCore.Property(int, fget=_getCurrentIndex, notify=frame_changed_notify)
    aoiCount = QtCore.Property(int, fget=_getAoiCount, notify=aois_changed_notify)
    mapMatrixString = QtCore.Property(
        str, fget=_getMapMatrix, notify=mapping_changed_notify
    )

    nFrames = QtCore.Property(int, fget=_getNFrames, notify=sequence_set)


def save_file_path_dialog() -> Path:
    set_qapplication()
    file_path, _ = QtWidgets.QFileDialog.getSaveFileName(caption="Save to file")
    if file_path == "":
        return None
    file_path = Path(file_path)
    return file_path


def open_file_path_dialog() -> Path:
    set_qapplication()
    file_path, _ = QtWidgets.QFileDialog.getOpenFileName(caption="Select file to open")
    if file_path == "":
        return None
    file_path = Path(file_path)
    return file_path


def select_directory_dialog() -> Path:
    set_qapplication()
    dir_path = QtWidgets.QFileDialog.getExistingDirectory(caption="Select a directory")
    if dir_path == "":
        return None
    dir_path = Path(dir_path)
    return dir_path


class Model(QtCore.QObject):

    aois_changed = QtCore.Signal()
    frame_average_changed = QtCore.Signal()

    def __init__(self):
        super().__init__()
        self._aois: imp.Aois = None
        self.image_sequence: imp.ImageSequence = None
        self._current_frame = 0
        self.pick_spots_param = PickSpotsParam()
        self.aois_edit_state = "idle"
        self.frame_average = 1
        self.mapper = None

    @property
    def aois(self):
        return self._aois

    @aois.setter
    def aois(self, value):
        self._aois = value
        self.aois_changed.emit()

    def get_coords(self):
        if self.aois is None:
            return (np.empty(1), np.empty(1))
        return (self.aois.get_all_x(), self.aois.get_all_y())

    def get_aoi_width(self):
        return self.pickSpotsParam.params[AOI_WIDTH_STR]

    def set_aois(self, aois):
        self.aois = aois

    def _pick_spots_wrapped(self):
        params = self.pick_spots_param.params
        aois = imp.pick_spots(
            self.image_sequence.get_averaged_image(
                start=self._current_frame, size=self.frame_average
            ),
            threshold=params[SPOT_BRIGHTNESS_STR],
            noise_dia=params[NOISE_DIA_STR],
            spot_dia=params[SPOT_DIA_STR],
            frame=self.current_frame,
            aoi_width=params[AOI_WIDTH_STR],
            frame_avg=1,
        )
        return aois

    def pick_spots(self):
        self.aois = self._pick_spots_wrapped()

    def get_aoi_counts(self):
        if self.aois is None:
            return 0
        return len(self.aois)

    @QtCore.Slot(int)
    def change_current_frame(self, new_frame_index: int):
        self._current_frame = new_frame_index

    @property
    def current_frame(self):
        return self._current_frame

    @current_frame.setter
    def current_frame(self, new_value: int):
        new_value = int(new_value)
        if new_value > len(self.image_sequence):
            new_value = len(self.image_sequence)
        elif new_value < 0:
            new_value = 0
        self._current_frame = new_value

    def _read_pick_spots_param(self):
        return self.pick_spots_param

    @QtCore.Signal
    def dummy_notify(self):
        pass

    def get_current_frame_image(self):
        return self.image_sequence.get_averaged_image(
            self._current_frame, size=self.frame_average
        )

    @QtCore.Slot()
    def gaussian_refine_aois(self):
        current_image = self.get_current_frame_image()
        self.aois = self.aois.gaussian_refine(image=current_image)

    @QtCore.Slot(tuple)
    def process_new_coord(self, coord: tuple):
        if self.aois_edit_state == "add":
            if self.aois is None:
                self.aois = imp.Aois(
                    np.array(coord)[np.newaxis, :], frame=self._current_frame
                )
            else:
                self.aois += coord
        elif self.aois is not None and self.aois_edit_state == "remove":
            self.aois = self.aois.remove_aoi_nearest_to_ref(coord)

    @QtCore.Slot(str)
    def change_aois_state(self, new_state: str):
        self.aois_edit_state = new_state

    def remove_close_aoi(self):
        dist_threshold = self.pickSpotsParam.params[DIST_STR]
        self.aois = self.aois.remove_close_aois(dist_threshold)

    def remove_empty_aoi(self):
        dist_threshold = self.pickSpotsParam.params[DIST_STR]
        ref_aois = self._pick_spots_wrapped()
        self.aois = self.aois.remove_aois_far_from_ref(ref_aois, radius=dist_threshold)

    def remove_occupied_aoi(self):
        dist_threshold = self.pickSpotsParam.params[DIST_STR]
        ref_aois = self._pick_spots_wrapped()
        self.aois = self.aois.remove_aois_near_ref(ref_aois, radius=dist_threshold)

    def save_aois(self):
        save_path = save_file_path_dialog()
        if save_path is not None:
            self.aois.to_npz(save_path)

    def load_aois(self):
        load_path = open_file_path_dialog()
        if load_path is not None:
            load_path = Path(load_path)
            if load_path.suffix == ".npz":
                self.aois = imp.Aois.from_npz(load_path)
            elif load_path.suffix == ".dat":
                self.aois = imp.Aois.from_imscroll_aoiinfo2(load_path)

    @QtCore.Slot(int)
    def update_frame_average(self, value: int):
        self.frame_average = value

    def _get_frame_average(self):
        return self.frame_average

    def _set_frame_average(self, value):
        self.frame_average = value
        self.frame_average_changed.emit()

    def load_mapping(self, path: Path):
        if path.suffix == ".dat":
            self.mapper = mapping.MapperBare.from_imscroll(path)
        elif path.suffix == ".npz":
            self.mapper = mapping.MapperBare.from_npz(path)

    def mapper_string(self):
        if self.mapper is None:
            return ""
        return repr(self.mapper.map_matrix)

    def map(self):
        if self.aois is not None and self.mapper is not None:
            self.aois = self.mapper.map(self.aois)

    def inverse_map(self):
        if self.aois is not None and self.mapper is not None:
            self.aois = self.mapper.inverse_map(self.aois)

    pickSpotsParam = QtCore.Property(
        QtCore.QObject, fget=_read_pick_spots_param, notify=dummy_notify
    )
    frameAverage = QtCore.Property(
        int, fget=_get_frame_average, fset=_set_frame_average, notify=dummy_notify
    )


class PickSpotsParam(QtCore.QObject):
    def __init__(self):
        super().__init__()
        self.params = {
            AOI_WIDTH_STR: 5,
            SPOT_DIA_STR: 5,
            NOISE_DIA_STR: 1,
            SPOT_BRIGHTNESS_STR: 50,
            DIST_STR: 5,
        }

    def _getAoiWidth(self):
        return self.params[AOI_WIDTH_STR]

    def _setAoiWidth(self, value):
        self.params[AOI_WIDTH_STR] = int(value)

    def _getSpotDia(self):
        return self.params[SPOT_DIA_STR]

    def _setSpotDia(self, value):
        self.params[SPOT_DIA_STR] = int(value)

    def _getNoiseDia(self):
        return self.params[NOISE_DIA_STR]

    def _setNoiseDia(self, value):
        self.params[NOISE_DIA_STR] = int(value)

    def _getSpotBrightness(self):
        return self.params[SPOT_BRIGHTNESS_STR]

    def _setSpotBrightness(self, value):
        self.params[SPOT_BRIGHTNESS_STR] = int(value)

    def _getDistThreshold(self):
        return str(self.params[DIST_STR])

    def _setDistThreshold(self, value):
        self.params[DIST_STR] = float(value)

    @QtCore.Signal
    def dummy_notify(self):
        pass

    aoiWidth = QtCore.Property(
        int, fget=_getAoiWidth, fset=_setAoiWidth, notify=dummy_notify
    )

    spotDia = QtCore.Property(
        int, fget=_getSpotDia, fset=_setSpotDia, notify=dummy_notify
    )

    noiseDia = QtCore.Property(
        int, fget=_getNoiseDia, fset=_setNoiseDia, notify=dummy_notify
    )

    spotBrightness = QtCore.Property(
        int, fget=_getSpotBrightness, fset=_setSpotBrightness, notify=dummy_notify
    )

    distThreshold = QtCore.Property(
        str, fget=_getDistThreshold, fset=_setDistThreshold, notify=dummy_notify
    )


class AoisView(QtCore.QObject):
    pick_aois = QtCore.Signal()
    gaussian_refine = QtCore.Signal()

    def __init__(self, view_box, model):
        super().__init__()
        self.marker = pg.ScatterPlotItem()
        self.marker.setBrush(255, 0, 0, 255)
        self.model = model
        self.update()
        view_box.addItem(self.marker)

    @QtCore.Slot()
    def update(self):
        coords = self.model.get_coords()
        aoi_width = self.model.get_aoi_width()
        # The coordinate of the view starts from the edge, so offsets 0.5
        self.marker.setData(
            coords[0] + 0.5,
            coords[1] + 0.5,
            symbol="s",
            pen=(0, 0, 255),
            brush=None,
            size=aoi_width,
            pxMode=False,
        )


class Window(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        layout = QtWidgets.QGridLayout()
        self.image_view = MyImageView()
        self.resize(640, 480)

        self.qml = QtQuickWidgets.QQuickWidget()
        # self.qml.setResizeMode(QtQuickWidgets.QQuickWidget.SizeRootObjectToView)
        qml_path = Path(__file__).parent / "qml/image_view.qml"
        root_context = self.qml.rootContext()
        root_context.setContextProperty("imageView", self.image_view)
        root_context.setContextProperty("dataModel", self.image_view.model)

        # Need to set context property before set source
        self.qml.setSource(QtCore.QUrl.fromLocalFile(str(qml_path)))

        layout.addWidget(self.image_view)
        layout.addWidget(self.qml, 0, 1)
        layout.setColumnStretch(0, 3)
        layout.setColumnStretch(1, 1)
        self.setLayout(layout)


def set_qapplication():
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])


def main():

    # Always start by initializing Qt (only once per application)
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])
    QQuickWindow.setGraphicsApi(QSGRendererInterface.OpenGL)
    path = select_directory_dialog()
    image_sequence = imp.ImageSequence(path)

    window = Window()
    window.image_view.setSequence(image_sequence)
    window.image_view.model.image_sequence = image_sequence
    window.show()

    # The following line is to delete the QQuickWidget before python exits. Otherwise, a
    # "TypeError: Cannot read property 'xxx' of null" will be thrown by QML because the
    # objects that have been made the QML's context properties could get deleted by
    # python before the QML, triggering a binding re-evaluation, thus the error.
    # See QTBUG-81247 comments.
    app.aboutToQuit.connect(window.qml.deleteLater)
    # Start the Qt event loop
    app.exec()
    sys.exit()


if __name__ == "__main__":
    main()
