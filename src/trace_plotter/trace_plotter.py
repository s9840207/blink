#  Copyright (C) 2020 Tzu-Yu Lee, National Taiwan University
#
#  This file (trace_plotter.py) is part of blink.
#
#  blink is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  blink is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with blink.  If not, see <https://www.gnu.org/licenses/>.

"""This module is a GUI plotting tool for CoSMoS intensity data visualization.

This module reads """

import sys
import typing
from collections import namedtuple
from pathlib import Path

import numpy as np
import pandas as pd
import pyqtgraph as pg
from PySide6.QtCore import (
    Property,
    QAbstractListModel,
    QModelIndex,
    QObject,
    QStringListModel,
    Qt,
    QUrl,
    Signal,
    Slot,
)
from PySide6.QtQuick import QQuickWindow, QSGRendererInterface
from PySide6.QtQuickWidgets import QQuickWidget
from PySide6.QtWidgets import QApplication, QGraphicsGridLayout, QGridLayout, QWidget

import blink.drift_correction as dcorr
import blink.image_processing as imp
import gui.file_selectors
import gui.image_view as iv
from blink import mapping, time_series, visualization

pg.setConfigOption("background", "w")
pg.setConfigOption("foreground", "k")


class TraceInfoModel(QAbstractListModel):
    """Stores information about the current trace and interacts with view."""

    def __init__(self, parameter_file_path):
        super(TraceInfoModel, self).__init__()
        self.parameter_file_path = parameter_file_path
        parameter_file = pd.ExcelFile(parameter_file_path)
        sheet_names = parameter_file.sheet_names
        self.sheet_model = QStringListModel(sheet_names[:-1])
        self.current_sheet = sheet_names[0]
        self.fov_model = QStringListModel()
        self.set_fov_list(parameter_file)
        self.current_molecule = 0
        self.max_molecule = 1
        self.set_data_list_storage()
        self.property_name_role = Qt.UserRole + 1
        self.value_role = Qt.UserRole + 2
        self.choose_delegate_role = Qt.UserRole + 3
        self.dataChanged.connect(self.update_current_molecule, Qt.UniqueConnection)
        self.category = np.array([])

    def rowCount(self, parent: QModelIndex = None) -> int:
        """See base class."""
        return len(self.data_list)

    def data(self, index: QModelIndex, role: int = Qt.DisplayRole):
        """See base class."""
        if role == Qt.DisplayRole:
            return self.data_list[index.row()].value
        elif role == Qt.EditRole:
            return self.data_list[index.row()].value
        elif role == self.choose_delegate_role:
            return self.data_list[index.row()].chooseDelegate
        elif role == self.property_name_role:
            return self.data_list[index.row()].key
        elif role == self.value_role:
            return self.data_list[index.row()].value
        return None

    def roleNames(self):
        """See base class."""
        role_names = super(TraceInfoModel, self).roleNames()
        role_names[self.choose_delegate_role] = b"chooseDelegate"
        role_names[self.property_name_role] = b"propertyName"
        role_names[self.value_role] = b"value"
        return role_names

    def flags(self, index: QModelIndex) -> Qt.ItemFlags:
        """See base class."""
        if not index.isValid():
            return Qt.NoItemFlags
        return Qt.ItemIsEditable | Qt.ItemIsEnabled

    def setData(self, index: QModelIndex, value: typing.Any, role: int = None) -> bool:
        """See base class."""
        if index.isValid() and role == Qt.EditRole:
            row = index.row()
            if row == 2:  # molecule number
                value = int(value)
                if value == self.current_molecule:
                    return True
                if 0 <= value < self.max_molecule:
                    self.data_list[row] = self.data_list[row]._replace(value=value)
                    self.dataChanged.emit(index, index)
            else:
                self.data_list[row] = self.data_list[row]._replace(value=value)
                self.dataChanged.emit(index, index)
            return True
        return False

    def set_category(self, value: str):
        """Set the molecule category entry in view to the input string value."""
        self.data_list[3] = self.data_list[3]._replace(value=value)
        index = self.createIndex(3, 0)
        self.dataChanged.emit(index, index)

    def set_max_molecule_number(self, number: int):
        """Set the maximum molecule number to the input int. Used by TraceModel"""
        self.max_molecule = number

    def update_current_molecule(
        self, topleft: QModelIndex, bottomright: QModelIndex, role: list = None
    ):
        """Connected to self.dataChanged signal. Update the current molecule
        number if it is changed."""
        if topleft == bottomright:  # Single element changed
            if topleft.row() == 2:  # Molecule number entry
                self.current_molecule = self.data_list[2].value
                self.molecule_changed.emit()

    def set_data_list_storage(self, molecule: int = None):
        """Setup the data storage for this list model."""
        if molecule is None:
            molecule = self.current_molecule
        # choose delegate enumeration is defined in main.qml
        entry = namedtuple("data_entry", ["key", "value", "chooseDelegate"])
        self.data_list = [
            entry("Sheet name", self.current_sheet, 2),
            entry("Field of view", self.current_fov, 3),
            entry("Molecule", molecule, 1),
            entry("Category", "", 0),
        ]

    def set_fov_list(self, parameter_file: pd.ExcelFile = None):
        """Set the FOV string list model from the given parameter file. An
        already opened ExcelFile can be passed as an argument."""
        if parameter_file is None:
            parameter_file = pd.ExcelFile(self.parameter_file_path)
        dfs = parameter_file.parse(sheet_name=self.current_sheet)
        fov_list = dfs["filename"].tolist()
        parameter_file.close()
        self.fov_model.setStringList(fov_list)
        self.current_fov = fov_list[0]

    @Slot()
    def onNextMoleculeButtonClicked(self):
        """Set molecule number entry to (current value + 1)"""
        molecule_index = self.createIndex(2, 0)
        self.setData(molecule_index, self.current_molecule + 1, Qt.EditRole)

    @Slot()
    def onPreviousMoleculeButtonClicked(self):
        """Set molecule number entry to (current value - 1)"""
        molecule_index = self.createIndex(2, 0)
        self.setData(molecule_index, self.current_molecule - 1, Qt.EditRole)

    @Slot(int)
    def onSheetComboActivated(self, index: int):
        """Changes current sheet to the selected value in view and reinitialize
        self. Notify the TraceModel to change its data.

        index: The index of the selected item in the ComboBox component."""
        activated_sheet = self.sheet_model.data(self.sheet_model.createIndex(index, 0))
        if activated_sheet != self.current_sheet:
            self.current_sheet = activated_sheet
            self.set_fov_list()
            self.current_molecule = 0
            self.set_data_list_storage()
            self.dataChanged.emit(self.createIndex(2, 0), self.createIndex(3, 0))
            self.trace_model_should_change_file.emit()

    @Slot(int)
    def onFovComboActivated(self, index: int):
        """Changes current FOV to the selected value in view and reinitialize
        self. Notify the TraceModel to change its data.

        index: The index of the selected item in the ComboBox component."""
        activated_fov = self.fov_model.data(self.fov_model.createIndex(index, 0))
        if activated_fov != self.current_fov:
            self.current_fov = activated_fov
            self.current_molecule = 0
            self.set_data_list_storage()
            self.dataChanged.emit(self.createIndex(2, 0), self.createIndex(3, 0))
            self.trace_model_should_change_file.emit()

    @Slot()
    def onAnalyzableButtonClicked(self):
        self.category[self.current_molecule] = 1
        self.set_category("analyzable")

    @Slot()
    def onDiscardButtonClicked(self):
        self.category[self.current_molecule] = 0
        self.set_category("")

    def get_category(self):
        if self.category[self.current_molecule] == 1:
            return "analyzable"
        return ""

    def initialize_category(self, data_dir=None):
        if data_dir is None:
            self._category_path = (
                self.parameter_file_path.parent / f"{self.current_fov}_category.npy"
            )
        else:
            self._category_path = data_dir / f"{self.current_fov}_category.npy"
        if self._category_path.exists():
            self.category = np.load(self._category_path)
        else:
            self.category = np.zeros(self.max_molecule, dtype=int)

    def save_category(self):
        np.save(self._category_path, self.category, allow_pickle=False)

    @Slot()
    def debug(self):
        breakpoint()

    def _read_sheet_model(self):
        return self.sheet_model

    def _read_fov_model(self):
        return self.fov_model

    @Signal
    def sheet_model_changed(self):
        pass

    @Signal
    def fov_model_changed(self):
        pass

    @Signal
    def molecule_changed(self):
        pass

    @Signal
    def trace_model_should_change_file(self):
        pass

    sheetModel = Property(QObject, _read_sheet_model, notify=sheet_model_changed)
    fovModel = Property(QObject, _read_fov_model, notify=fov_model_changed)


class TraceModel(QObject):
    """Trace Model"""

    dataChanged = Signal()
    aoiImageOn = Signal()
    aoiImageOff = Signal()
    moving_average_changed = Signal()

    def __init__(self, trace_info_model: TraceInfoModel):
        super().__init__()
        self.moving_average = 1
        self.trace_info_model = trace_info_model
        self.trace_info_model.molecule_changed.connect(
            self.change_molecule, Qt.UniqueConnection
        )
        self.trace_info_model.trace_model_should_change_file.connect(
            self.change_file, Qt.UniqueConnection
        )
        self.datapath = gui.file_selectors.def_data_path()
        self.set_data_storage()
        self.show_aoi_image = False

    def get_category(self) -> typing.Union[str, None]:
        """Searches the current molecule in the AOI_catories dict and return the
        category (key). If the molecule is not found return None."""
        molecule = self.trace_info_model.current_molecule
        found = False
        for key, value in self.AOI_categories.items():
            if found is True:
                break
            if key == "analyzable":
                for key2, value2 in self.AOI_categories["analyzable"].items():
                    if molecule in value2:
                        category = key2
                        found = True
                        break
            else:
                if molecule in value:
                    category = key
                    found = True
                    break
        if found:
            return category
        return ""

    def change_molecule(self):
        """Notifies the data model that the current molecule is changed.

        Connected to the TraceInfoModel.molecule_changed signal. Also updates
        the category entry."""
        category = self.trace_info_model.get_category()
        self.trace_info_model.set_category(category)
        if self.show_aoi_image:
            self.setImageScales()
        self.dataChanged.emit()

    def set_data_storage(self):
        """
        Read trace data from file and store as attributes.

        Updates the model to the loaded data.
        """
        self.traces = time_series.TimeTraces.from_npz_eb(
            self.datapath / (self.trace_info_model.current_fov + "_traces.npz")
        )
        self.AOI_categories = dict()
        category = self.get_category()
        self.trace_info_model.set_category(category)
        self.channels = self.traces.get_channels()
        self.trace_info_model.set_max_molecule_number(self.traces.n_traces)
        self.trace_info_model.initialize_category(self.datapath)
        self.x_bounds = self.find_x_bounds()

    def change_file(self):
        """Updates the data model when switching to another data file.

        Connected to the TraceInfoModel.trace_model_should_change_file signal."""
        self.trace_info_model.save_category()
        self.set_data_storage()
        self.dataChanged.emit()

    @Slot()
    def save_fig(self):
        # TODO: alter this function
        """Save matplotlib traces plot of the current molecule as SVG file."""
        current_molecule = self.trace_info_model.current_molecule
        # category = self.get_category()
        fov_dir = self.datapath / self.trace_info_model.current_fov
        if not fov_dir.exists():
            fov_dir.mkdir()
        fig = visualization.plot_one_trace(
            self.traces, current_molecule, self.moving_average
        )
        fig.savefig(fov_dir / f"molecule{current_molecule}.svg", format="svg")

    def find_x_bounds(self):
        max_time_list = []
        min_time_list = []
        for channel in self.channels:
            time = self.traces.get_time(channel)
            max_time_list.append(time[-1])
            min_time_list.append(time[0])
        return (min(min_time_list), max(max_time_list))

    @Slot()
    def onToggleAoiImageButtonClicked(self):
        if self.show_aoi_image:
            self.aoiImageOff.emit()
        else:
            self.aoiImageOn.emit()
            self.loadAois()
            self.loadImageGroup()
            self.loadMapper()
            self.loadDriftCorrector()
            self.setImageScales()
        self.show_aoi_image = not self.show_aoi_image

    def loadAois(self):
        self.aois = imp.Aois.from_npz(
            self.datapath / (self.trace_info_model.current_fov + "_aoi.npz")
        )
        
        


    def loadImageGroup(self):
        path = iv.select_directory_dialog()
        if path is None:
            self.aoiImageOff.emit()
            self.show_aoi_image = False
            return
        self.image_group = imp.ImageGroup(path)

    def loadMapper(self):
        path = iv.open_file_path_dialog()
        if path is None:
            self.aoiImageOff.emit()
            self.show_aoi_image = False
            return
        self.mapper = mapping.Mapper.from_npz(path)

    def loadDriftCorrector(self):
        drift_list_path = self.datapath / (
            self.trace_info_model.current_fov + "_driftlist.npy"
        )
        if drift_list_path.is_file():
            self.drift_corrector = dcorr.DriftCorrector.from_npy(drift_list_path)
        else:
            self.drift_corrector = dcorr.DriftCorrector(None)

    def getAoiImage(self, channel, frame):
        aois = self.mapper.map(self.aois, to_channel=channel.em)
        aoi = aois[self.trace_info_model.current_molecule]
        aoi.channel = "red"
        time = self.traces.get_time(channel)[frame]
        drifted_aoi = self.drift_corrector.shift_aois_by_time(aoi, time)
        index = drifted_aoi.get_subimage_slice(11)
        image = self.image_group.sequences[channel].get_one_frame(frame)[index]
        return image

    def setImageScales(self):
        self.image_scales = dict()
        for channel in self.channels:
            data = self.traces._data[channel].sel(
                molecule=self.trace_info_model.current_molecule
            )
            background = data.background.values
            raw_intensity = data.raw_intensity.values
            self.image_scales[channel] = (
                background.mean() / 36,
                raw_intensity.max() / 36,
            )

    def _get_moving_average(self):
        return self.moving_average

    def _set_moving_average(self, val):
        self.moving_average = val
        self.dataChanged.emit()

    movingAverage = Property(
        int,
        fget=_get_moving_average,
        fset=_set_moving_average,
        notify=moving_average_changed,
    )


class TracePlot(pg.GraphicsLayoutWidget):
    def __init__(self, model: TraceModel):
        super().__init__()
        self.data_model = model
        self.data_model.dataChanged.connect(self.update)
        # xkcd purple blue,[tried green]
        self._colors = {
            imp.Channel("blue", "blue"): "#632de9",
            imp.Channel("green", "green"): "#14C823",
            imp.Channel("red", "red"): "#e50000",
            imp.Channel("green", "red"): "#ebb434",
            imp.Channel("red", "ir"): "#980043",
            imp.Channel("red", "green"): "#600000"
        }
        pens = {
            key: pg.mkPen(color=value, width=2) for key, value in self._colors.items()
        }
        self.label = pg.LabelItem(justify="right")
        self.addItem(self.label)
        self.plots = {
            channel: self.addPlot(row=row + 1, col=0)
            for row, channel in enumerate(self.data_model.channels)
        }
        
        self.int_curves = {
            channel: plot.plot(pen=pens[channel])
            for channel, plot in self.plots.items()
        }
        self.state_curves = {
            channel: plot.plot(pen=pg.mkPen(color="k", width=3))
            for channel, plot in self.plots.items()
        }
        self.interval_curves = {
            channel: plot.plot(pen=pg.mkPen(color="#FFD300", width=3))
            for channel, plot in self.plots.items()
        }

        # Create a cursor in the last subplot to select time
        self.vline = pg.InfiniteLine(angle=90, movable=True)
        self.vline.sigPositionChanged.connect(self.vlineMoved)
        self.plots[self.data_model.channels[-1]].addItem(self.vline)

        # Limit the size of the AOI images
        self.ci.layout.setColumnMaximumWidth(1, 100)
        self.image_plots = dict()
        self.images = dict()
        for channel in self.data_model.channels:
            image_plot = pg.ViewBox()
            image = pg.ImageItem()
            image_plot.addItem(image)
            self.image_plots[channel] = image_plot
            self.images[channel] = image
        for plot in self.image_plots.values():
            layout = QGraphicsGridLayout()
            layout.setMaximumWidth(100)
            layout.setMaximumHeight(100)
            plot.setLayout(layout)

        self.data_model.aoiImageOn.connect(self.showAoiImage)
        self.data_model.aoiImageOff.connect(self.hideAoiImage)

        self.update()

    def update(self):
        molecule = self.data_model.trace_info_model.current_molecule
        for channel in self.data_model.channels:
            traces = self.data_model.traces
            time = traces.get_time(channel)
            if self.data_model.moving_average > 1:
                kernel = (
                    np.ones(self.data_model.moving_average)
                    / self.data_model.moving_average
                )
                time = np.convolve(time, kernel, "valid")
                if traces.has_variable(channel, "intensity"):
                    intensity = np.convolve(
                        traces.get_intensity(channel, molecule), kernel, "valid"
                    )
                    self.int_curves[channel].setData(time, intensity)
                if traces.has_variable(channel, "is_colocalized"):
                    interval = np.convolve(
                        traces.get_is_colocalized(channel, molecule), kernel, "valid"
                    )
                    self.interval_curves[channel].setData(time, 1000 * interval)
                if traces.has_variable(channel, "viterbi_path"):
                    vit = np.convolve(
                        traces.get_viterbi_path(channel, molecule), kernel, "valid"
                    )
                    self.state_curves[channel].setData(time, vit)
            else:
                if traces.has_variable(channel, "intensity"):
                    self.int_curves[channel].setData(
                        time, traces.get_intensity(channel, molecule)
                    )
                if traces.has_variable(channel, "is_colocalized"):
                    self.interval_curves[channel].setData(
                        time, 1000 * (traces.get_is_colocalized(channel, molecule))
                    )
                if traces.has_variable(channel, "viterbi_path"):
                    vit = traces.get_viterbi_path(channel, molecule)
                    self.state_curves[channel].setData(time, vit)
        for channel in self.data_model.channels:
            self.plots[channel].setXRange(*self.data_model.x_bounds)

    def vlineMoved(self):
        cursor_time = self.vline.value()
        if (
            cursor_time > self.data_model.x_bounds[0]
            and cursor_time < self.data_model.x_bounds[1]
        ):
            intensity = self.data_model.traces.get_intensity_from_time(
                self.data_model.trace_info_model.current_molecule, cursor_time
            )
            txt_list = [f"t={cursor_time:.1f}"]
            for i, channel in enumerate(self.data_model.channels, start=1):
                txt = (
                    f',<span style="color: {self._colors[channel]}">'
                    f"y{i}={intensity[channel]:.0f}</span>"
                )
                txt_list.append(txt)
            self.label.setText(" ".join(txt_list))
        if self.data_model.show_aoi_image:
            self.updateAoiImage()

    def showAoiImage(self):
        for row, channel in enumerate(self.data_model.channels):
            # row + 1 because the first row is for label
            self.addItem(self.image_plots[channel], row=row + 1, col=1)

    def hideAoiImage(self):
        for channel in self.data_model.channels:
            self.removeItem(self.image_plots[channel])

    def updateAoiImage(self):
        cursor_time = self.vline.value()
        frame_index = self.data_model.traces.get_index_from_time(cursor_time)
        for channel in self.data_model.channels:
            self.images[channel].setImage(
                self.data_model.getAoiImage(channel, frame_index[channel])
            )
            self.images[channel].setLevels(self.data_model.image_scales[channel])


class TracePlotFret(pg.GraphicsLayoutWidget):
    def __init__(self, model: TraceModel):
        super().__init__()
        self.data_model = model
        self.data_model.dataChanged.connect(self.update)
        # xkcd purple blue,[tried green]
        pens = {
            "fret": pg.mkPen(color="#632de9", width=2),
            "total": pg.mkPen(color="#000000", width=2),
            "donor": pg.mkPen(color="#14C823", width=2),
            "red": pg.mkPen(color="#e50000", width=2),
            "acceptor": pg.mkPen(color="#ebb434", width=2),
        }
        self.subplot_names = ["total", "donor", "acceptor", "fret"]
        self.plots = {
            channel: self.addPlot(row=row, col=0)
            for row, channel in enumerate(self.subplot_names)
        }
        self.plots["fret"].setYRange(-0.2, 1.2)
        self.int_curves = {
            channel: plot.plot(pen=pens[channel])
            for channel, plot in self.plots.items()
        }
        self.int_curves["red"] = self.plots["acceptor"].plot(pen=pens["red"])
        self.update()

    def update(self):
        molecule = self.data_model.trace_info_model.current_molecule
        min_time_list = []
        max_time_list = []
        for channel in self.data_model.channels:
            traces = self.data_model.traces
            time = traces.get_time(channel)
            min_time_list.append(min(time))
            max_time_list.append(max(time)) 
        time = traces.get_time(imp.Channel("green", "red"))
        donor = traces.get_intensity(imp.Channel("green", "green"), molecule)
        acceptor = traces.get_intensity(imp.Channel("green", "red"), molecule)
        self.int_curves["donor"].setData(time, donor)
        self.int_curves["acceptor"].setData(time, acceptor)
        self.int_curves["total"].setData(time, donor + acceptor)
        self.int_curves["fret"].setData(time, acceptor / (donor + acceptor))
        red_channel = imp.Channel("green", "red")
        self.int_curves["red"].setData(
            traces.get_time(red_channel), traces.get_intensity(red_channel, molecule)
        )
        # TODO: the following lines seems to affect switching performance, need
        # further check
        for plot in self.plots.values():
            plot.setXRange(float(min(min_time_list)), float(max(max_time_list)))


class MainWindow(QWidget):
    def __init__(self, fret=False):
        super().__init__()

        parameter_file_path = gui.file_selectors.get_xlsx_parameter_file_path()
        self.trace_info_model = TraceInfoModel(parameter_file_path)
        trace_model = TraceModel(self.trace_info_model)

        view = QQuickWidget()
        view.setResizeMode(QQuickWidget.SizeRootObjectToView)
        root_context = view.rootContext()
        root_context.setContextProperty("traceInfoModel", trace_model.trace_info_model)
        root_context.setContextProperty("traceModel", trace_model)
        qml_path = Path(__file__).parent / "qml/main.qml"
        view.setSource(QUrl.fromLocalFile(str(qml_path)))

        layout = QGridLayout()
        if fret:
            plot = TracePlotFret(model=trace_model)
        else:
            plot = TracePlot(model=trace_model)
        layout.addWidget(plot, 0, 0)
        layout.addWidget(view, 0, 1)
        self.setLayout(layout)
        self.qml = view

    def closeEvent(self, event):
        self.trace_info_model.save_category()
        event.accept()


def main():
    """Starts the GUI window after asking for the parameter file."""

    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    QQuickWindow.setGraphicsApi(QSGRendererInterface.OpenGL)
    if len(sys.argv) > 1:
        fret = sys.argv[1] == "fret"
    else:
        fret = False
    window = MainWindow(fret)
    window.show()
    # The following line is to delete the QQuickWidget before python exits. Otherwise, a
    # "TypeError: Cannot read property 'xxx' of null" will be thrown by QML because the
    # objects that have been made the QML's context properties could get deleted by
    # python before the QML, triggering a binding re-evaluation, thus the error.
    # See QTBUG-81247 comments.
    app.aboutToQuit.connect(window.qml.deleteLater)
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
