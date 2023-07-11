import sys
from pathlib import Path

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication, QFileDialog, QWidget


def def_data_path():
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    w = QWidget()
    data_dir_str = QFileDialog.getExistingDirectory(w, caption="Select data path")
    datapath = Path(data_dir_str)
    # Let the event loop terminate after 1 ms and quit QApplication
    QTimer.singleShot(1, app.quit)
    app.exec()
    return datapath


def qt_getfile():
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    w = QWidget()
    data_dir_str = QFileDialog.getOpenFileName(w, caption="Select file")[0]
    datapath = Path(data_dir_str)
    # Let the event loop terminate after 1 ms and quit QApplication
    QTimer.singleShot(1, app.quit)
    app.exec()
    return datapath


def qt_save_file():
    # There are still some problem need to solve!
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    w = QWidget()
    FILTERS = ("SVG Images (*.svg)", ".svg" "PNG Images (*.png)")
    filter_string = ""
    for file_filter in FILTERS:
        filter_string += file_filter + ";;"
    filter_string = filter_string[:-2]

    file_path, selected_filter = QFileDialog.getSaveFileName(
        w, caption="Save to file", filter=filter_string
    )
    file_path = Path(file_path)
    # if file_path.suffix == '':
    #     if selected_filter ==
    #     file_path.with_suffix()
    # Let the event loop terminate after 1 ms and quit QApplication
    QTimer.singleShot(1, app.quit)
    app.exec()
    return file_path


def get_xlsx_parameter_file_path() -> Path:
    """Opens a window to select the xlsx parameter file."""
    while True:
        xlsx_parameter_file_path = qt_getfile()
        print(xlsx_parameter_file_path)
        yes_no = input("Confirm [y/n]: ")
        if yes_no == "y":
            xlsx_parameter_file_path = Path(xlsx_parameter_file_path)
            break
    return xlsx_parameter_file_path
