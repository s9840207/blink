import QtQuick 6.2
import QtQuick.Controls 6.2


SpinBox {
    id: control
    implicitWidth: 80
    down.indicator.implicitHeight: 20
    down.indicator.implicitWidth: 20
    up.indicator.implicitHeight: 20
    up.indicator.implicitWidth: 20
    contentItem: TextInput {
        // This item was copied from the source code, I only needed
        // to modify its selectByMouse property
        z: 2
        text: control.displayText

        font: control.font
        color: control.palette.text
        selectionColor: control.palette.highlight
        selectedTextColor: control.palette.highlightedText
        horizontalAlignment: Qt.AlignHCenter
        verticalAlignment: Qt.AlignVCenter

        readOnly: !control.editable
        validator: control.validator
        inputMethodHints: control.inputMethodHints
        selectByMouse: true
    }

    editable: true
}
