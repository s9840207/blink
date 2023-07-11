import QtQuick 6.2
import QtQuick.Controls 6.2
import QtQuick.Layouts 6.2

Item{
    id: root
    width: 500; height: 480

    Column {
    spacing: 10

    Row {
	    Button {
		width: 60
		text: 'Pick'
		onClicked: imageView.onPickButtonPressed()

	    }
	    Button {
		width: 60
		text: 'Fit'
		onClicked: imageView.onFitButtonPressed()
	    }
	    Button {
		width: 60
		text: 'Add'
		onClicked: imageView.onAddButtonPressed()
	    }
	    Button {
		width: 60
		text: 'Remove'
		onClicked: imageView.onRemoveButtonPressed()
	    }
    }

    GridLayout {
	columns: 2

	Text {
	    text: 'AOI width'
	    verticalAlignment: Text.AlignVCenter
	}
	MySpinBox {
	    value: dataModel.pickSpotsParam.aoiWidth
	    onValueModified: {dataModel.pickSpotsParam.aoiWidth = value}
	}

	Text {
	    text: 'Spot diameter'
	    verticalAlignment: Text.AlignVCenter
	}
	MySpinBox {
	    value: dataModel.pickSpotsParam.spotDia
	    onValueModified: {dataModel.pickSpotsParam.spotDia = value}
	    from: 1
	    to: 99
	    stepSize: 2
	}

	Text {
	    text: 'Noise diameter'
	    verticalAlignment: Text.AlignVCenter
	}
	MySpinBox {
	    value: dataModel.pickSpotsParam.noiseDia
	    onValueModified: {dataModel.pickSpotsParam.noiseDia = value}
	    from: 1
	    to: 10
	}

	Text {
	    text: 'Spot brightness'
	    verticalAlignment: Text.AlignVCenter
	}
	MySpinBox {
	    value: dataModel.pickSpotsParam.spotBrightness
	    onValueModified: {dataModel.pickSpotsParam.spotBrightness = value}
	    to: 1000
	}

	Text {
	    text: 'Distance threshold'
	    verticalAlignment: Text.AlignVCenter
	}
	TextField {
	    implicitWidth: 80
	    text: dataModel.pickSpotsParam.distThreshold
	    onEditingFinished: {dataModel.pickSpotsParam.distThreshold = text}
	    horizontalAlignment: TextInput.AlignRight
	    validator: DoubleValidator{bottom: 0}
	    selectByMouse: true
	}

	Text {
	    text: 'AOI counts'
	}
	Text {
	    Layout.preferredWidth: 80
	    text: imageView.aoiCount
	    horizontalAlignment: Text.AlignRight
	} 

	Text {
	    text: 'Current frame'
	}
	Text {
	    Layout.preferredWidth: 80
	    text: imageView.idx
	    horizontalAlignment: Text.AlignRight
	}

	Text {
	    text: 'Frame average'
	}

	MySpinBox {
		value: dataModel.frameAverage
		onValueModified: {dataModel.frameAverage = value}
		from: 1
		to: imageView.nFrames
	}

    }
    Button {
        text: 'Remove close AOI'
        onClicked: imageView.remove_close_aoi() // Signal defined in image_view.py
    }
    Button {
        text: 'Remove empty AOI'
        onClicked: imageView.remove_empty_aoi()
    }
    Button {
        text: 'Remove occupied AOI'
        onClicked: imageView.remove_occupied_aoi()
    }
    Row {

    Button {
        text: 'Save'
        onClicked: imageView.save_aois()
    }
    Button {
        text: 'Load'
        onClicked: imageView.load_aois()
    }
    }
    Row {
	    Button {
		text: 'LoadMapping'
		onClicked: imageView.loadMapping()
	    }
	    Button {
		text: 'Map'
		onClicked: imageView.map()
	    }
	    Button {
		text: 'Inverse Map'
		onClicked: imageView.inverseMap()
	    }
	    }
	    Text {
		text: imageView.mapMatrixString
	    }
    }

}
