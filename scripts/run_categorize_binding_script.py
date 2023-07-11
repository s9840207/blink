from typing import List

import blink.categorize_binding_traces_script as c
import gui.file_selectors


def main():
    """main function"""
    xlsx_parameter_file_path = gui.file_selectors.get_xlsx_parameter_file_path()
    sheet_list = input_sheets_for_analysis()
    datapath = gui.file_selectors.def_data_path()
    c.categorize_binding_traces(xlsx_parameter_file_path, sheet_list, datapath)


def input_sheets_for_analysis() -> List[str]:
    """Request user input in console for list of sheets to be analyzed.

    The input format should be 'SHEET1, SHEET2, SHEET3, '..."""
    while True:
        input_str = input("Enter the sheets to be analyzed: ")
        sheet_list_out = input_str.split(", ")
        print(sheet_list_out)
        yes_no = input("Confirm [y/n]: ")
        if yes_no == "y":
            break
    return sheet_list_out


if __name__ == "__main__":
    main()
