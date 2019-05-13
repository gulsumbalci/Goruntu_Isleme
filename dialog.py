# -*- coding: utf-8 -*-
"""
Created on Sat Dec 09 23:16:40 2017

@author: GULSUM
"""

import sys
from PyQt4 import QtGui

from main import MainWindow

def main():
    app = QtGui.QApplication(sys.argv)

    mainWindow = MainWindow()
    mainWindow.show()

    return app.exec_()

if __name__ == "__main__":
    main()