import math
import os
import queue
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import matplotlib
import wx

matplotlib.use("WXAgg")
import wx.lib.inspection
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as FigCanvas
from matplotlib.figure import Figure
from matplotlib.widgets import RectangleSelector

from pds.gui.console_capture import ConsoleCapture
from pds.gui.hdf_to_tree import HDFToTree, myTreeCtrl
from pds.utils import CtrCorrectionPsic, FileLockException, HdfDataFile, ImageAna, bytes_to_str, image_point_F, psic_from_spec, safe_eval_hdf


class Integrator(wx.Frame, wx.Notebook):
    """Main interface for integrating data. Provides GUI for HDF data processing and visualization."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.newFile = True
        self.firstOpen = True
        self.hdfRoot = None
        self.hdfObject = None
        self.hdfTreeObject = None
        self.customTreeObject = None
        self.lastDirectory = os.getcwd()  # Track last used directory
        # Threading support
        self.hdf_lock = threading.Lock()  # Lock for HDF5 file access
        self.integrateContinue = False  # Flag for cancellation

        wx.Frame.__init__(self, args[0], -1, title="HDF Integrator", size=(1024, 780))

        self.menuBar = wx.MenuBar()
        self.fileMenu = wx.Menu()
        self.loadHDF = self.fileMenu.Append(-1, "Load Project File...")
        self.fileMenu.AppendSeparator()
        self.saveAttributes = self.fileMenu.Append(-1, "Save attributes...")
        self.fileMenu.AppendSeparator()
        self.saveHKL = self.fileMenu.Append(-1, "Save HKLFFerr...", "Write H, K, L, F, and Ferr\n" + "values to a file")
        self.saveHKLE = self.fileMenu.Append(-1, "Save HKLEFFerr...", "Write H, K, L, E, F, and\n" + "Ferr values to a file")
        self.saveCTRab = self.fileMenu.Append(-1, "Save CTR, alpha, beta", "Write H, K, L, F, \n" + "Ferr, alpha and beta to file")
        self.saveRIDS = self.fileMenu.Append(-1, "Save RIDS...", "Write E, H, K, L, F, \n" + "Ferr, alpha and beta to file")
        self.saveIIoIbgr = self.fileMenu.Append(-1, "Save Intensities...", "Write H, K, L, F, \n" + "Ferr, I, Io, Ibgr, Sec")
        self.menuBar.Append(self.fileMenu, "File")

        self.editMenu = wx.Menu()
        self.copyParams = self.editMenu.Append(-1, "Copy parameters...")
        self.menuBar.Append(self.editMenu, "Edit")

        self.viewMenu = wx.Menu()
        self.viewAreaCorrection = self.viewMenu.Append(-1, "View Area Correction")
        self.viewMenu.AppendSeparator()
        self.toggleOutputWindow = self.viewMenu.Append(-1, "Show Output Window", "Toggle stdout/stderr output window", wx.ITEM_CHECK)
        self.menuBar.Append(self.viewMenu, "View")

        self.SetMenuBar(self.menuBar)

        self.statusBar = self.CreateStatusBar(5)
        self.statusBar.SetStatusWidths([238, -1, 238, 56, 91])

        self.statusSizer = wx.BoxSizer(wx.VERTICAL)

        self.integrateCancel = wx.Button(self.statusBar, label="Cancel")
        self.integrateCancel.Bind(wx.EVT_BUTTON, self.integrateStop)
        self.integrateCancel.Hide()

        self.mainSplitter = wx.SplitterWindow(self, style=wx.SP_3D | wx.SP_LIVE_UPDATE)
        self.splitWindow = wx.SplitterWindow(self.mainSplitter, style=wx.SP_3D | wx.SP_LIVE_UPDATE)
        self.rightSplitter = wx.SplitterWindow(self.splitWindow, style=wx.SP_3D | wx.SP_LIVE_UPDATE)

        self.treeBook = wx.Notebook(self.splitWindow, style=wx.SUNKEN_BORDER)
        self.treePanel = wx.Panel(self.treeBook)
        self.treeInfo = wx.Panel(self.treeBook)
        self.treeBook.AddPage(self.treePanel, "All")
        self.treeBook.AddPage(self.treeInfo, "Info")

        self.dataWindow = wx.SplitterWindow(self.rightSplitter)
        self.graphPanel = wx.Panel(self.dataWindow, style=wx.SUNKEN_BORDER)
        self.rodPanel = wx.Panel(self.dataWindow, style=wx.SUNKEN_BORDER)
        self.infoPanel = wx.Panel(self.rightSplitter, style=wx.SUNKEN_BORDER)

        self.splitWindow.SplitVertically(self.treeBook, self.rightSplitter, 180)
        self.splitWindow.SetMinimumPaneSize(32)
        self.rightSplitter.SplitVertically(self.dataWindow, self.infoPanel, 500)
        self.rightSplitter.SetMinimumPaneSize(32)

        self.dataWindow.SetMinimumPaneSize(32)
        self.dataWindow.SplitHorizontally(self.graphPanel, self.rodPanel, 391)

        self.paramBook = wx.Notebook(self.infoPanel, style=wx.SUNKEN_BORDER)
        self.basicPage = wx.Panel(self.paramBook)
        self.morePage = wx.Panel(self.paramBook)
        self.paramBook.AddPage(self.basicPage, "Basic")
        self.paramBook.AddPage(self.morePage, "More")

        self.hdfTree = myTreeCtrl(self.treePanel)
        self.customSelection = customSelector(self)

        # Set up variables for lazy initialization of output window
        self.outputPanel = None
        self.outputText = None
        self.clearOutputBtn = None
        self.consoleCapture = None
        self.outputWindowVisible = False

        self.originalStdout = sys.stdout
        self.originalStderr = sys.stderr
        self.captureEnabled = False
        self.fig4 = Figure(figsize=(1, 1))
        self.canvas4 = FigCanvas(self.graphPanel, -1, self.fig4)

        self.rodFig = Figure(figsize=(1, 1))
        self.rodCanvas = FigCanvas(self.rodPanel, -1, self.rodFig)
        self.rodCanvas.mpl_connect("motion_notify_event", self.updateCursorStatus)
        self.rodCanvas.mpl_connect("button_release_event", self.clickToSelect)

        self.treeSizer = wx.BoxSizer(wx.VERTICAL)
        self.treeSizer.Add(self.hdfTree, proportion=1, flag=wx.EXPAND | wx.ALL, border=4)

        self.graphControlSizer = wx.BoxSizer(wx.HORIZONTAL)

        # Bad Point checkbox
        self.badPointLbl = wx.StaticText(self.graphPanel, label="Bad Point: ")
        self.badPointToggle = wx.CheckBox(self.graphPanel)

        # Image max field
        self.imageMaxLbl = wx.StaticText(self.graphPanel, label="Image Max: ")
        self.imageMaxField = wx.TextCtrl(self.graphPanel, size=(56, -1), style=wx.TE_PROCESS_ENTER)
        self.keepMaxLbl = wx.StaticText(self.graphPanel, label="Freeze: ")
        self.keepMaxToggle = wx.CheckBox(self.graphPanel)
        self.imageMaxValue = wx.StaticText(self.graphPanel, label="ROI Max: ")

        self.graphControlSizer.Add(self.badPointLbl, flag=wx.ALIGN_CENTER | wx.LEFT, border=8)
        self.graphControlSizer.Add(self.badPointToggle, flag=wx.ALIGN_CENTER)
        self.graphControlSizer.Add(wx.StaticLine(self.graphPanel, style=wx.LI_VERTICAL), flag=wx.EXPAND | wx.LEFT, border=16)
        self.graphControlSizer.Add(self.imageMaxLbl, flag=wx.ALIGN_CENTER | wx.LEFT, border=16)
        self.graphControlSizer.Add(self.imageMaxField, flag=wx.ALIGN_CENTER)
        self.graphControlSizer.Add(self.keepMaxLbl, flag=wx.ALIGN_CENTER | wx.LEFT, border=8)
        self.graphControlSizer.Add(self.keepMaxToggle, flag=wx.ALIGN_CENTER)
        self.graphControlSizer.Add(self.imageMaxValue, flag=wx.ALIGN_CENTER | wx.LEFT, border=8)

        self.graphSizer = wx.BoxSizer(wx.VERTICAL)
        self.graphSizer.Add(self.canvas4, proportion=1, flag=wx.EXPAND | wx.ALL, border=4)
        self.graphSizer.Add(self.graphControlSizer, flag=wx.ALIGN_CENTER | wx.BOTTOM, border=4)

        self.rodSizer = wx.BoxSizer(wx.VERTICAL)
        self.rodSizer.Add(self.rodCanvas, proportion=1, flag=wx.EXPAND | wx.ALL, border=4)

        # The top is the point parameters, the middle the scan parameters, and the bottom the point information and integrate button
        self.infoSizer1 = wx.BoxSizer(wx.VERTICAL)
        # This spaces the parameter label and the tab box appropriately
        self.paramSizer1 = wx.BoxSizer(wx.VERTICAL)
        # This spaces the basic tab contents
        self.basicSizer1 = wx.BoxSizer(wx.VERTICAL)
        self.basicSizer2 = wx.BoxSizer(wx.HORIZONTAL)
        self.basicSizer3 = wx.BoxSizer(wx.VERTICAL)
        self.basicSizer4 = wx.BoxSizer(wx.VERTICAL)

        # The top label
        self.pointParamLbl = wx.StaticText(self.infoPanel, label="Point Parameters:")

        # Point basic parameter sizers
        self.colNbgrSizer1 = wx.BoxSizer(wx.HORIZONTAL)
        self.colNbgrSizer2 = wx.BoxSizer(wx.HORIZONTAL)
        self.colPowerSizer1 = wx.BoxSizer(wx.HORIZONTAL)
        self.colPowerSizer2 = wx.BoxSizer(wx.HORIZONTAL)
        self.colWidthSizer1 = wx.BoxSizer(wx.HORIZONTAL)
        self.colWidthSizer2 = wx.BoxSizer(wx.HORIZONTAL)
        self.rowNbgrSizer1 = wx.BoxSizer(wx.HORIZONTAL)
        self.rowNbgrSizer2 = wx.BoxSizer(wx.HORIZONTAL)
        self.rowPowerSizer1 = wx.BoxSizer(wx.HORIZONTAL)
        self.rowPowerSizer2 = wx.BoxSizer(wx.HORIZONTAL)
        self.rowWidthSizer1 = wx.BoxSizer(wx.HORIZONTAL)
        self.rowWidthSizer2 = wx.BoxSizer(wx.HORIZONTAL)
        self.flagSizer1 = wx.BoxSizer(wx.HORIZONTAL)
        self.flagSizer2 = wx.BoxSizer(wx.HORIZONTAL)
        self.roiSizer1 = wx.BoxSizer(wx.HORIZONTAL)
        self.roiSizer2 = wx.BoxSizer(wx.HORIZONTAL)
        self.rotateSizer1 = wx.BoxSizer(wx.HORIZONTAL)
        self.rotateSizer2 = wx.BoxSizer(wx.HORIZONTAL)
        self.applySizer = wx.BoxSizer(wx.HORIZONTAL)

        # Create all the fields here
        self.colNbgrField = wx.TextCtrl(self.basicPage, style=wx.TE_PROCESS_ENTER)
        self.colPowerField = wx.TextCtrl(self.basicPage, style=wx.TE_PROCESS_ENTER)
        self.colWidthField = wx.TextCtrl(self.basicPage, style=wx.TE_PROCESS_ENTER)
        self.rowNbgrField = wx.TextCtrl(self.basicPage, style=wx.TE_PROCESS_ENTER)
        self.rowPowerField = wx.TextCtrl(self.basicPage, style=wx.TE_PROCESS_ENTER)
        self.rowWidthField = wx.TextCtrl(self.basicPage, style=wx.TE_PROCESS_ENTER)
        self.flagField = wx.TextCtrl(self.basicPage, style=wx.TE_PROCESS_ENTER)
        self.roiField = wx.TextCtrl(self.basicPage, style=wx.TE_PROCESS_ENTER)
        self.rotateField = wx.TextCtrl(self.basicPage, style=wx.TE_PROCESS_ENTER)

        # The 'Apply above to:' row (placed here for tab traversal order)
        self.applyLbl1 = wx.StaticText(self.basicPage, label="Apply above to: ")
        self.applyScan = wx.Button(self.basicPage, label="Scan")
        self.applyCustom = wx.Button(self.basicPage, label="Custom...")

        self.applySizer.Add(self.applyLbl1, flag=wx.ALIGN_CENTER | wx.LEFT | wx.RIGHT, border=4)
        self.applySizer.Add(self.applyScan, proportion=1, flag=wx.EXPAND | wx.ALL)
        self.applySizer.Add(self.applyCustom, proportion=1, flag=wx.EXPAND | wx.ALL)

        # Number of columns to use for background
        self.colNbgrLbl1 = wx.StaticText(self.basicPage, label="# bgr col: ")
        self.colNbgrScan = wx.Button(self.basicPage, label="Scan")
        self.colNbgrCustom = wx.Button(self.basicPage, label="Custom")
        self.colNbgrFreeze = wx.CheckBox(self.basicPage)

        self.colNbgrSizer1.Add(self.colNbgrLbl1, flag=wx.ALIGN_CENTER | wx.LEFT | wx.RIGHT, border=4)
        self.colNbgrSizer1.Add(self.colNbgrField, proportion=1, flag=wx.EXPAND | wx.ALL)
        self.colNbgrSizer2.Add(self.colNbgrScan, proportion=3, flag=wx.EXPAND | wx.ALL)
        self.colNbgrSizer2.Add(self.colNbgrCustom, proportion=3, flag=wx.EXPAND | wx.ALL)
        self.colNbgrSizer2.AddStretchSpacer()
        self.colNbgrSizer2.Add(self.colNbgrFreeze, flag=wx.ALIGN_CENTER)
        self.colNbgrSizer2.AddStretchSpacer()

        # Power of column background subtraction
        self.colPowerLbl1 = wx.StaticText(self.basicPage, label="bgr col power: ")
        self.colPowerScan = wx.Button(self.basicPage, label="Scan")
        self.colPowerCustom = wx.Button(self.basicPage, label="Custom")
        self.colPowerFreeze = wx.CheckBox(self.basicPage)

        self.colPowerSizer1.Add(self.colPowerLbl1, flag=wx.ALIGN_CENTER | wx.LEFT | wx.RIGHT, border=4)
        self.colPowerSizer1.Add(self.colPowerField, proportion=1, flag=wx.EXPAND | wx.ALL)
        self.colPowerSizer2.Add(self.colPowerScan, proportion=3, flag=wx.EXPAND | wx.ALL)
        self.colPowerSizer2.Add(self.colPowerCustom, proportion=3, flag=wx.EXPAND | wx.ALL)
        self.colPowerSizer2.AddStretchSpacer()
        self.colPowerSizer2.Add(self.colPowerFreeze, flag=wx.ALIGN_CENTER)
        self.colPowerSizer2.AddStretchSpacer()

        # Width of column background subtraction peaks
        self.colWidthLbl1 = wx.StaticText(self.basicPage, label="bgr col width: ")
        self.colWidthScan = wx.Button(self.basicPage, label="Scan")
        self.colWidthCustom = wx.Button(self.basicPage, label="Custom")
        self.colWidthFreeze = wx.CheckBox(self.basicPage)

        self.colWidthSizer1.Add(self.colWidthLbl1, flag=wx.ALIGN_CENTER | wx.LEFT | wx.RIGHT, border=4)
        self.colWidthSizer1.Add(self.colWidthField, proportion=1, flag=wx.EXPAND | wx.ALL)
        self.colWidthSizer2.Add(self.colWidthScan, proportion=3, flag=wx.EXPAND | wx.ALL)
        self.colWidthSizer2.Add(self.colWidthCustom, proportion=3, flag=wx.EXPAND | wx.ALL)
        self.colWidthSizer2.AddStretchSpacer()
        self.colWidthSizer2.Add(self.colWidthFreeze, flag=wx.ALIGN_CENTER)
        self.colWidthSizer2.AddStretchSpacer()

        # Number of rows to use for background
        self.rowNbgrLbl1 = wx.StaticText(self.basicPage, label="# bgr row: ")
        self.rowNbgrScan = wx.Button(self.basicPage, label="Scan")
        self.rowNbgrCustom = wx.Button(self.basicPage, label="Custom")
        self.rowNbgrFreeze = wx.CheckBox(self.basicPage)

        self.rowNbgrSizer1.Add(self.rowNbgrLbl1, flag=wx.ALIGN_CENTER | wx.LEFT | wx.RIGHT, border=4)
        self.rowNbgrSizer1.Add(self.rowNbgrField, proportion=1, flag=wx.EXPAND | wx.ALL)
        self.rowNbgrSizer2.Add(self.rowNbgrScan, proportion=3, flag=wx.EXPAND | wx.ALL)
        self.rowNbgrSizer2.Add(self.rowNbgrCustom, proportion=3, flag=wx.EXPAND | wx.ALL)
        self.rowNbgrSizer2.AddStretchSpacer()
        self.rowNbgrSizer2.Add(self.rowNbgrFreeze, flag=wx.ALIGN_CENTER)
        self.rowNbgrSizer2.AddStretchSpacer()

        # Power of row background subtraction
        self.rowPowerLbl1 = wx.StaticText(self.basicPage, label="bgr row power: ")
        self.rowPowerScan = wx.Button(self.basicPage, label="Scan")
        self.rowPowerCustom = wx.Button(self.basicPage, label="Custom")
        self.rowPowerFreeze = wx.CheckBox(self.basicPage)

        self.rowPowerSizer1.Add(self.rowPowerLbl1, flag=wx.ALIGN_CENTER | wx.LEFT | wx.RIGHT, border=4)
        self.rowPowerSizer1.Add(self.rowPowerField, proportion=1, flag=wx.EXPAND | wx.ALL)
        self.rowPowerSizer2.Add(self.rowPowerScan, proportion=3, flag=wx.EXPAND | wx.ALL)
        self.rowPowerSizer2.Add(self.rowPowerCustom, proportion=3, flag=wx.EXPAND | wx.ALL)
        self.rowPowerSizer2.AddStretchSpacer()
        self.rowPowerSizer2.Add(self.rowPowerFreeze, flag=wx.ALIGN_CENTER)
        self.rowPowerSizer2.AddStretchSpacer()

        # Width of row background subtraction peaks
        self.rowWidthLbl1 = wx.StaticText(self.basicPage, label="bgr row width: ")
        self.rowWidthScan = wx.Button(self.basicPage, label="Scan")
        self.rowWidthCustom = wx.Button(self.basicPage, label="Custom")
        self.rowWidthFreeze = wx.CheckBox(self.basicPage)

        self.rowWidthSizer1.Add(self.rowWidthLbl1, flag=wx.ALIGN_CENTER | wx.LEFT | wx.RIGHT, border=4)
        self.rowWidthSizer1.Add(self.rowWidthField, proportion=1, flag=wx.EXPAND | wx.ALL)
        self.rowWidthSizer2.Add(self.rowWidthScan, proportion=3, flag=wx.EXPAND | wx.ALL)
        self.rowWidthSizer2.Add(self.rowWidthCustom, proportion=3, flag=wx.EXPAND | wx.ALL)
        self.rowWidthSizer2.AddStretchSpacer()
        self.rowWidthSizer2.Add(self.rowWidthFreeze, flag=wx.ALIGN_CENTER)
        self.rowWidthSizer2.AddStretchSpacer()

        # What method to use for background subtraction
        self.flagLbl1 = wx.StaticText(self.basicPage, label="Flag: ")
        self.flagScan = wx.Button(self.basicPage, label="Scan")
        self.flagCustom = wx.Button(self.basicPage, label="Custom")
        self.flagFreeze = wx.CheckBox(self.basicPage)

        self.flagSizer1.Add(self.flagLbl1, flag=wx.ALIGN_CENTER | wx.LEFT | wx.RIGHT, border=4)
        self.flagSizer1.Add(self.flagField, proportion=1, flag=wx.EXPAND | wx.ALL)
        self.flagSizer2.Add(self.flagScan, proportion=3, flag=wx.EXPAND | wx.ALL)
        self.flagSizer2.Add(self.flagCustom, proportion=3, flag=wx.EXPAND | wx.ALL)
        self.flagSizer2.AddStretchSpacer()
        self.flagSizer2.Add(self.flagFreeze, flag=wx.ALIGN_CENTER)
        self.flagSizer2.AddStretchSpacer()

        # The ROI to use
        self.roiLbl1 = wx.StaticText(self.basicPage, label="ROI: ")
        self.roiScan = wx.Button(self.basicPage, label="Scan")
        self.roiCustom = wx.Button(self.basicPage, label="Custom")
        self.roiFreeze = wx.CheckBox(self.basicPage)

        self.roiSizer1.Add(self.roiLbl1, flag=wx.ALIGN_CENTER | wx.LEFT | wx.RIGHT, border=4)
        self.roiSizer1.Add(self.roiField, proportion=2, flag=wx.EXPAND | wx.ALL)
        self.roiSizer2.Add(self.roiScan, proportion=3, flag=wx.EXPAND | wx.ALL)
        self.roiSizer2.Add(self.roiCustom, proportion=3, flag=wx.EXPAND | wx.ALL)
        self.roiSizer2.AddStretchSpacer()
        self.roiSizer2.Add(self.roiFreeze, flag=wx.ALIGN_CENTER)
        self.roiSizer2.AddStretchSpacer()

        # How much (if any) to rotate the image
        self.rotateLbl1 = wx.StaticText(self.basicPage, label="Rotate image: ")
        self.rotateScan = wx.Button(self.basicPage, label="Scan")
        self.rotateCustom = wx.Button(self.basicPage, label="Custom")
        self.rotateFreeze = wx.CheckBox(self.basicPage)

        self.rotateSizer1.Add(self.rotateLbl1, flag=wx.ALIGN_CENTER | wx.LEFT | wx.RIGHT, border=4)
        self.rotateSizer1.Add(self.rotateField, proportion=1, flag=wx.EXPAND | wx.ALL)
        self.rotateSizer2.Add(self.rotateScan, proportion=3, flag=wx.EXPAND | wx.ALL)
        self.rotateSizer2.Add(self.rotateCustom, proportion=3, flag=wx.EXPAND | wx.ALL)
        self.rotateSizer2.AddStretchSpacer()
        self.rotateSizer2.Add(self.rotateFreeze, flag=wx.ALIGN_CENTER)
        self.rotateSizer2.AddStretchSpacer()

        # Add all the label / input field pairs
        self.basicSizer3.Add((1, 1), proportion=1, flag=wx.EXPAND | wx.ALL)
        self.basicSizer3.Add(self.colNbgrSizer1, proportion=1, flag=wx.EXPAND | wx.ALL)
        self.basicSizer3.Add(self.colPowerSizer1, proportion=1, flag=wx.EXPAND | wx.ALL)
        self.basicSizer3.Add(self.colWidthSizer1, proportion=1, flag=wx.EXPAND | wx.ALL)
        self.basicSizer3.Add(self.rowNbgrSizer1, proportion=1, flag=wx.EXPAND | wx.ALL)
        self.basicSizer3.Add(self.rowPowerSizer1, proportion=1, flag=wx.EXPAND | wx.ALL)
        self.basicSizer3.Add(self.rowWidthSizer1, proportion=1, flag=wx.EXPAND | wx.ALL)
        self.basicSizer3.Add(self.flagSizer1, proportion=1, flag=wx.EXPAND | wx.ALL)
        self.basicSizer3.Add(self.roiSizer1, proportion=1, flag=wx.EXPAND | wx.ALL)
        self.basicSizer3.Add(self.rotateSizer1, proportion=1, flag=wx.EXPAND | wx.ALL)

        # The sizer for the headings above the scan / custom buttons and freeze check box
        self.headingSizer = wx.BoxSizer(wx.HORIZONTAL)
        self.headingSizer.AddStretchSpacer(prop=5)
        self.headingSizer.Add(wx.StaticText(self.basicPage, label="Apply to:"), flag=wx.ALIGN_CENTER | wx.LEFT, border=4)
        self.headingSizer.AddStretchSpacer(prop=6)
        self.headingSizer.Add(wx.StaticText(self.basicPage, label="Freeze:"), flag=wx.ALIGN_CENTER | wx.LEFT, border=4)
        self.headingSizer.AddStretchSpacer(prop=1)

        # Add all the label / button pairs
        self.basicSizer4.Add(self.headingSizer, proportion=1, flag=wx.EXPAND | wx.ALL)
        self.basicSizer4.Add(self.colNbgrSizer2, proportion=1, flag=wx.EXPAND | wx.ALL)
        self.basicSizer4.Add(self.colPowerSizer2, proportion=1, flag=wx.EXPAND | wx.ALL)
        self.basicSizer4.Add(self.colWidthSizer2, proportion=1, flag=wx.EXPAND | wx.ALL)
        self.basicSizer4.Add(self.rowNbgrSizer2, proportion=1, flag=wx.EXPAND | wx.ALL)
        self.basicSizer4.Add(self.rowPowerSizer2, proportion=1, flag=wx.EXPAND | wx.ALL)
        self.basicSizer4.Add(self.rowWidthSizer2, proportion=1, flag=wx.EXPAND | wx.ALL)
        self.basicSizer4.Add(self.flagSizer2, proportion=1, flag=wx.EXPAND | wx.ALL)
        self.basicSizer4.Add(self.roiSizer2, proportion=1, flag=wx.EXPAND | wx.ALL)
        self.basicSizer4.Add(self.rotateSizer2, proportion=1, flag=wx.EXPAND | wx.ALL)

        # Add the label / input field and label / button sizers next to each other
        self.basicSizer2.Add(self.basicSizer3, proportion=1, flag=wx.EXPAND | wx.ALL)
        self.basicSizer2.Add(self.basicSizer4, proportion=1, flag=wx.EXPAND | wx.ALL)

        # Add the total parameter sizer and the 'Apply above to:' sizer
        self.basicSizer1.Add(self.basicSizer2, proportion=9, flag=wx.EXPAND | wx.ALL)
        self.basicSizer1.Add(self.applySizer, proportion=1, flag=wx.EXPAND | wx.ALL)

        self.basicPage.SetSizerAndFit(self.basicSizer1)

        # Add a text field to the 'More' page for notes
        self.histSizer = wx.BoxSizer(wx.VERTICAL)
        self.histLabel = wx.StaticText(self.morePage, label="Notes: ")
        self.histBox = wx.TextCtrl(self.morePage, style=wx.TE_PROCESS_ENTER | wx.TE_MULTILINE)
        self.histButtonSizer = wx.BoxSizer(wx.HORIZONTAL)
        self.histApplyLbl = wx.StaticText(self.morePage, label="Apply to:")
        self.histScan = wx.Button(self.morePage, label="Scan")
        self.histCustom = wx.Button(self.morePage, label="Custom")
        self.histButtonSizer.Add(self.histApplyLbl, flag=wx.RIGHT | wx.ALIGN_CENTER, border=4)
        self.histButtonSizer.Add(self.histScan, proportion=1, flag=wx.EXPAND | wx.RIGHT, border=4)
        self.histButtonSizer.Add(self.histCustom, proportion=1, flag=wx.EXPAND)
        self.histSizer.Add(self.histLabel, flag=wx.TOP | wx.LEFT, border=4)
        self.histSizer.Add(self.histBox, proportion=1, flag=wx.EXPAND | wx.ALL, border=4)
        self.histSizer.Add(self.histButtonSizer, flag=wx.EXPAND | wx.BOTTOM | wx.LEFT | wx.RIGHT, border=4)
        self.morePage.SetSizerAndFit(self.histSizer)

        self.paramSizer1.Add(self.pointParamLbl)
        self.paramSizer1.Add(self.paramBook, proportion=1, flag=wx.EXPAND | wx.LEFT, border=4)

        # Scan parameter sizers
        self.corrSizer1 = wx.BoxSizer(wx.VERTICAL)
        self.corrSizer2 = wx.BoxSizer(wx.VERTICAL)
        self.scaleSizer = wx.BoxSizer(wx.HORIZONTAL)
        self.beamSlitSizer = wx.BoxSizer(wx.HORIZONTAL)
        self.detSlitSizer = wx.BoxSizer(wx.HORIZONTAL)
        self.sampleAngleSizer = wx.BoxSizer(wx.HORIZONTAL)
        self.sampleDiameterSizer = wx.BoxSizer(wx.HORIZONTAL)
        self.samplePolygonSizer = wx.BoxSizer(wx.HORIZONTAL)
        self.badMapSizer = wx.BoxSizer(wx.HORIZONTAL)
        self.applyCorrSizer = wx.BoxSizer(wx.HORIZONTAL)

        # Panel to hold scan-level correction values for tab-traversal
        self.corrPanel = wx.Panel(self.infoPanel, style=wx.SIMPLE_BORDER | wx.TAB_TRAVERSAL)
        self.corrPanel.SetBackgroundColour(wx.WHITE)

        self.scaleField = wx.TextCtrl(self.corrPanel, style=wx.TE_PROCESS_ENTER)
        self.beamSlitField = wx.TextCtrl(self.corrPanel, style=wx.TE_PROCESS_ENTER)
        self.detSlitField = wx.TextCtrl(self.corrPanel, style=wx.TE_PROCESS_ENTER)
        self.sampleAngleField = wx.TextCtrl(self.corrPanel, style=wx.TE_PROCESS_ENTER)
        self.sampleDiameterField = wx.TextCtrl(self.corrPanel, style=wx.TE_PROCESS_ENTER)
        self.samplePolygonField = wx.TextCtrl(self.corrPanel, style=wx.TE_PROCESS_ENTER)
        self.badMapField = wx.TextCtrl(self.corrPanel, style=wx.TE_PROCESS_ENTER)

        # The 'Apply to Custom' button
        self.applyCorrLbl = wx.StaticText(self.corrPanel, label="Apply above to: ")
        self.applyCorrCustom = wx.Button(self.corrPanel, label="Custom...")

        self.applyCorrSizer.Add(self.applyCorrLbl, flag=wx.ALIGN_CENTER | wx.RIGHT, border=4)
        self.applyCorrSizer.Add(self.applyCorrCustom, proportion=1, flag=wx.EXPAND | wx.ALL)

        # How much to scale I
        self.scaleLbl1 = wx.StaticText(self.corrPanel, label="Scale: ")

        self.scaleSizer.Add(self.scaleLbl1, flag=wx.ALIGN_CENTER | wx.RIGHT, border=4)
        self.scaleSizer.Add(self.scaleField, proportion=1, flag=wx.EXPAND | wx.ALL)

        # The beam slits
        self.beamSlitLbl1 = wx.StaticText(self.corrPanel, label="Beam slits: ")

        self.beamSlitSizer.Add(self.beamSlitLbl1, flag=wx.ALIGN_CENTER | wx.RIGHT, border=4)
        self.beamSlitSizer.Add(self.beamSlitField, proportion=1, flag=wx.EXPAND | wx.ALL)

        # The detector slits
        self.detSlitLbl1 = wx.StaticText(self.corrPanel, label="Detector slits: ")

        self.detSlitSizer.Add(self.detSlitLbl1, flag=wx.ALIGN_CENTER | wx.RIGHT, border=4)
        self.detSlitSizer.Add(self.detSlitField, proportion=1, flag=wx.EXPAND | wx.ALL)

        # The sample angles
        self.sampleAngleLbl1 = wx.StaticText(self.corrPanel, label="Sample angles: ")

        self.sampleAngleSizer.Add(self.sampleAngleLbl1, flag=wx.ALIGN_CENTER | wx.RIGHT, border=4)
        self.sampleAngleSizer.Add(self.sampleAngleField, proportion=1, flag=wx.EXPAND | wx.ALL)

        # The sample diameter
        self.sampleDiameterLbl1 = wx.StaticText(self.corrPanel, label="Sample diameter: ")

        self.sampleDiameterSizer.Add(self.sampleDiameterLbl1, flag=wx.ALIGN_CENTER | wx.RIGHT, border=4)
        self.sampleDiameterSizer.Add(self.sampleDiameterField, proportion=1, flag=wx.EXPAND | wx.ALL)

        # The sample polygon
        self.samplePolygonLbl1 = wx.StaticText(self.corrPanel, label="Sample polygon: ")

        self.samplePolygonSizer.Add(self.samplePolygonLbl1, flag=wx.ALIGN_CENTER | wx.RIGHT, border=4)
        self.samplePolygonSizer.Add(self.samplePolygonField, proportion=1, flag=wx.EXPAND | wx.ALL)

        # The bad pixel map
        self.badMapLbl1 = wx.StaticText(self.corrPanel, label="Bad pixel map: ")

        self.badMapSizer.Add(self.badMapLbl1, flag=wx.ALIGN_CENTER | wx.RIGHT, border=4)
        self.badMapSizer.Add(self.badMapField, proportion=1, flag=wx.EXPAND | wx.ALL)

        # The top label
        self.scanParamLbl = wx.StaticText(self.infoPanel, label="Scan Parameters:")

        # Add each row to the larger sizer
        self.corrSizer2.Add(self.scaleSizer, proportion=1, flag=wx.EXPAND | wx.LEFT, border=4)
        self.corrSizer2.Add(self.beamSlitSizer, proportion=1, flag=wx.EXPAND | wx.LEFT, border=4)
        self.corrSizer2.Add(self.detSlitSizer, proportion=1, flag=wx.EXPAND | wx.LEFT, border=4)
        self.corrSizer2.Add(self.sampleAngleSizer, proportion=1, flag=wx.EXPAND | wx.LEFT, border=4)
        self.corrSizer2.Add(self.sampleDiameterSizer, proportion=1, flag=wx.EXPAND | wx.LEFT, border=4)
        self.corrSizer2.Add(self.samplePolygonSizer, proportion=1, flag=wx.EXPAND | wx.LEFT, border=4)
        self.corrSizer2.Add(self.badMapSizer, proportion=1, flag=wx.EXPAND | wx.LEFT, border=4)
        self.corrSizer2.Add(self.applyCorrSizer, proportion=1, flag=wx.EXPAND | wx.LEFT, border=4)

        # Calculate the appropriate sizes
        self.corrPanel.SetSizerAndFit(self.corrSizer2)

        # Add the label and the panel to be displayed
        self.corrSizer1.Add(self.scanParamLbl, flag=wx.LEFT, border=4)
        self.corrSizer1.Add(self.corrPanel, proportion=1, flag=wx.EXPAND | wx.LEFT, border=4)

        # Data information sizers and layout
        self.dataPanel = wx.Panel(self.infoPanel, style=wx.SIMPLE_BORDER)
        self.dataPanel.SetBackgroundColour(wx.WHITE)

        # Section label
        self.dataLbl = wx.StaticText(self.infoPanel, label="Point Information:")

        # Integration buttons
        self.integrateSizer = wx.BoxSizer(wx.HORIZONTAL)
        self.integrateScan = wx.Button(self.infoPanel, label="Integrate Scan")
        self.integrateCustom = wx.Button(self.infoPanel, label="Integrate Custom...")
        self.integrateSizer.Add(self.integrateScan, proportion=1, flag=wx.EXPAND)
        self.integrateSizer.Add(self.integrateCustom, proportion=1, flag=wx.EXPAND)

        # Main data sizers
        self.dataSizer1 = wx.BoxSizer(wx.VERTICAL)
        self.dataSizer2 = wx.BoxSizer(wx.VERTICAL)

        # Row sizers for point information display
        self.hklSizer = wx.BoxSizer(wx.HORIZONTAL)
        self.iSizer = wx.BoxSizer(wx.HORIZONTAL)
        self.fSizer = wx.BoxSizer(wx.HORIZONTAL)
        self.abSecSizer = wx.BoxSizer(wx.HORIZONTAL)
        self.attenSizer = wx.BoxSizer(wx.HORIZONTAL)

        # The labels for H, K, and L
        self.hLbl = wx.StaticText(self.dataPanel, label="H: ")
        self.kLbl = wx.StaticText(self.dataPanel, label="K: ")
        self.lLbl = wx.StaticText(self.dataPanel, label="L: ")

        # Adding the HKL labels to their sizer
        self.hklSizer.Add(self.hLbl, proportion=1, flag=wx.ALIGN_CENTER | wx.LEFT | wx.RIGHT, border=4)
        self.hklSizer.Add(self.kLbl, proportion=1, flag=wx.ALIGN_CENTER | wx.LEFT | wx.RIGHT, border=4)
        self.hklSizer.Add(self.lLbl, proportion=1, flag=wx.ALIGN_CENTER | wx.LEFT | wx.RIGHT, border=4)

        # The labels for I, Ierr, and Ibgr
        self.iLbl = wx.StaticText(self.dataPanel, label="I: ")
        self.iErrLbl = wx.StaticText(self.dataPanel, label="Ierr: ")
        self.iBgrLbl = wx.StaticText(self.dataPanel, label="Ibgr: ")

        # Adding the I labels to their sizer
        self.iSizer.Add(self.iLbl, proportion=1, flag=wx.ALIGN_CENTER | wx.LEFT | wx.RIGHT, border=4)
        self.iSizer.Add(self.iErrLbl, proportion=1, flag=wx.ALIGN_CENTER | wx.LEFT | wx.RIGHT, border=4)
        self.iSizer.Add(self.iBgrLbl, proportion=1, flag=wx.ALIGN_CENTER | wx.LEFT | wx.RIGHT, border=4)

        # The labels for F, Ferr, and Ctot
        self.fLbl = wx.StaticText(self.dataPanel, label="F: ")
        self.fErrLbl = wx.StaticText(self.dataPanel, label="Ferr: ")
        self.ctotLbl = wx.StaticText(self.dataPanel, label="Ctot: ")

        # Adding the F labels to their sizer
        self.fSizer.Add(self.fLbl, proportion=1, flag=wx.ALIGN_CENTER | wx.LEFT | wx.RIGHT, border=4)
        self.fSizer.Add(self.fErrLbl, proportion=1, flag=wx.ALIGN_CENTER | wx.LEFT | wx.RIGHT, border=4)
        self.fSizer.Add(self.ctotLbl, proportion=1, flag=wx.ALIGN_CENTER | wx.LEFT | wx.RIGHT, border=4)

        # The labels for alpha, beta, and seconds
        self.aLbl = wx.StaticText(self.dataPanel, label="Alpha: ")
        self.bLbl = wx.StaticText(self.dataPanel, label="Beta: ")
        self.secLbl = wx.StaticText(self.dataPanel, label="Seconds: ")

        # Adding the alpha, beta, and seconds labels to their sizer
        self.abSecSizer.Add(self.aLbl, proportion=1, flag=wx.ALIGN_CENTER | wx.LEFT | wx.RIGHT, border=4)
        self.abSecSizer.Add(self.bLbl, proportion=1, flag=wx.ALIGN_CENTER | wx.LEFT | wx.RIGHT, border=4)
        self.abSecSizer.Add(self.secLbl, proportion=1, flag=wx.ALIGN_CENTER | wx.LEFT | wx.RIGHT, border=4)
        # The labels for transmission, filters, and corrdet
        # added April 2015, JES
        self.transmLbl = wx.StaticText(self.dataPanel, label="transm: ")
        self.filtersLbl = wx.StaticText(self.dataPanel, label="filters: ")
        self.corrdetLbl = wx.StaticText(self.dataPanel, label="corrdet: ")

        # Adding the attenuator labels to their sizer
        # added April 2015, JES
        self.attenSizer.Add(self.transmLbl, proportion=1, flag=wx.ALIGN_CENTER | wx.LEFT | wx.RIGHT, border=4)
        self.attenSizer.Add(self.filtersLbl, proportion=1, flag=wx.ALIGN_CENTER | wx.LEFT | wx.RIGHT, border=4)
        self.attenSizer.Add(self.corrdetLbl, proportion=1, flag=wx.ALIGN_CENTER | wx.LEFT | wx.RIGHT, border=4)

        # Adding the row sizers to the sizer for the panel
        self.dataSizer2.Add(self.hklSizer, proportion=1, flag=wx.EXPAND | wx.LEFT, border=4)
        self.dataSizer2.Add(self.iSizer, proportion=1, flag=wx.EXPAND | wx.LEFT, border=4)
        self.dataSizer2.Add(self.fSizer, proportion=1, flag=wx.EXPAND | wx.LEFT, border=4)
        self.dataSizer2.Add(self.abSecSizer, proportion=1, flag=wx.EXPAND | wx.LEFT, border=4)
        self.dataSizer2.Add(self.attenSizer, proportion=1, flag=wx.EXPAND | wx.LEFT, border=4)

        # Calculate the appropriate sizes
        self.dataPanel.SetSizerAndFit(self.dataSizer2)

        # Add the label, point information, and integrate button to the overall sizer
        self.dataSizer1.Add(self.dataLbl, flag=wx.LEFT, border=4)
        self.dataSizer1.Add(self.dataPanel, proportion=1, flag=wx.EXPAND | wx.LEFT, border=4)
        self.dataSizer1.Add(self.integrateSizer, flag=wx.EXPAND | wx.LEFT, border=4)

        self.infoSizer1.Add(self.paramSizer1, proportion=12, flag=wx.EXPAND | wx.LEFT, border=4)
        self.infoSizer1.Add(self.corrSizer1, proportion=9, flag=wx.EXPAND)
        self.infoSizer1.Add(self.dataSizer1, proportion=6, flag=wx.EXPAND)

        self.treePanel.SetSizerAndFit(self.treeSizer)
        self.graphPanel.SetSizerAndFit(self.graphSizer)
        self.rodPanel.SetSizerAndFit(self.rodSizer)
        self.infoPanel.SetSizerAndFit(self.infoSizer1)

        self.colNbgrLbl1.SetToolTip(labelTips["colNbgr"])
        self.colPowerLbl1.SetToolTip(labelTips["colPower"])
        self.colWidthLbl1.SetToolTip(labelTips["colWidth"])
        self.rowNbgrLbl1.SetToolTip(labelTips["rowNbgr"])
        self.rowPowerLbl1.SetToolTip(labelTips["rowPower"])
        self.rowWidthLbl1.SetToolTip(labelTips["rowWidth"])
        self.flagLbl1.SetToolTip(labelTips["flag"])
        self.roiLbl1.SetToolTip(labelTips["roi"])
        self.rotateLbl1.SetToolTip(labelTips["rotate"])

        self.scaleLbl1.SetToolTip(labelTips["scale"])
        self.beamSlitLbl1.SetToolTip(labelTips["beamSlit"])
        self.detSlitLbl1.SetToolTip(labelTips["detSlit"])
        self.sampleAngleLbl1.SetToolTip(labelTips["sampleAngle"])
        self.sampleDiameterLbl1.SetToolTip(labelTips["sampleDiameter"])
        self.samplePolygonLbl1.SetToolTip(labelTips["samplePolygon"])
        self.badMapLbl1.SetToolTip(labelTips["badMap"])

        # Window bindings
        self.statusBar.Bind(wx.EVT_SIZE, self.onSize)
        self.Bind(wx.EVT_CLOSE, self.onClose)

        # Menu bindings
        self.Bind(wx.EVT_MENU, self.saveHKLEFFerr, self.saveHKLE)
        self.Bind(wx.EVT_MENU, self.saveRIDSdata, self.saveRIDS)
        self.Bind(wx.EVT_MENU, self.saveIdata, self.saveIIoIbgr)
        self.Bind(wx.EVT_MENU, self.saveHKLFFerr, self.saveHKL)
        self.Bind(wx.EVT_MENU, self.saveCTRabData, self.saveCTRab)
        self.Bind(wx.EVT_MENU, self.saveAttrFile, self.saveAttributes)
        self.Bind(wx.EVT_MENU, self.loadFileDialog, self.loadHDF)

        self.Bind(wx.EVT_MENU, self.copyFromTo, self.copyParams)

        self.Bind(wx.EVT_MENU, self.showAreaCorrection, self.viewAreaCorrection)

        self.Bind(wx.EVT_MENU, self.toggleOutputWindowVisibility, self.toggleOutputWindow)

        # Escape and delete key bindings
        self.Bind(wx.EVT_CHAR_HOOK, self.escapeKey)
        self.Bind(wx.EVT_CHAR_HOOK, self.deleteItem)

        # Freeze ROI key binding
        self.Bind(wx.EVT_CHAR_HOOK, self.freezeROI)

        # Bind switching items in the tree to updating the graphs and data fields
        self.Bind(wx.EVT_TREE_SEL_CHANGED, self.newSelected, self.hdfTree)

        # Bind the bad point toggle to updating the current selection
        self.badPointToggle.Bind(wx.EVT_CHECKBOX, self.updateBadPoint)
        self.Bind(wx.EVT_CHAR_HOOK, self.updateBadPoint)

        # Bind the image max field to updating the current selection
        self.imageMaxField.Bind(wx.EVT_TEXT_ENTER, self.updateImageMax)
        self.imageMaxField.Bind(wx.EVT_KILL_FOCUS, self.updateImageMax)
        self.imageMaxField.Bind(wx.EVT_SET_FOCUS, self.onImageMaxSetFocus)
        self.imageMaxField.Bind(wx.EVT_LEFT_DOWN, self.onImageMaxClick)

        # Bind losing focus on a field (tab or click away) to updating that field
        self.colNbgrField.Bind(wx.EVT_KILL_FOCUS, self.updateItem)
        self.colPowerField.Bind(wx.EVT_KILL_FOCUS, self.updateItem)
        self.colWidthField.Bind(wx.EVT_KILL_FOCUS, self.updateItem)
        self.rowNbgrField.Bind(wx.EVT_KILL_FOCUS, self.updateItem)
        self.rowPowerField.Bind(wx.EVT_KILL_FOCUS, self.updateItem)
        self.rowWidthField.Bind(wx.EVT_KILL_FOCUS, self.updateItem)
        self.flagField.Bind(wx.EVT_KILL_FOCUS, self.updateItem)
        self.roiField.Bind(wx.EVT_KILL_FOCUS, self.updateItem)
        self.rotateField.Bind(wx.EVT_KILL_FOCUS, self.updateItem)

        self.histBox.Bind(wx.EVT_KILL_FOCUS, self.updateItem)

        self.scaleField.Bind(wx.EVT_KILL_FOCUS, self.applyToScan)
        self.beamSlitField.Bind(wx.EVT_KILL_FOCUS, self.applyToScan)
        self.detSlitField.Bind(wx.EVT_KILL_FOCUS, self.applyToScan)
        self.sampleAngleField.Bind(wx.EVT_KILL_FOCUS, self.applyToScan)
        self.sampleDiameterField.Bind(wx.EVT_KILL_FOCUS, self.applyToScan)
        self.samplePolygonField.Bind(wx.EVT_KILL_FOCUS, self.applyToScan)
        self.badMapField.Bind(wx.EVT_KILL_FOCUS, self.applyToScan)

        # Bind pressing enter in a field to updating that field (focus remains)
        self.colNbgrField.Bind(wx.EVT_TEXT_ENTER, self.updateItem)
        self.colPowerField.Bind(wx.EVT_TEXT_ENTER, self.updateItem)
        self.colWidthField.Bind(wx.EVT_TEXT_ENTER, self.updateItem)
        self.rowNbgrField.Bind(wx.EVT_TEXT_ENTER, self.updateItem)
        self.rowPowerField.Bind(wx.EVT_TEXT_ENTER, self.updateItem)
        self.rowWidthField.Bind(wx.EVT_TEXT_ENTER, self.updateItem)
        self.flagField.Bind(wx.EVT_TEXT_ENTER, self.updateItem)
        self.roiField.Bind(wx.EVT_TEXT_ENTER, self.updateItem)
        self.rotateField.Bind(wx.EVT_TEXT_ENTER, self.updateItem)

        self.scaleField.Bind(wx.EVT_TEXT_ENTER, self.applyToScan)
        self.beamSlitField.Bind(wx.EVT_TEXT_ENTER, self.applyToScan)
        self.detSlitField.Bind(wx.EVT_TEXT_ENTER, self.applyToScan)
        self.sampleAngleField.Bind(wx.EVT_TEXT_ENTER, self.applyToScan)
        self.sampleDiameterField.Bind(wx.EVT_TEXT_ENTER, self.applyToScan)
        self.samplePolygonField.Bind(wx.EVT_TEXT_ENTER, self.applyToScan)
        self.badMapField.Bind(wx.EVT_TEXT_ENTER, self.applyToScan)

        # Bind the scan buttons to copying the chosen parameter to all the points in a scan
        self.colNbgrScan.Bind(wx.EVT_BUTTON, self.applyToScan)
        self.colPowerScan.Bind(wx.EVT_BUTTON, self.applyToScan)
        self.colWidthScan.Bind(wx.EVT_BUTTON, self.applyToScan)
        self.rowNbgrScan.Bind(wx.EVT_BUTTON, self.applyToScan)
        self.rowPowerScan.Bind(wx.EVT_BUTTON, self.applyToScan)
        self.rowWidthScan.Bind(wx.EVT_BUTTON, self.applyToScan)
        self.flagScan.Bind(wx.EVT_BUTTON, self.applyToScan)
        self.roiScan.Bind(wx.EVT_BUTTON, self.applyToScan)
        self.rotateScan.Bind(wx.EVT_BUTTON, self.applyToScan)
        self.applyScan.Bind(wx.EVT_BUTTON, self.applyToScan)

        self.histScan.Bind(wx.EVT_BUTTON, self.applyToScan)

        # Bind the custom buttons to copying the chosen parameter(s) to a chosen subset
        self.colNbgrCustom.Bind(wx.EVT_BUTTON, self.applyToCustom)
        self.colPowerCustom.Bind(wx.EVT_BUTTON, self.applyToCustom)
        self.colWidthCustom.Bind(wx.EVT_BUTTON, self.applyToCustom)
        self.rowNbgrCustom.Bind(wx.EVT_BUTTON, self.applyToCustom)
        self.rowPowerCustom.Bind(wx.EVT_BUTTON, self.applyToCustom)
        self.rowWidthCustom.Bind(wx.EVT_BUTTON, self.applyToCustom)
        self.flagCustom.Bind(wx.EVT_BUTTON, self.applyToCustom)
        self.roiCustom.Bind(wx.EVT_BUTTON, self.applyToCustom)
        self.rotateCustom.Bind(wx.EVT_BUTTON, self.applyToCustom)
        self.applyCustom.Bind(wx.EVT_BUTTON, self.applyToCustom)
        self.applyCorrCustom.Bind(wx.EVT_BUTTON, self.applyToCustom)

        self.histCustom.Bind(wx.EVT_BUTTON, self.applyToCustom)

        # Bind the 'Integrate Scan' button to integrating the currently selected scan
        self.integrateScan.Bind(wx.EVT_BUTTON, self.integrateSelectedScan)
        # Bind the integrate button to integrating a chosen subset
        self.integrateCustom.Bind(wx.EVT_BUTTON, self.applyToCustom)

        # Initially show only the main content (no output window)
        self.mainSplitter.Initialize(self.splitWindow)
        self.statusSizer.Add(self.mainSplitter, proportion=664, flag=wx.EXPAND)
        self.statusSizer.Add(self.statusBar, proportion=31, flag=wx.EXPAND)
        self.SetSizer(self.statusSizer)
        self.CenterOnScreen()
        self.Show()

    def loadFileDialog(self, event: wx.Event) -> None:
        """Open a dialog to choose an HDF file to load."""
        loadDialog = wx.FileDialog(
            self,
            message="Load file...",
            defaultDir=self.lastDirectory,
            defaultFile="",
            wildcard="HDF files (*.ph5)|*.ph5|" + "All file(*.*)|*",
            style=wx.FD_OPEN,
        )
        if loadDialog.ShowModal() == wx.ID_OK:
            if not os.path.isfile(loadDialog.GetPath()):
                print("Error: File does not exist")
                return
            # Update last directory
            self.lastDirectory = os.path.dirname(loadDialog.GetPath())

            try:
                self.hdfObject.close()
                self.lockFile.release()
                print("Lock released")
            except Exception:
                pass

            # Since hdfObject is now closed, clear all variables to prevent attempted read / writes
            self.newFile = True
            self.firstOpen = True
            self.hdfTree.DeleteAllItems()
            try:
                self.customSelection.customTree.DeleteAllItems()
            except Exception:
                pass
            del self.hdfRoot
            del self.hdfObject
            del self.hdfTreeObject
            del self.customTreeObject
            self.hdfRoot = None
            self.hdfObject = None
            self.hdfTreeObject = None
            self.customTreeObject = None

            print("Loading " + loadDialog.GetPath())
            self.loadFile(loadDialog.GetPath())
        loadDialog.Destroy()

    def loadFile(self, fileName: str) -> None:
        """Load an HDF file into an hdf_data object and parse it into a tree."""
        try:
            self.hdfObject = HdfDataFile(fileName)
        except FileLockException as e:
            print("Error: " + str(e))
            return
        except Exception:
            return
        self.hdfTreeObject = HDFToTree()
        self.hdfTreeObject.populateTree(self.hdfTree, self.hdfObject)
        self.hdfRoot = self.hdfTree.GetRootItem()
        self.hdfTree.Expand(self.hdfRoot)
        self.hdfTree.SelectItem(self.hdfRoot)

        self.customTreeObject = HDFToTree()
        self.customTreeObject.populateTree(self.customSelection.customTree, self.hdfObject)
        self.customSelection.customRoot = self.customSelection.customTree.GetRootItem()

    def escapeKey(self, event: wx.Event) -> None:
        """Return focus to the tree when the escape key is pressed."""
        event.Skip()
        if event.GetKeyCode() == wx.WXK_ESCAPE:
            self.hdfTree.SetFocus()

    def deleteItem(self, event: wx.Event) -> None:
        """Delete selected item when delete key is pressed while tree has focus."""
        event.Skip()
        if event.GetKeyCode() == wx.WXK_DELETE:
            if not self.FindFocus() == self.hdfTree:
                return
            thisItem = self.hdfTree.GetSelection()
            itemText = self.hdfTree.GetItemText(thisItem)
            confirmation = wx.MessageDialog(
                self,
                message="Are you sure you " + "want to delete " + itemText + "?",
                caption="Warning!",
                style=wx.YES_NO | wx.NO_DEFAULT,
            )
            if confirmation.ShowModal() == wx.ID_NO:
                confirmation.Destroy()
                return
            confirmation.Destroy()

            self.hdfTreeObject.deleteItem(self.hdfTree, self.hdfObject, thisItem)
            self.customSelection.customTree.DeleteAllItems()
            self.customTreeObject.populateTree(self.customSelection.customTree, self.hdfObject)
            self.customSelection.customRoot = self.customSelection.customTree.GetRootItem()

    def freezeROI(self, event: wx.Event) -> None:
        """Toggle the 'Freeze ROI' box when the 'F2' key is pressed."""
        event.Skip()
        if event.GetKeyCode() == wx.WXK_F2:
            if not self.roiField.GetValue() == "":
                self.roiFreeze.SetValue(not self.roiFreeze.GetValue())

    def updateCursorStatus(self, event: Any) -> None:
        """Update status bar text with L and F coordinates of cursor."""
        if event.inaxes:
            self.statusBar.SetStatusText("L: %3.2f, F: %3.2f" % (event.xdata, event.ydata), 1)
        else:
            self.statusBar.SetStatusText("", 1)

    @staticmethod
    def closestL(a: float, l_values: list[float]) -> float:
        """Find the L value closest to the given value a."""
        return min(l_values, key=lambda x: abs(x - a))

    def clickToSelect(self, event: Any) -> None:
        """Select the point nearest to the click location."""
        if event.inaxes:
            ofMe = self.hdfTree.GetSelection()
            myParent = self.hdfTree.GetItemParent(ofMe)
            childrenList = []
            dataLookup = {}
            item, cookie = self.hdfTree.GetFirstChild(myParent)
            while item:
                iterData = self.hdfTree.GetItemData(item)
                childrenList.append(iterData)
                dataLookup[iterData] = item
                item, cookie = self.hdfTree.GetNextChild(myParent, cookie)
            allLs = {}
            if bytes_to_str(self.hdfObject.get_all("type", [childrenList[0]])[childrenList[0]]).startswith("Escan"):
                allLs = self.hdfObject.get_all("Energy", childrenList)
            elif bytes_to_str(self.hdfObject.get_all("type", [childrenList[0]])[childrenList[0]]).startswith("ascan"):
                get_this = self.hdfObject.get_all("info", [childrenList[0]])[childrenList[0]].split()[1]
                allLs = self.hdfObject.get_all(get_this, childrenList)
            else:
                allLs = self.hdfObject.get_all("L", childrenList)

            allLs = dict([L, child] for child, L in allLs.items())
            thisPoint = self.closestL(event.xdata, allLs.keys())
            try:
                self.hdfTree.SelectItem(dataLookup[allLs[thisPoint]])
            except Exception:
                print("Error: can't locate point.")
            self.hdfTree.SetFocus()

    def copyFromTo(self, event: wx.Event) -> None:
        """Copy parameters from one set of points to another by closest L value."""
        self.customSelection.customTree.Expand(self.customSelection.customRoot)
        self.customSelection.CenterOnParent()
        self.customSelection.SetTitle("Copy from...")
        userReply = self.customSelection.ShowModal()
        if userReply == wx.ID_CANCEL:
            self.hdfTree.SetFocus()
            return
        treeSelections = self.customSelection.customTree.GetSelections()
        fromThese = []
        for selected in treeSelections:
            fromThese.extend(self.customTreeObject.getRelevantChildren(self.customSelection.customTree, self.hdfObject, selected))
        fromThese = list(set(fromThese))
        fromDict = {}
        fromDict = self.hdfObject.get_all("L", fromThese)
        fromDict = dict([L, child] for child, L in fromDict.items())
        possibleLs = fromDict.keys()

        self.customSelection.SetTitle("Apply to...")
        userReply = self.customSelection.ShowModal()
        self.hdfTree.SetFocus()
        self.customSelection.SetTitle("Choose points...")
        if userReply == wx.ID_CANCEL:
            return
        treeSelections = self.customSelection.customTree.GetSelections()
        toThese = []
        for selected in treeSelections:
            toThese.extend(self.customTreeObject.getRelevantChildren(self.customSelection.customTree, self.hdfObject, selected))
        toThese = list(set(toThese))

        for itemData in toThese:
            toL = self.hdfObject.get_all("L", [itemData])
            toL = toL[itemData]
            copyFrom = self.closestL(toL, possibleLs)
            copyData = fromDict[copyFrom]
            self.hdfObject.set_all(("det_0", "image_changed"), 1.0, [itemData])
            self.hdfObject.set_all(("det_0", "F_changed"), 1.0, [itemData])
            for key in (
                "cnbgr",
                "cpow",
                "cwidth",
                "rnbgr",
                "rpow",
                "rwidth",
                "bgrflag",
                "roi",
                "rotangle",
                "bad_point",
                "scale",
                "beam_slits",
                "det_slits",
                "sample_angles",
                "sample_polygon",
                "sample_diameter",
                "bad_pixel_map",
            ):
                holding = self.hdfObject.get_all(("det_0", key), [copyData])
                holding = holding[copyData]
                self.hdfObject.set_all(("det_0", key), holding, [itemData])
        ofMe = self.hdfTree.GetSelection()
        itemData = self.hdfTree.GetItemData(ofMe)
        if itemData is None:
            return
        myParent = self.hdfTree.GetItemParent(ofMe)
        self.updateFourPlot(myParent, itemData)
        self.updateRodPlot(myParent, itemData)
        self.updateFields(itemData)

    def buildPsicG(self, itemData: int) -> list:
        """Build data structure for gonio_psic from unparsed G array."""
        psicG = []
        # cell
        psicG.append(
            [
                self.hdfObject[itemData]["real_a"],
                self.hdfObject[itemData]["real_b"],
                self.hdfObject[itemData]["real_c"],
                self.hdfObject[itemData]["real_alpha"],
                self.hdfObject[itemData]["real_beta"],
                self.hdfObject[itemData]["real_gamma"],
                self.hdfObject[itemData]["lambda"],
            ]
        )
        # or0
        psicG.append(
            {
                "h": [self.hdfObject[itemData]["or0_h"], self.hdfObject[itemData]["or0_k"], self.hdfObject[itemData]["or0_L"]],
                "delta": self.hdfObject[itemData]["or0_del"],
                "eta": self.hdfObject[itemData]["or0_eta"],
                "chi": self.hdfObject[itemData]["or0_chi"],
                "phi": self.hdfObject[itemData]["or0_phi"],
                "nu": self.hdfObject[itemData]["or0_nu"],
                "mu": self.hdfObject[itemData]["or0_mu"],
                "lam": self.hdfObject[itemData]["or0_lambda"],
            }
        )
        # or1
        psicG.append(
            {
                "h": [self.hdfObject[itemData]["or1_h"], self.hdfObject[itemData]["or1_k"], self.hdfObject[itemData]["or1_L"]],
                "delta": self.hdfObject[itemData]["or1_del"],
                "eta": self.hdfObject[itemData]["or1_eta"],
                "chi": self.hdfObject[itemData]["or1_chi"],
                "phi": self.hdfObject[itemData]["or1_phi"],
                "nu": self.hdfObject[itemData]["or1_nu"],
                "mu": self.hdfObject[itemData]["or1_mu"],
                "lam": self.hdfObject[itemData]["or1_lambda"],
            }
        )
        # n
        psicG.append(list(self.hdfObject[itemData]["haz"]))
        return psicG

    def _createOutputWindow(self) -> None:
        """Create the output window components lazily."""
        if self.outputPanel is not None:
            return

        # Create the output window (initially hidden)
        self.outputPanel = wx.Panel(self.mainSplitter)
        self.outputText = wx.TextCtrl(self.outputPanel, style=wx.TE_MULTILINE | wx.TE_READONLY | wx.HSCROLL)

        # Set font to match the application default (same as other text controls)
        default_font = wx.SystemSettings.GetFont(wx.SYS_DEFAULT_GUI_FONT)
        self.outputText.SetFont(default_font)

        # Create clear button for output window
        self.clearOutputBtn = wx.Button(self.outputPanel, label="Clear")
        self.clearOutputBtn.Bind(wx.EVT_BUTTON, self.onClearOutput)

        # Layout for output panel
        outputSizer = wx.BoxSizer(wx.VERTICAL)
        buttonSizer = wx.BoxSizer(wx.HORIZONTAL)
        buttonSizer.Add(wx.StaticText(self.outputPanel, label="Console Output:"), 0, wx.ALIGN_CENTER_VERTICAL | wx.LEFT, 5)
        buttonSizer.AddStretchSpacer()
        buttonSizer.Add(self.clearOutputBtn, 0, wx.RIGHT, 5)
        outputSizer.Add(buttonSizer, 0, wx.EXPAND | wx.ALL, 2)
        outputSizer.Add(self.outputText, 1, wx.EXPAND | wx.ALL, 2)
        self.outputPanel.SetSizer(outputSizer)

        # Initially hide the output panel
        self.outputPanel.Hide()

        # Create console capture now that we have the text control
        self.consoleCapture = ConsoleCapture(self.outputText)

    def toggleOutputWindowVisibility(self, event: wx.Event) -> None:
        """Toggle the visibility of the stdout/stderr output window."""
        if self.outputWindowVisible:
            # Hide the output window
            self.mainSplitter.Unsplit(self.outputPanel)
            self.outputPanel.Hide()
            self.outputWindowVisible = False
            self.toggleOutputWindow.SetItemLabel("Show Output Window")
            # Disable capture when hiding
            self.disableCapture()
        else:
            # Create output window if it doesn't exist yet
            self._createOutputWindow()

            # Show the output window
            self.outputPanel.Show()
            self.mainSplitter.SplitHorizontally(self.splitWindow, self.outputPanel, -150)
            self.outputWindowVisible = True
            self.toggleOutputWindow.SetItemLabel("Hide Output Window")
            # Enable capture when showing
            self.enableCapture()

        # Force layout update
        self.Layout()

    def enableCapture(self) -> None:
        """Enable stdout/stderr capture to the output window."""
        if not self.captureEnabled:
            sys.stdout = self.consoleCapture
            sys.stderr = self.consoleCapture
            self.captureEnabled = True

    def disableCapture(self) -> None:
        """Disable stdout/stderr capture."""
        if self.captureEnabled:
            sys.stdout = self.originalStdout
            sys.stderr = self.originalStderr
            self.captureEnabled = False

    def onClearOutput(self, event: wx.Event) -> None:
        """Clear the output window."""
        self.outputText.Clear()

    def onClose(self, event: wx.Event) -> None:
        """Handle window closing by restoring stdout/stderr."""
        self.disableCapture()
        try:
            self.hdfObject.close()
            self.lockFile.release()
        except Exception:
            pass
        event.Skip()

    def showAreaCorrection(self, event: wx.Event) -> None:
        """Plot the area correction for the selected data point."""
        ofMe = self.hdfTree.GetSelection()
        itemData = self.hdfTree.GetItemData(ofMe)
        if itemData is None:
            return
        # Because the G array is already parsed, we need to build the appropriate data structures to pass to gonio_psic
        psicG = self.buildPsicG(itemData)

        psic = psic_from_spec(psicG, preparsed=True)
        angles = self.hdfObject[itemData]
        psic.set_angles(
            phi=angles.get("phi", None),
            chi=angles.get("chi", None),
            eta=angles.get("eta", None),
            mu=angles.get("mu", None),
            nu=angles.get("nu", None),
            delta=angles.get("del", None),
        )
        beam_slits = safe_eval_hdf(self.hdfObject[itemData]["det_0"]["beam_slits"])
        det_slits = safe_eval_hdf(self.hdfObject[itemData]["det_0"]["det_slits"])
        if det_slits == {}:
            det_slits = None
        sample = {
            "dia": safe_eval_hdf(self.hdfObject[itemData]["det_0"]["sample_diameter"]),
            "angles": safe_eval_hdf(self.hdfObject[itemData]["det_0"]["sample_angles"]),
            "polygon": safe_eval_hdf(self.hdfObject[itemData]["det_0"]["sample_polygon"]),
        }
        cor = CtrCorrectionPsic(gonio=psic, beam_slits=beam_slits, det_slits=det_slits, sample=sample)
        cor.ctot_stationary(plot=True)

    def updateBadPoint(self, event: wx.Event) -> None:
        """Update a point when the bad point checkbox is toggled."""
        event.Skip()
        try:
            if event.GetKeyCode() != wx.WXK_F4:
                return
            else:
                self.badPointToggle.SetValue(not self.badPointToggle.GetValue())
        except Exception:
            pass
        try:
            ofMe = self.hdfTree.GetSelection()
            itemData = self.hdfTree.GetItemData(ofMe)
        except Exception:
            print("Error getting tree selection")
            self.badPointToggle.SetValue(False)
            return
        if itemData is None:
            self.badPointToggle.SetValue(False)
            return
        myParent = self.hdfTree.GetItemParent(ofMe)
        self.hdfObject[itemData]["det_0"]["bad_point"] = str(self.badPointToggle.GetValue())
        self.updateF(itemData)
        self.updateRodPlot(myParent, itemData)
        self.updateLabels(itemData)

    def onImageMaxClick(self, event: wx.Event) -> None:
        """Handle imageMaxField mouse click to ensure it receives focus."""
        self.imageMaxField.SetFocus()
        event.Skip()

    def onImageMaxSetFocus(self, event: wx.Event) -> None:
        """Handle imageMaxField receiving focus."""
        event.Skip()

    def updateImageMax(self, event: wx.Event) -> None:
        """Update a point when the image max is changed."""
        ofMe = self.hdfTree.GetSelection()
        myParent = self.hdfTree.GetItemParent(ofMe)
        itemData = self.hdfTree.GetItemData(ofMe)
        if itemData is None:
            self.imageMaxField.Clear()
            return
        newValue = bytes_to_str(self.imageMaxField.GetValue())
        currentValue = bytes_to_str(self.hdfObject[itemData]["det_0"]["image_max"])
        # Don't revert the field on kill focus - allow the user to edit it
        # The value will be applied when they press Enter or move to another point
        if event.GetEventType() == wx.EVT_KILL_FOCUS.typeId:
            # Only update if the value actually changed
            try:
                if newValue.strip() == "":
                    self.imageMaxField.SetValue(str(currentValue))
                    return
                if str(int(newValue)) == currentValue:
                    return
                elif int(newValue) <= 0 and int(newValue) != -1:
                    self.imageMaxField.SetValue(currentValue)
                    return
                else:
                    self.hdfObject[itemData]["det_0"]["image_max"] = str(int(newValue))
                    self.updateFourPlot(myParent, itemData)
                    self.imageMaxValue.SetLabel("ROI Max: " + str(self.hdfObject[itemData]["det_0"]["real_image_max"]))
            except Exception:
                self.imageMaxField.SetValue(str(currentValue))
            return
        # Handle Enter key press
        try:
            if str(int(newValue)) == currentValue:
                return
            elif int(newValue) <= 0 and int(newValue) != -1:
                self.imageMaxField.SetValue(currentValue)
            else:
                self.hdfObject[itemData]["det_0"]["image_max"] = str(int(newValue))
                self.updateFourPlot(myParent, itemData)
                self.imageMaxValue.SetLabel("ROI Max: " + str(self.hdfObject[itemData]["det_0"]["real_image_max"]))
        except Exception:
            self.imageMaxField.SetValue(str(currentValue))

    def updateItem(self, event: wx.Event) -> None:
        """Update data when text field loses focus after validating changes."""
        event.Skip()
        try:
            ofMe = self.hdfTree.GetSelection()
            itemData = self.hdfTree.GetItemData(ofMe)
        except Exception:
            print("Error getting tree selection")
            self.clearFields()
            return
        if itemData is None:
            self.clearFields()
            return
        possibilities = {
            self.colNbgrField: ("cnbgr", int),
            self.colPowerField: ("cpow", float),
            self.colWidthField: ("cwidth", int),
            self.rowNbgrField: ("rnbgr", int),
            self.rowPowerField: ("rpow", float),
            self.rowWidthField: ("rwidth", int),
            self.flagField: ("bgrflag", int),
            self.roiField: ("roi", list),
            self.rotateField: ("rotangle", float),
        }
        myParent = self.hdfTree.GetItemParent(ofMe)
        whatField = event.GetEventObject()
        if whatField == self.histBox:
            self.hdfObject[itemData]["hist"] = str(self.histBox.GetValue())
            return
        updateThis, ofType = possibilities[whatField]

        try:
            new_value = safe_eval_hdf(whatField.GetValue())
            if ofType is list and new_value is None:
                new_value = []
            elif ofType is dict and new_value is None:
                new_value = {}
            elif new_value is not None:
                new_value = ofType(new_value)

            # Special validation for ROI field - must be a list of exactly 4 integers
            if whatField == self.roiField:
                if not isinstance(new_value, list):
                    print(f"Invalid ROI format: must be a list, got {type(new_value).__name__}")
                    # Get the current value and ensure it's properly formatted
                    current_roi = safe_eval_hdf(self.hdfObject[itemData]["det_0"][updateThis])
                    whatField.SetValue(str(current_roi) if current_roi is not None else "[]")
                    return
                if len(new_value) != 4 and len(new_value) != 0:
                    print(f"Invalid ROI format: must have 4 values [x1, y1, x2, y2] or be empty, got {len(new_value)} values")
                    # Get the current value and ensure it's properly formatted
                    current_roi = safe_eval_hdf(self.hdfObject[itemData]["det_0"][updateThis])
                    whatField.SetValue(str(current_roi) if current_roi is not None else "[]")
                    return
                if len(new_value) == 4:
                    try:
                        # Ensure all values are numeric
                        new_value = [int(v) for v in new_value]
                    except (ValueError, TypeError):
                        print("Invalid ROI format: all values must be integers")
                        # Get the current value and ensure it's properly formatted
                        current_roi = safe_eval_hdf(self.hdfObject[itemData]["det_0"][updateThis])
                        whatField.SetValue(str(current_roi) if current_roi is not None else "[]")
                        return

            current_value = safe_eval_hdf(self.hdfObject[itemData]["det_0"][updateThis])
            if current_value == new_value:
                pass
            else:
                # Always store as string in HDF5, but preserve the actual data type for comparison
                self.hdfObject[itemData]["det_0"][updateThis] = str(new_value) if new_value is not None else ""
                whatField.SetValue(str(new_value) if new_value is not None else "")
                self.hdfObject[itemData]["det_0"]["image_changed"] = "True"
                self.hdfObject[itemData]["det_0"]["F_changed"] = 1.0
                self.updateFourPlot(myParent, itemData)
                self.updateRodPlot(myParent, itemData)
                self.updateLabels(itemData)
        except Exception as e:
            print(f"Error updating {updateThis}: {e}")
            whatField.SetValue(str(self.hdfObject[itemData]["det_0"][updateThis]))

    def applyToScan(self, event: wx.Event) -> None:
        """Apply parameter values to every point in the current scan."""
        ofMe = self.hdfTree.GetSelection()
        itemData = self.hdfTree.GetItemData(ofMe)
        if itemData is None:
            self.clearFields()
            return
        possibilities = {
            self.colNbgrScan: ("cnbgr", int, self.colNbgrField),
            self.colPowerScan: ("cpow", float, self.colPowerField),
            self.colWidthScan: ("cwidth", int, self.colWidthField),
            self.rowNbgrScan: ("rnbgr", int, self.rowNbgrField),
            self.rowPowerScan: ("rpow", float, self.rowPowerField),
            self.rowWidthScan: ("rwidth", int, self.rowWidthField),
            self.flagScan: ("bgrflag", int, self.flagField),
            self.roiScan: ("roi", list, self.roiField),
            self.rotateScan: ("rotangle", float, self.rotateField),
            self.scaleField: ("scale", float, self.scaleField),
            self.beamSlitField: ("beam_slits", dict, self.beamSlitField),
            self.detSlitField: ("det_slits", dict, self.detSlitField),
            self.sampleAngleField: ("sample_angles", dict, self.sampleAngleField),
            self.sampleDiameterField: ("sample_diameter", float, self.sampleDiameterField),
            self.samplePolygonField: ("sample_polygon", list, self.samplePolygonField),
        }
        myParent = self.hdfTree.GetItemParent(ofMe)
        whatButton = event.GetEventObject()
        toChange = []
        if whatButton == self.applyScan:
            toChange = list(possibilities.values())
        elif whatButton == self.badMapField:
            if whatButton.GetValue() == "":
                whatButton.SetValue(str(self.hdfObject[itemData]["det_0"]["bad_pixel_map"]))
                return
            elif whatButton.GetValue() == str(self.hdfObject[itemData]["det_0"]["bad_pixel_map"]):
                return
            else:
                updateThese = []
                item, cookie = self.hdfTree.GetFirstChild(myParent)
                while item:
                    updateThese.append(self.hdfTree.GetItemData(item))
                    item, cookie = self.hdfTree.GetNextChild(myParent, cookie)
                del item
                allBPM = self.hdfObject.get_all(("det_0", "bad_pixel_map"), updateThese)
                updateThese = [item for item in updateThese if str(allBPM[item]) != str(whatButton.GetValue())]
                self.hdfObject.set_all(("det_0", "bad_pixel_map"), str(self.hdfObject[itemData]["det_0"]["bad_pixel_map"]), updateThese)
                self.hdfObject.set_all(("det_0", "pixel_map_changed"), "True", updateThese)
                self.hdfObject.set_all(("det_0", "image_changed"), "True", updateThese)
                self.hdfObject.set_all(("det_0", "F_changed"), 1.0, updateThese)
                self.updateFourPlot(myParent, itemData)
                self.updateF(itemData)
                self.updateRodPlot(myParent, itemData)
                return
        elif whatButton == self.histScan:
            updateThese = []
            item, cookie = self.hdfTree.GetFirstChild(myParent)
            while item:
                updateThese.append(self.hdfTree.GetItemData(item))
                item, cookie = self.hdfTree.GetNextChild(myParent, cookie)
            self.hdfObject.set_all("hist", str(self.histBox.GetValue()), updateThese)
            return
        else:
            toChange = [possibilities[whatButton]]
        if whatButton in [self.scaleField, self.beamSlitField, self.detSlitField, self.sampleAngleField, self.sampleDiameterField, self.samplePolygonField]:
            fOnly = True
        else:
            fOnly = False
        updateThese = []
        item, cookie = self.hdfTree.GetFirstChild(myParent)
        while item:
            updateThese.append(self.hdfTree.GetItemData(item))
            item, cookie = self.hdfTree.GetNextChild(myParent, cookie)
        del item
        for updateThis, ofType, whatField in toChange:
            try:
                new_value = safe_eval_hdf(whatField.GetValue())
                if ofType is list and new_value is None:
                    new_value = []
                elif ofType is dict and new_value is None:
                    new_value = {}
                elif new_value is not None:
                    new_value = ofType(new_value)

                allValues = self.hdfObject.get_all(("det_0", updateThis), updateThese)
                justThese = [item for item in updateThese if str(allValues[item]) != str(new_value)]
                self.hdfObject.set_all(("det_0", updateThis), str(new_value) if new_value is not None else "", justThese)
                if not fOnly:
                    self.hdfObject.set_all(("det_0", "image_changed"), "True", justThese)
                self.hdfObject.set_all(("det_0", "F_changed"), 1.0, justThese)
            except Exception:
                whatField.SetValue(str(self.hdfObject[itemData]["det_0"][updateThis]))
        if fOnly:
            self.updateF(itemData)
        self.updateRodPlot(myParent, itemData)
        self.updateLabels(itemData)

    def integrateStop(self, event: wx.Event) -> None:
        """Interrupt the current integration."""
        self.integrateContinue = False

    def applyToCustom(self, event: wx.Event) -> None:
        """Apply parameter values to a custom selection of scans and points."""
        ofMe = self.hdfTree.GetSelection()
        itemData = self.hdfTree.GetItemData(ofMe)
        if itemData is None and event.GetEventObject() != self.integrateCustom:
            return
        while itemData is None:
            ofMe = self.hdfTree.GetFirstChild(ofMe)[0]
            itemData = self.hdfTree.GetItemData(ofMe)
        myParent = self.hdfTree.GetItemParent(ofMe)
        if self.firstOpen:
            self.firstOpen = False
            self.customSelection.customTree.Expand(self.customSelection.customRoot)
            pickMe = self.customTreeObject.reverseLookup[itemData]
            self.customSelection.customTree.SelectItem(pickMe)
            pickMe = self.customSelection.customTree.GetItemParent(pickMe)
            while pickMe != self.customSelection.customRoot:
                self.customSelection.customTree.Expand(pickMe)
                pickMe = self.customSelection.customTree.GetItemParent(pickMe)
        self.customSelection.CenterOnParent()
        userReply = self.customSelection.ShowModal()
        self.hdfTree.SetFocus()
        if userReply == wx.ID_CANCEL:
            return

        treeSelections = self.customSelection.customTree.GetSelections()
        updateThese = []
        for selected in treeSelections:
            updateThese.extend(self.customTreeObject.getRelevantChildren(self.customSelection.customTree, self.hdfObject, selected))
        updateThese = list(set(updateThese))
        updateThese.sort()

        # If integrate custom was pressed, integrate the selected cans and return.
        # Otherwise, skip this section and apply the appropriate attribute(s) to the selected scans.
        if event.GetEventObject() == self.integrateCustom:
            self.integrateContinue = True
            integrateProgress = 0
            self.onSize(None)
            self.integrateCancel.Show()
            for iterData in updateThese:
                iterItem = self.hdfTreeObject.reverseLookup[iterData]
                iterString = self.hdfTreeObject.statusString(self.hdfTree, self.hdfObject, iterItem)
                self.statusBar.SetStatusText("Integrating " + iterString, 2)
                if not self.integrateContinue:
                    print("Integration aborted on " + iterString.lower())
                    break
                try:
                    if safe_eval_hdf(self.hdfObject[iterData]["det_0"]["image_changed"]):
                        self.integratePoint(iterData)
                    # Check F_changed flag - handle both float (1.0/0.0) and string representations
                    f_changed_val = self.hdfObject[iterData]["det_0"]["F_changed"]
                    if (isinstance(f_changed_val, (int, float)) and f_changed_val != 0) or safe_eval_hdf(f_changed_val):
                        self.updateF(iterData)
                except:
                    print("Error reading " + iterString.lower())
                    raise
                integrateProgress += 1
                while wx.GetApp().HasPendingEvents():
                    wx.GetApp().Yield(True)
            self.integrateContinue = False
            self.hdfTree.SetFocus()
            if self.hdfTree.GetSelection() == ofMe:
                self.updateRodPlot(myParent, itemData)
            self.statusBar.SetStatusText("", 2)
            self.integrateCancel.Hide()
            return
        elif event.GetEventObject() == self.histCustom:
            self.hdfObject.set_all("hist", str(self.histBox.GetValue()), updateThese)
            return

        possibilities1 = {
            self.colNbgrCustom: ("cnbgr", int, self.colNbgrField),
            self.colPowerCustom: ("cpow", float, self.colPowerField),
            self.colWidthCustom: ("cwidth", int, self.colWidthField),
            self.rowNbgrCustom: ("rnbgr", int, self.rowNbgrField),
            self.rowPowerCustom: ("rpow", float, self.rowPowerField),
            self.rowWidthCustom: ("rwidth", int, self.rowWidthField),
            self.flagCustom: ("bgrflag", int, self.flagField),
            self.roiCustom: ("roi", list, self.roiField),
            self.rotateCustom: ("rotangle", float, self.rotateField),
        }

        possibilities2 = {
            self.scaleField: ("scale", float, self.scaleField),
            self.beamSlitField: ("beam_slits", dict, self.beamSlitField),
            self.detSlitField: ("det_slits", dict, self.detSlitField),
            self.sampleAngleField: ("sample_angles", dict, self.sampleAngleField),
            self.sampleDiameterField: ("sample_diameter", float, self.sampleDiameterField),
            self.samplePolygonField: ("sample_polygon", list, self.samplePolygonField),
            self.badMapField: ("bad_pixel_map", str, self.badMapField),
        }
        whatButton = event.GetEventObject()
        toChange = []
        fOnly = False
        if whatButton == self.applyCustom:
            toChange = list(possibilities1.values())
        elif whatButton == self.applyCorrCustom:
            toChange = list(possibilities2.values())
            fOnly = True
        else:
            toChange = [possibilities1[whatButton]]
        for updateThis, ofType, whatField in toChange:
            try:
                if updateThis.startswith("bad_pixel_map"):
                    allBPM = self.hdfObject.get_all(("det_0", "bad_pixel_map"), updateThese)
                    justThese = [item for item in updateThese if str(allBPM[item]) != str(self.badMapField.GetValue())]
                    self.hdfObject.set_all(("det_0", "bad_pixel_map"), str(self.badMapField.GetValue()), justThese)
                    self.hdfObject.set_all(("det_0", "pixel_map_changed"), "True", justThese)
                    self.hdfObject.set_all(("det_0", "image_changed"), "True", justThese)
                    self.hdfObject.set_all(("det_0", "F_changed"), 1.0, justThese)
                    if justThese:
                        fOnly = False
                else:
                    new_value = safe_eval_hdf(whatField.GetValue())
                    if ofType is list and new_value is None:
                        new_value = []
                    elif ofType is dict and new_value is None:
                        new_value = {}
                    elif new_value is not None:
                        new_value = ofType(new_value)

                    allValues = self.hdfObject.get_all(("det_0", updateThis), updateThese)
                    justThese = [item for item in updateThese if str(allValues[item]) != str(new_value)]
                    self.hdfObject.set_all(("det_0", updateThis), str(new_value) if new_value is not None else "", justThese)
                    if not fOnly:
                        self.hdfObject.set_all(("det_0", "image_changed"), "True", justThese)
                    self.hdfObject.set_all(("det_0", "F_changed"), 1.0, justThese)
            except:
                print("Error updating selection")
                raise
        if fOnly:
            self.updateF(itemData)
        self.updateRodPlot(myParent, itemData)
        self.updateLabels(itemData)

    def integrateSelectedScan(self, event: wx.Event) -> None:
        """Integrate all points in the currently selected scan using threading."""
        ofMe = self.hdfTree.GetSelection()
        itemData = self.hdfTree.GetItemData(ofMe)
        updateThese = []
        if itemData is None:
            myParent = ofMe  # When itemData is None, ofMe is the parent
            firstChild, cookie = self.hdfTree.GetFirstChild(ofMe)
            childData = self.hdfTree.GetItemData(firstChild)
            if childData is None:
                return
            while firstChild:
                childData = self.hdfTree.GetItemData(firstChild)
                updateThese.append(childData)
                firstChild, cookie = self.hdfTree.GetNextChild(ofMe, cookie)
        else:
            myParent = self.hdfTree.GetItemParent(ofMe)
            item, cookie = self.hdfTree.GetFirstChild(myParent)
            while item:
                iterData = self.hdfTree.GetItemData(item)
                updateThese.append(iterData)
                item, cookie = self.hdfTree.GetNextChild(myParent, cookie)

        if len(updateThese) == 0:
            return

        # Start timing
        start_time = time.perf_counter()

        self.integrateContinue = True
        self.onSize(None)
        self.integrateCancel.Show()

        # Create queues for coordination
        task_queue = queue.Queue()
        result_queue = queue.Queue()

        # Add all points to task queue
        for iterData in updateThese:
            task_queue.put(iterData)

        total_points = len(updateThese)

        # Progress callback for GUI updates
        def update_progress(iterData, written, total):
            """Update GUI progress from writer thread."""
            if not self.integrateContinue:
                return
            try:
                iterItem = self.hdfTreeObject.reverseLookup[iterData]
                iterString = self.hdfTreeObject.statusString(self.hdfTree, self.hdfObject, iterItem)
                self.statusBar.SetStatusText(f"Integrating {iterString} ({written}/{total})", 2)
                # Process GUI events periodically
                if written % max(1, total // 10) == 0:  # Update every 10%
                    while wx.GetApp().HasPendingEvents():
                        wx.GetApp().Yield(True)
            except Exception:
                pass  # Ignore errors in progress updates

        # Start writer thread
        writer_thread = threading.Thread(target=self._write_results_worker, args=(result_queue, total_points, update_progress), daemon=True)
        writer_thread.start()

        # Process points in parallel using thread pool
        num_workers = min(4, total_points)  # Use up to 4 worker threads
        try:
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                # Submit all worker tasks
                futures = []
                for _ in range(num_workers):
                    future = executor.submit(self._integrate_point_worker, task_queue, result_queue)
                    futures.append(future)

                # Wait for all workers to complete with periodic GUI updates
                completed = set()
                while len(completed) < len(futures):
                    # Check for completed futures
                    for future in futures:
                        if future not in completed and future.done():
                            try:
                                future.result()
                            except Exception as e:
                                print(f"Worker thread error: {e}")
                                import traceback

                                traceback.print_exc()
                            completed.add(future)

                    # Yield to GUI to prevent freezing
                    if len(completed) < len(futures):
                        wx.GetApp().Yield(True)
                        time.sleep(0.01)  # Small sleep to prevent busy-waiting
        except Exception as e:
            print(f"Error in thread pool: {e}")
            import traceback

            traceback.print_exc()
            self.integrateContinue = False

        # Wait for writer thread to finish with periodic GUI updates
        try:
            timeout = 300  # 5 minute timeout
            check_interval = 0.1  # Check every 100ms
            elapsed = 0.0
            while writer_thread.is_alive() and elapsed < timeout:
                writer_thread.join(timeout=check_interval)
                elapsed += check_interval
                # Yield to GUI to prevent freezing
                wx.GetApp().Yield(True)

            if writer_thread.is_alive():
                print("Warning: Writer thread did not complete in time")
        except Exception as e:
            print(f"Error waiting for writer thread: {e}")

        self.integrateContinue = False

        # Calculate and print total integration time
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        print(f"Integration completed in {elapsed_time:.2f} seconds ({elapsed_time / 60:.2f} minutes) for {total_points} point(s)")

        self.hdfTree.SetFocus()
        if self.hdfTree.GetSelection() == ofMe:
            if itemData is not None:
                self.updateRodPlot(myParent, itemData)
            elif len(updateThese) > 0:
                # When itemData is None, use first child's data for rod plot
                self.updateRodPlot(myParent, updateThese[0])
        self.statusBar.SetStatusText("", 2)
        self.integrateCancel.Hide()
        return

    def updateFields(self, itemData: int) -> None:
        """Load parameter values from HDF object to GUI fields."""
        self.badPointToggle.SetValue(safe_eval_hdf(self.hdfObject[itemData]["det_0"]["bad_point"]))
        self.imageMaxField.SetValue(bytes_to_str(self.hdfObject[itemData]["det_0"]["image_max"]))
        self.imageMaxValue.SetLabel("ROI Max: " + bytes_to_str(self.hdfObject[itemData]["det_0"]["real_image_max"]))
        self.colNbgrField.SetValue(bytes_to_str(self.hdfObject[itemData]["det_0"]["cnbgr"]))
        self.colPowerField.SetValue(bytes_to_str(self.hdfObject[itemData]["det_0"]["cpow"]))
        self.colWidthField.SetValue(bytes_to_str(self.hdfObject[itemData]["det_0"]["cwidth"]))
        self.rowNbgrField.SetValue(bytes_to_str(self.hdfObject[itemData]["det_0"]["rnbgr"]))
        self.rowPowerField.SetValue(bytes_to_str(self.hdfObject[itemData]["det_0"]["rpow"]))
        self.rowWidthField.SetValue(bytes_to_str(self.hdfObject[itemData]["det_0"]["rwidth"]))
        self.flagField.SetValue(bytes_to_str(self.hdfObject[itemData]["det_0"]["bgrflag"]))
        # ROI field might contain lists, so use safe_eval_hdf
        roi_val = safe_eval_hdf(self.hdfObject[itemData]["det_0"]["roi"])
        self.roiField.SetValue(str(roi_val if roi_val is not None else "[]"))
        self.rotateField.SetValue(bytes_to_str(self.hdfObject[itemData]["det_0"]["rotangle"]))

        self.histBox.SetValue(bytes_to_str(self.hdfObject[itemData]["hist"]))

        # Handle None values for scan parameters
        scale_val = safe_eval_hdf(self.hdfObject[itemData]["det_0"]["scale"])
        beam_slits_val = safe_eval_hdf(self.hdfObject[itemData]["det_0"]["beam_slits"])
        det_slits_val = safe_eval_hdf(self.hdfObject[itemData]["det_0"]["det_slits"])
        sample_angles_val = safe_eval_hdf(self.hdfObject[itemData]["det_0"]["sample_angles"])
        sample_diameter_val = safe_eval_hdf(self.hdfObject[itemData]["det_0"]["sample_diameter"])
        sample_polygon_val = safe_eval_hdf(self.hdfObject[itemData]["det_0"]["sample_polygon"])
        bad_pixel_map_val = safe_eval_hdf(self.hdfObject[itemData]["det_0"]["bad_pixel_map"])

        self.scaleField.SetValue(str(scale_val if scale_val is not None else ""))
        self.beamSlitField.SetValue(str(beam_slits_val if beam_slits_val is not None else "{}"))
        self.detSlitField.SetValue(str(det_slits_val if det_slits_val is not None else "{}"))
        self.sampleAngleField.SetValue(str(sample_angles_val if sample_angles_val is not None else "{}"))
        self.sampleDiameterField.SetValue(str(sample_diameter_val if sample_diameter_val is not None else ""))
        self.samplePolygonField.SetValue(str(sample_polygon_val if sample_polygon_val is not None else "[]"))
        self.badMapField.SetValue(str(bad_pixel_map_val if bad_pixel_map_val is not None else ""))

        self.updateLabels(itemData)

    def updateLabels(self, itemData: int) -> None:
        """Update information labels with values from HDF object."""
        # print 'Updating labels'
        self.hLbl.SetLabel("H: " + str(self.hdfObject[itemData]["H"]))
        self.kLbl.SetLabel("K: " + str(self.hdfObject[itemData]["K"]))
        self.lLbl.SetLabel("L: " + str(self.hdfObject[itemData]["L"]))
        self.iLbl.SetLabel("I: " + str(round(self.hdfObject[itemData]["det_0"]["I"], 2)))
        self.iErrLbl.SetLabel("Ierr: " + str(round(self.hdfObject[itemData]["det_0"]["Ierr"], 2)))
        self.iBgrLbl.SetLabel("Ibgr: " + str(round(self.hdfObject[itemData]["det_0"]["Ibgr"], 2)))
        self.fLbl.SetLabel("F: " + str(round(self.hdfObject[itemData]["det_0"]["F"], 2)))
        self.fErrLbl.SetLabel("Ferr: " + str(round(self.hdfObject[itemData]["det_0"]["Ferr"], 2)))
        self.ctotLbl.SetLabel("Ctot: " + str(round(self.hdfObject[itemData]["det_0"]["ctot"], 2)))
        self.aLbl.SetLabel("Alpha: " + str(round(self.hdfObject[itemData]["det_0"]["alpha"], 2)))
        self.bLbl.SetLabel("Beta: " + str(round(self.hdfObject[itemData]["det_0"]["beta"], 2)))
        self.secLbl.SetLabel("Seconds: " + str(round(self.hdfObject[itemData]["Seconds"], 2)))
        self.transmLbl.SetLabel("transm: " + str(round(self.hdfObject[itemData]["transm"], 4)))
        self.filtersLbl.SetLabel("filters: " + str(round(self.hdfObject[itemData]["filters"], 2)))
        self.corrdetLbl.SetLabel("corrdet: " + str(round(self.hdfObject[itemData]["corrdet"], 2)))

    def updateF(self, itemData: int) -> None:
        """Calculate and update F value for a data point."""
        if safe_eval_hdf(self.hdfObject[itemData]["det_0"]["bad_point"]):
            self.hdfObject[itemData]["det_0"]["F"] = 0
            self.hdfObject[itemData]["det_0"]["Ferr"] = 0
            self.hdfObject[itemData]["det_0"]["F_changed"] = 0.0
            return
        sample_params = {
            "dia": safe_eval_hdf(self.hdfObject[itemData]["det_0"]["sample_diameter"]),
            "angles": safe_eval_hdf(self.hdfObject[itemData]["det_0"]["sample_angles"]),
            "polygon": safe_eval_hdf(self.hdfObject[itemData]["det_0"]["sample_polygon"]),
        }
        corr_params = {
            "scale": safe_eval_hdf(self.hdfObject[itemData]["det_0"]["scale"]),
            "geom": self.hdfObject[itemData]["geom"],
            "beam_slits": safe_eval_hdf(self.hdfObject[itemData]["det_0"]["beam_slits"]),
            "det_slits": safe_eval_hdf(self.hdfObject[itemData]["det_0"]["det_slits"]),
            "sample": sample_params,
        }
        if corr_params["det_slits"] == {}:
            corr_params["det_slits"] = None
        # TPT changed 'numPoints' to 'dims' transmission
        psicG = self.buildPsicG(itemData)
        scan_dict = {
            "I": [self.hdfObject[itemData]["det_0"]["I"]],
            "io": [self.hdfObject[itemData]["io"]],
            "Ierr": [self.hdfObject[itemData]["det_0"]["Ierr"]],
            "Ibgr": [self.hdfObject[itemData]["det_0"]["Ibgr"]],
            "dims": (1, 0),
            "transm": [self.hdfObject[itemData]["transm"]],
            "phi": float(self.hdfObject[itemData].get("phi")),
            "chi": float(self.hdfObject[itemData].get("chi")),
            "eta": float(self.hdfObject[itemData].get("eta")),
            "mu": float(self.hdfObject[itemData].get("mu")),
            "nu": float(self.hdfObject[itemData].get("nu")),
            "del": float(self.hdfObject[itemData].get("del")),
            "G": psicG,
        }
        fDict = image_point_F(scan=scan_dict, point=0, corr_params=corr_params, preparsed=True)
        self.hdfObject[itemData]["det_0"]["F"] = fDict["F"]
        self.hdfObject[itemData]["det_0"]["Ferr"] = fDict["Ferr"]
        self.hdfObject[itemData]["det_0"]["ctot"] = fDict["ctot"]
        self.hdfObject[itemData]["det_0"]["alpha"] = fDict["alpha"]
        self.hdfObject[itemData]["det_0"]["beta"] = fDict["beta"]
        self.hdfObject[itemData]["det_0"]["F_changed"] = 0.0

    def integratePoint(self, itemData: int) -> None:
        """Integrate a single data point without updating the GUI."""
        pixel_map_changed = self.hdfObject[itemData]["det_0"].get("pixel_map_changed", "False")
        if safe_eval_hdf(pixel_map_changed):
            self.hdfObject[itemData]["det_0"]["pixel_map_changed"] = "False"
            self.hdfObject.write_point(self.hdfObject[itemData])
            self.hdfObject.read_point(itemData)
        self.hdfObject[itemData]["det_0"]["image_changed"] = "False"
        self.hdfObject[itemData]["det_0"]["F_changed"] = 1.0

        # Get the corrected image data - this is already image data, not a filename
        corrected_image = self.hdfObject[itemData]["det_0"]["corrected_image"]
        if corrected_image is None:
            print(f"Warning: No corrected image data available for point {itemData}")
            # Set default values and return early
            self.hdfObject[itemData]["det_0"]["I"] = 0.0
            self.hdfObject[itemData]["det_0"]["Ierr"] = 0.0
            self.hdfObject[itemData]["det_0"]["Ibgr"] = 0.0
            return

        # Check if corrected_image is bytes literal like b'[]' which indicates missing data
        if isinstance(corrected_image, bytes) or (isinstance(corrected_image, str) and corrected_image.startswith("b'")):
            print(f"Warning: corrected_image contains bytes literal {corrected_image} for point {itemData}")
            # Set default values and return early
            self.hdfObject[itemData]["det_0"]["I"] = 0.0
            self.hdfObject[itemData]["det_0"]["Ierr"] = 0.0
            self.hdfObject[itemData]["det_0"]["Ibgr"] = 0.0
            return

        # Check if corrected_image has the expected shape attribute for numpy arrays
        if not hasattr(corrected_image, "shape"):
            print(f"Warning: corrected_image is not a numpy array for point {itemData}, type: {type(corrected_image)}")
            # Set default values and return early
            self.hdfObject[itemData]["det_0"]["I"] = 0.0
            self.hdfObject[itemData]["det_0"]["Ierr"] = 0.0
            self.hdfObject[itemData]["det_0"]["Ibgr"] = 0.0
            return

        imageAna = ImageAna(
            corrected_image,
            safe_eval_hdf(self.hdfObject[itemData]["det_0"]["roi"]),
            safe_eval_hdf(self.hdfObject[itemData]["det_0"]["rotangle"]),
            safe_eval_hdf(self.hdfObject[itemData]["det_0"]["bgrflag"]),
            safe_eval_hdf(self.hdfObject[itemData]["det_0"]["cnbgr"]),
            safe_eval_hdf(self.hdfObject[itemData]["det_0"]["cwidth"]),
            safe_eval_hdf(self.hdfObject[itemData]["det_0"]["cpow"]),
            safe_eval_hdf(self.hdfObject[itemData]["det_0"]["ctan"]),
            safe_eval_hdf(self.hdfObject[itemData]["det_0"]["rnbgr"]),
            safe_eval_hdf(self.hdfObject[itemData]["det_0"]["rwidth"]),
            safe_eval_hdf(self.hdfObject[itemData]["det_0"]["rpow"]),
            safe_eval_hdf(self.hdfObject[itemData]["det_0"]["rtan"]),
            safe_eval_hdf(self.hdfObject[itemData]["det_0"]["nline"]),
            safe_eval_hdf(self.hdfObject[itemData]["det_0"]["filter"]),
            safe_eval_hdf(self.hdfObject[itemData]["det_0"]["compress"]),
            False,  # 'plot'
            None,  # 'fig'
            "",  # 'figtitle'
            im_max=safe_eval_hdf(self.hdfObject[itemData]["det_0"]["image_max"]),
        )
        # TPT changed getVars to get_vars
        (
            holding1,  # 'clpimg',
            holding2,  # 'bgrimg',
            integrated,
            self.hdfObject[itemData]["det_0"]["I"],
            self.hdfObject[itemData]["det_0"]["Ibgr"],
            self.hdfObject[itemData]["det_0"]["Ierr"],
            self.hdfObject[itemData]["det_0"]["I_c"],
            self.hdfObject[itemData]["det_0"]["I_r"],
            self.hdfObject[itemData]["det_0"]["Ibgr_c"],
            self.hdfObject[itemData]["det_0"]["Ibgr_r"],
            self.hdfObject[itemData]["det_0"]["Ierr_c"],
            self.hdfObject[itemData]["det_0"]["Ierr_r"],
        ) = imageAna.get_vars()
        self.hdfObject[itemData]["det_0"]["integrated"] = str(integrated)
        self.updateF(itemData)

    def _read_point_data_threadsafe(self, itemData: int) -> dict[str, Any]:
        """Thread-safe method to read point data from HDF5 file."""
        with self.hdf_lock:
            # Get point dictionary reference once to avoid multiple __getitem__ calls
            # which could trigger write_point on previous point during iteration
            point_dict = self.hdfObject[itemData]

            # Read point data into a copy to avoid state conflicts
            pixel_map_changed = point_dict["det_0"].get("pixel_map_changed", "False")
            if safe_eval_hdf(pixel_map_changed):
                point_dict["det_0"]["pixel_map_changed"] = "False"
                self.hdfObject.write_point(point_dict, str(itemData))
                self.hdfObject.read_point(str(itemData))
                # Re-get the point_dict after read_point in case it changed
                point_dict = self.hdfObject[itemData]

            # Create a deep copy of the point data we need
            point_data = {
                "corrected_image": point_dict["det_0"]["corrected_image"],
                "roi": safe_eval_hdf(point_dict["det_0"]["roi"]),
                "rotangle": safe_eval_hdf(point_dict["det_0"]["rotangle"]),
                "bgrflag": safe_eval_hdf(point_dict["det_0"]["bgrflag"]),
                "cnbgr": safe_eval_hdf(point_dict["det_0"]["cnbgr"]),
                "cwidth": safe_eval_hdf(point_dict["det_0"]["cwidth"]),
                "cpow": safe_eval_hdf(point_dict["det_0"]["cpow"]),
                "ctan": safe_eval_hdf(point_dict["det_0"]["ctan"]),
                "rnbgr": safe_eval_hdf(point_dict["det_0"]["rnbgr"]),
                "rwidth": safe_eval_hdf(point_dict["det_0"]["rwidth"]),
                "rpow": safe_eval_hdf(point_dict["det_0"]["rpow"]),
                "rtan": safe_eval_hdf(point_dict["det_0"]["rtan"]),
                "nline": safe_eval_hdf(point_dict["det_0"]["nline"]),
                "filter": safe_eval_hdf(point_dict["det_0"]["filter"]),
                "compress": safe_eval_hdf(point_dict["det_0"]["compress"]),
                "image_max": safe_eval_hdf(point_dict["det_0"]["image_max"]),
                "image_changed": safe_eval_hdf(point_dict["det_0"]["image_changed"]),
                "F_changed": safe_eval_hdf(point_dict["det_0"]["F_changed"]),
            }
            return point_data

    def _process_point_integration(self, itemData: int, point_data: dict[str, Any]) -> dict[str, Any] | None:
        """Process integration for a single point. Returns results dict or None if skipped."""
        # Check if we need to integrate
        if not point_data["image_changed"]:
            return None

        corrected_image = point_data["corrected_image"]

        # Validate image data
        if corrected_image is None:
            return {
                "I": 0.0,
                "Ierr": 0.0,
                "Ibgr": 0.0,
                "I_c": 0.0,
                "Ierr_c": 0.0,
                "Ibgr_c": 0.0,
                "I_r": 0.0,
                "Ierr_r": 0.0,
                "Ibgr_r": 0.0,
                "integrated": False,
            }

        # Check if corrected_image is bytes literal like b'[]' which indicates missing data
        if isinstance(corrected_image, bytes) or (isinstance(corrected_image, str) and corrected_image.startswith("b'")):
            return {
                "I": 0.0,
                "Ierr": 0.0,
                "Ibgr": 0.0,
                "I_c": 0.0,
                "Ierr_c": 0.0,
                "Ibgr_c": 0.0,
                "I_r": 0.0,
                "Ierr_r": 0.0,
                "Ibgr_r": 0.0,
                "integrated": False,
            }

        # Check if corrected_image has the expected shape attribute for numpy arrays
        if not hasattr(corrected_image, "shape"):
            return {
                "I": 0.0,
                "Ierr": 0.0,
                "Ibgr": 0.0,
                "I_c": 0.0,
                "Ierr_c": 0.0,
                "Ibgr_c": 0.0,
                "I_r": 0.0,
                "Ierr_r": 0.0,
                "Ibgr_r": 0.0,
                "integrated": False,
            }

        # Perform integration
        imageAna = ImageAna(
            corrected_image,
            point_data["roi"],
            point_data["rotangle"],
            point_data["bgrflag"],
            point_data["cnbgr"],
            point_data["cwidth"],
            point_data["cpow"],
            point_data["ctan"],
            point_data["rnbgr"],
            point_data["rwidth"],
            point_data["rpow"],
            point_data["rtan"],
            point_data["nline"],
            point_data["filter"],
            point_data["compress"],
            False,  # 'plot'
            None,  # 'fig'
            "",  # 'figtitle'
            im_max=point_data["image_max"],
        )

        (
            holding1,  # 'clpimg',
            holding2,  # 'bgrimg',
            integrated,
            I,  # noqa: E741
            Ibgr,
            Ierr,
            I_c,
            I_r,
            Ibgr_c,
            Ibgr_r,
            Ierr_c,
            Ierr_r,
        ) = imageAna.get_vars()

        return {
            "I": I,
            "Ibgr": Ibgr,
            "Ierr": Ierr,
            "I_c": I_c,
            "I_r": I_r,
            "Ibgr_c": Ibgr_c,
            "Ibgr_r": Ibgr_r,
            "Ierr_c": Ierr_c,
            "Ierr_r": Ierr_r,
            "integrated": integrated,
        }

    def _process_point_F(self, itemData: int) -> dict[str, Any] | None:
        """Calculate F value for a point. Must be called with HDF lock held."""
        with self.hdf_lock:
            # Get point dictionary reference once to avoid multiple __getitem__ calls
            point_dict = self.hdfObject[itemData]

            if safe_eval_hdf(point_dict["det_0"]["bad_point"]):
                return {
                    "F": 0,
                    "Ferr": 0,
                    "ctot": 0.0,
                    "alpha": 0.0,
                    "beta": 0.0,
                }

            sample_params = {
                "dia": safe_eval_hdf(point_dict["det_0"]["sample_diameter"]),
                "angles": safe_eval_hdf(point_dict["det_0"]["sample_angles"]),
                "polygon": safe_eval_hdf(point_dict["det_0"]["sample_polygon"]),
            }
            corr_params = {
                "scale": safe_eval_hdf(point_dict["det_0"]["scale"]),
                "geom": point_dict["geom"],
                "beam_slits": safe_eval_hdf(point_dict["det_0"]["beam_slits"]),
                "det_slits": safe_eval_hdf(point_dict["det_0"]["det_slits"]),
                "sample": sample_params,
            }
            if corr_params["det_slits"] == {}:
                corr_params["det_slits"] = None

            psicG = self.buildPsicG(itemData)
            scan_dict = {
                "I": [point_dict["det_0"]["I"]],
                "io": [point_dict["io"]],
                "Ierr": [point_dict["det_0"]["Ierr"]],
                "Ibgr": [point_dict["det_0"]["Ibgr"]],
                "dims": (1, 0),
                "transm": [point_dict["transm"]],
                "phi": float(point_dict.get("phi")),
                "chi": float(point_dict.get("chi")),
                "eta": float(point_dict.get("eta")),
                "mu": float(point_dict.get("mu")),
                "nu": float(point_dict.get("nu")),
                "del": float(point_dict.get("del")),
                "G": psicG,
            }
            fDict = image_point_F(scan=scan_dict, point=0, corr_params=corr_params, preparsed=True)
            return {
                "F": fDict["F"],
                "Ferr": fDict["Ferr"],
                "ctot": fDict["ctot"],
                "alpha": fDict["alpha"],
                "beta": fDict["beta"],
            }

    def _write_point_results(self, itemData: int, integration_results: dict[str, Any] | None, f_results: dict[str, Any] | None) -> None:
        """Write integration and F results back to HDF5 file. Must be called with lock."""
        with self.hdf_lock:
            # Get point dictionary reference once to avoid multiple __getitem__ calls
            # which could trigger write_point on previous point during iteration
            point_dict = self.hdfObject[itemData]

            if integration_results is not None:
                point_dict["det_0"]["image_changed"] = "False"
                point_dict["det_0"]["F_changed"] = 1.0
                point_dict["det_0"]["I"] = integration_results["I"]
                point_dict["det_0"]["Ibgr"] = integration_results["Ibgr"]
                point_dict["det_0"]["Ierr"] = integration_results["Ierr"]
                point_dict["det_0"]["I_c"] = integration_results["I_c"]
                point_dict["det_0"]["I_r"] = integration_results["I_r"]
                point_dict["det_0"]["Ibgr_c"] = integration_results["Ibgr_c"]
                point_dict["det_0"]["Ibgr_r"] = integration_results["Ibgr_r"]
                point_dict["det_0"]["Ierr_c"] = integration_results["Ierr_c"]
                point_dict["det_0"]["Ierr_r"] = integration_results["Ierr_r"]
                point_dict["det_0"]["integrated"] = str(integration_results["integrated"])

            if f_results is not None:
                point_dict["det_0"]["F"] = f_results["F"]
                point_dict["det_0"]["Ferr"] = f_results["Ferr"]
                point_dict["det_0"]["ctot"] = f_results["ctot"]
                point_dict["det_0"]["alpha"] = f_results["alpha"]
                point_dict["det_0"]["beta"] = f_results["beta"]
                point_dict["det_0"]["F_changed"] = 0.0

            # Write the point data using the point number directly to avoid another __getitem__ call
            self.hdfObject.write_point(point_dict, str(itemData))

    def _integrate_point_worker(self, task_queue: queue.Queue, result_queue: queue.Queue) -> None:
        """Worker thread: reads and processes a single point."""
        while True:
            try:
                # Check for cancellation
                if not self.integrateContinue:
                    break

                try:
                    iterData = task_queue.get(timeout=0.1)
                except queue.Empty:
                    break

                try:
                    # Read point data (thread-safe)
                    point_data = self._read_point_data_threadsafe(iterData)

                    # Process integration (CPU-bound, NumPy releases GIL)
                    integration_results = self._process_point_integration(iterData, point_data)

                    # Process F calculation if needed
                    f_results = None
                    if point_data["F_changed"] or integration_results is not None:
                        # For F calculation, we need to read I values after integration
                        # So we'll do this in the writer thread after results are written
                        pass

                    # Send results to writer
                    result_queue.put((iterData, integration_results, f_results, point_data))

                except Exception as e:
                    print(f"Error processing point {iterData}: {e}")
                    import traceback

                    traceback.print_exc()
                    result_queue.put((iterData, None, None, None))
                finally:
                    task_queue.task_done()

            except Exception as e:
                print(f"Error in worker thread: {e}")
                import traceback

                traceback.print_exc()
                break

    def _write_results_worker(self, result_queue: queue.Queue, total_points: int, progress_callback) -> None:
        """Single writer thread: writes results sequentially to HDF5."""
        written = 0
        while written < total_points:
            if not self.integrateContinue:
                # Drain queue on cancellation
                try:
                    while not result_queue.empty():
                        result_queue.get_nowait()
                except queue.Empty:
                    pass
                break

            try:
                item = result_queue.get(timeout=1.0)
                iterData, integration_results, f_results, point_data = item

                try:
                    # Write integration results first
                    self._write_point_results(iterData, integration_results, None)

                    # Calculate F if needed (after I values are written)
                    f_results = None
                    if point_data and (point_data["F_changed"] or integration_results is not None):
                        f_results = self._process_point_F(iterData)
                        if f_results:
                            # Write F results using the same method for consistency
                            self._write_point_results(iterData, None, f_results)

                    written += 1

                    # Update progress on main thread
                    if progress_callback:
                        wx.CallAfter(progress_callback, iterData, written, total_points)

                except Exception as e:
                    print(f"Error writing point {iterData}: {e}")
                    import traceback

                    traceback.print_exc()

            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error in writer thread: {e}")
                import traceback

                traceback.print_exc()
                break

    def updateRodPlot(self, myParent: Any, itemData: int) -> None:
        """Update the rod plot showing L vs I/F values."""
        iterList = []
        pendingLList = []
        pendingFList = []
        pendingFerrList = []
        doneLList = []
        doneFList = []
        doneFerrList = []
        item, cookie = self.hdfTree.GetFirstChild(myParent)
        while item:
            iterData = self.hdfTree.GetItemData(item)
            iterList.append(iterData)
            item, cookie = self.hdfTree.GetNextChild(myParent, cookie)
        iterImageChanged = self.hdfObject.get_all(("det_0", "image_changed"), iterList)
        iterFChanged = self.hdfObject.get_all(("det_0", "F_changed"), iterList)
        if bytes_to_str(self.hdfObject[itemData]["type"]).startswith("Escan"):
            iterLList = {k: v / 1000.0 for k, v in self.hdfObject.get_all("Energy", iterList).items()}
        elif bytes_to_str(self.hdfObject[itemData]["type"]).startswith("ascan"):
            get_this = self.hdfObject[itemData]["info"].split()[1]
            iterLList = self.hdfObject.get_all(get_this, iterList)
        else:
            iterLList = self.hdfObject.get_all("L", iterList)

        iterFList = self.hdfObject.get_all(("det_0", "F"), iterList)
        iterFerrList = self.hdfObject.get_all(("det_0", "Ferr"), iterList)
        for key in iterImageChanged.keys():
            # Handle F_changed as float (1.0/0.0) or string - consistent boolean evaluation
            f_changed_val = iterFChanged[key]
            f_changed_bool = (isinstance(f_changed_val, (int, float)) and f_changed_val != 0) or safe_eval_hdf(f_changed_val)
            if not (safe_eval_hdf(iterImageChanged[key]) or f_changed_bool):
                doneLList.append(iterLList[key])
                doneFList.append(iterFList[key])
                doneFerrList.append(iterFerrList[key])
            else:
                pendingLList.append(iterLList[key])
                pendingFList.append(iterFList[key])
                pendingFerrList.append(iterFerrList[key])
        self.rodFig.clear()
        rodPlot = self.rodFig.add_subplot(111)
        rodPlot.plot(pendingLList, pendingFList, "0.8", marker=".", linestyle="")
        try:
            rodPlot.errorbar(pendingLList, pendingFList, pendingFerrList, fmt="0.8", linestyle="")
        except Exception:
            pass
        rodPlot.plot(doneLList, doneFList, "b.")
        try:
            rodPlot.errorbar(doneLList, doneFList, doneFerrList, fmt="b", linestyle="")
        except Exception:
            pass
        if bytes_to_str(self.hdfObject[itemData]["type"]).startswith("Escan"):
            rodPlot.plot([e / 1000.0 for e in self.hdfObject[itemData]["Energy"]], self.hdfObject[itemData]["det_0"]["F"], "ro")
        elif bytes_to_str(self.hdfObject[itemData]["type"]).startswith("ascan"):
            rodPlot.plot(self.hdfObject[itemData][get_this], self.hdfObject[itemData]["det_0"]["F"], "ro")
        else:
            rodPlot.plot(self.hdfObject[itemData]["L"], self.hdfObject[itemData]["det_0"]["F"], "ro")
        try:
            if not bytes_to_str(self.hdfObject[itemData]["type"]).startswith("Escan") and not bytes_to_str(self.hdfObject[itemData]["type"]).startswith(
                "ascan"
            ):
                rodPlot.semilogy()
        except Exception:
            pass
        doneLList.extend(pendingLList)
        doneFList.extend(pendingFList)
        if doneLList:
            minL = math.floor(min(doneLList))
            maxL = math.ceil(max(doneLList))
            try:
                minF = min([f for f in doneFList if f > 0]) / 10.0**0.1
            except Exception:
                minF = 0
            maxF = max(doneFList) * (10**0.1)
            if minF == 0 and maxF == 0:
                minF = 0.1
                maxF = 1
            rodPlot.axis([minL, maxL, minF, maxF])
        self.rodCanvas.draw()

    def updateFourPlot(self, myParent: Any, itemData: int) -> None:
        """Update the four-panel plot showing image, ROI, and profile data."""

        def updateROIFromClick(eclick: Any, erelease: Any) -> None:
            x1, y1 = eclick.xdata, eclick.ydata
            x2, y2 = erelease.xdata, erelease.ydata
            thisROI = str(list(map(int, map(round, [x1, y1, x2, y2]))))
            self.hdfObject[itemData]["det_0"]["roi"] = thisROI
            thisROI = self.hdfObject[itemData]["det_0"]["roi"]
            self.roiField.SetValue(thisROI)
            self.hdfObject[itemData]["det_0"]["image_changed"] = "True"
            self.hdfObject[itemData]["det_0"]["F_changed"] = "True"
            self.updateFourPlot(myParent, itemData)
            self.imageMaxValue.SetLabel("ROI Max: " + str(self.hdfObject[itemData]["det_0"]["real_image_max"]))
            self.updateRodPlot(myParent, itemData)
            self.updateLabels(itemData)
            self.hdfTree.SetFocus()

        pixel_map_changed = self.hdfObject[itemData]["det_0"].get("pixel_map_changed", "False")
        if safe_eval_hdf(pixel_map_changed):
            self.hdfObject[itemData]["det_0"]["pixel_map_changed"] = "False"
            self.hdfObject.write_point(self.hdfObject[itemData])
            self.hdfObject.read_point(itemData)
        self.fig4.clear()
        if not safe_eval_hdf(self.hdfObject[itemData]["det_0"]["image_changed"]):
            # Get the corrected image data - this is already image data, not a filename
            corrected_image = self.hdfObject[itemData]["det_0"]["corrected_image"]
            if corrected_image is None:
                print(f"Warning: No corrected image data available for point {itemData}")
                return

            # Check if corrected_image is bytes literal like b'[]' which indicates missing data
            if isinstance(corrected_image, bytes) or (isinstance(corrected_image, str) and corrected_image.startswith("b'")):
                print(f"Warning: corrected_image contains bytes literal {corrected_image} for point {itemData}")
                return

            # Check if corrected_image has the expected shape attribute for numpy arrays
            if not hasattr(corrected_image, "shape"):
                print(f"Warning: corrected_image is not a numpy array for point {itemData}, type: {type(corrected_image)}")
                return

            imageAna = ImageAna(
                corrected_image,
                safe_eval_hdf(self.hdfObject[itemData]["det_0"]["roi"]),
                safe_eval_hdf(self.hdfObject[itemData]["det_0"]["rotangle"]),
                safe_eval_hdf(self.hdfObject[itemData]["det_0"]["bgrflag"]),
                safe_eval_hdf(self.hdfObject[itemData]["det_0"]["cnbgr"]),
                safe_eval_hdf(self.hdfObject[itemData]["det_0"]["cwidth"]),
                safe_eval_hdf(self.hdfObject[itemData]["det_0"]["cpow"]),
                safe_eval_hdf(self.hdfObject[itemData]["det_0"]["ctan"]),
                safe_eval_hdf(self.hdfObject[itemData]["det_0"]["rnbgr"]),
                safe_eval_hdf(self.hdfObject[itemData]["det_0"]["rwidth"]),
                safe_eval_hdf(self.hdfObject[itemData]["det_0"]["rpow"]),
                safe_eval_hdf(self.hdfObject[itemData]["det_0"]["rtan"]),
                safe_eval_hdf(self.hdfObject[itemData]["det_0"]["nline"]),
                safe_eval_hdf(self.hdfObject[itemData]["det_0"]["filter"]),
                safe_eval_hdf(self.hdfObject[itemData]["det_0"]["compress"]),
                False,  # 'plot'
                None,  # 'fig'
                "",  # 'figtitle'
                None,  # 'clpimg'
                None,  # 'bgrimg'
                False,  # 'integrated'
                self.hdfObject[itemData]["det_0"]["I"],
                self.hdfObject[itemData]["det_0"]["Ibgr"],
                self.hdfObject[itemData]["det_0"]["Ierr"],
                self.hdfObject[itemData]["det_0"]["I_c"],
                self.hdfObject[itemData]["det_0"]["I_r"],
                self.hdfObject[itemData]["det_0"]["Ibgr_c"],
                self.hdfObject[itemData]["det_0"]["Ibgr_r"],
                self.hdfObject[itemData]["det_0"]["Ierr_c"],
                self.hdfObject[itemData]["det_0"]["Ierr_r"],
                safe_eval_hdf(self.hdfObject[itemData]["det_0"]["image_max"]),
            )
        else:
            self.hdfObject[itemData]["det_0"]["image_changed"] = "False"
            self.hdfObject[itemData]["det_0"]["F_changed"] = 1.0

            # Get the corrected image data - this is already image data, not a filename
            corrected_image = self.hdfObject[itemData]["det_0"]["corrected_image"]
            if corrected_image is None:
                print(f"Warning: No corrected image data available for point {itemData}")
                return

            # Check if corrected_image is bytes literal like b'[]' which indicates missing data
            if isinstance(corrected_image, bytes) or (isinstance(corrected_image, str) and corrected_image.startswith("b'")):
                print(f"Warning: corrected_image contains bytes literal {corrected_image} for point {itemData}")
                return

            # Check if corrected_image has the expected shape attribute for numpy arrays
            if not hasattr(corrected_image, "shape"):
                print(f"Warning: corrected_image is not a numpy array for point {itemData}, type: {type(corrected_image)}")
                return

            imageAna = ImageAna(
                corrected_image,
                safe_eval_hdf(self.hdfObject[itemData]["det_0"]["roi"]),
                safe_eval_hdf(self.hdfObject[itemData]["det_0"]["rotangle"]),
                safe_eval_hdf(self.hdfObject[itemData]["det_0"]["bgrflag"]),
                safe_eval_hdf(self.hdfObject[itemData]["det_0"]["cnbgr"]),
                safe_eval_hdf(self.hdfObject[itemData]["det_0"]["cwidth"]),
                safe_eval_hdf(self.hdfObject[itemData]["det_0"]["cpow"]),
                safe_eval_hdf(self.hdfObject[itemData]["det_0"]["ctan"]),
                safe_eval_hdf(self.hdfObject[itemData]["det_0"]["rnbgr"]),
                safe_eval_hdf(self.hdfObject[itemData]["det_0"]["rwidth"]),
                safe_eval_hdf(self.hdfObject[itemData]["det_0"]["rpow"]),
                safe_eval_hdf(self.hdfObject[itemData]["det_0"]["rtan"]),
                safe_eval_hdf(self.hdfObject[itemData]["det_0"]["nline"]),
                safe_eval_hdf(self.hdfObject[itemData]["det_0"]["filter"]),
                safe_eval_hdf(self.hdfObject[itemData]["det_0"]["compress"]),
                False,  # 'plot'
                None,  # 'fig'
                "",  # 'figtitle'
                im_max=safe_eval_hdf(self.hdfObject[itemData]["det_0"]["image_max"]),
            )
            # TPT changed getVars to get_vars
            (
                holding1,  # 'clpimg',
                holding2,  # 'bgrimg',
                self.hdfObject[itemData]["det_0"]["integrated"],
                self.hdfObject[itemData]["det_0"]["I"],
                self.hdfObject[itemData]["det_0"]["Ibgr"],
                self.hdfObject[itemData]["det_0"]["Ierr"],
                self.hdfObject[itemData]["det_0"]["I_c"],
                self.hdfObject[itemData]["det_0"]["I_r"],
                self.hdfObject[itemData]["det_0"]["Ibgr_c"],
                self.hdfObject[itemData]["det_0"]["Ibgr_r"],
                self.hdfObject[itemData]["det_0"]["Ierr_c"],
                self.hdfObject[itemData]["det_0"]["Ierr_r"],
            ) = imageAna.get_vars()
        # TPT rename embedPlot to embed_plot
        (im_max, colormap, subplot2) = imageAna.embed_plot(self.fig4)  # 'colormap',
        self.hdfObject[itemData]["det_0"]["real_image_max"] = str(im_max)
        # Check F_changed flag - handle both float (1.0/0.0) and string representations
        f_changed_val = self.hdfObject[itemData]["det_0"]["F_changed"]
        if (isinstance(f_changed_val, (int, float)) and f_changed_val != 0) or safe_eval_hdf(f_changed_val):
            self.updateF(itemData)
        toggle_selector.RS = RectangleSelector(
            subplot2,
            updateROIFromClick,
            useblit=True,
            button=[1],
            minspanx=5,
            minspany=5,
            spancoords="pixels",
            interactive=True,
            props={"edgecolor": "red", "alpha": 1, "fill": False},
        )
        self.canvas4.draw()

    def clearFields(self) -> None:
        """Clear all GUI input fields when no scan point is selected."""
        self.badPointToggle.SetValue(False)
        if not self.keepMaxToggle.GetValue():
            self.imageMaxField.Clear()

        if not self.colNbgrFreeze.GetValue():
            self.colNbgrField.Clear()
        if not self.colPowerFreeze.GetValue():
            self.colPowerField.Clear()
        if not self.colWidthFreeze.GetValue():
            self.colWidthField.Clear()
        if not self.rowNbgrFreeze.GetValue():
            self.rowNbgrField.Clear()
        if not self.rowPowerFreeze.GetValue():
            self.rowPowerField.Clear()
        if not self.rowWidthFreeze.GetValue():
            self.rowWidthField.Clear()
        if not self.flagFreeze.GetValue():
            self.flagField.Clear()
        if not self.roiFreeze.GetValue():
            self.roiField.Clear()
        if not self.rotateFreeze.GetValue():
            self.rotateField.Clear()

        self.histBox.Clear()

        self.scaleField.Clear()
        self.beamSlitField.Clear()
        self.detSlitField.Clear()
        self.sampleAngleField.Clear()
        self.sampleDiameterField.Clear()
        self.samplePolygonField.Clear()
        self.badMapField.Clear()

        self.hLbl.SetLabel("H: ")
        self.kLbl.SetLabel("K: ")
        self.lLbl.SetLabel("L: ")
        self.iLbl.SetLabel("I: ")
        self.iErrLbl.SetLabel("Ierr: ")
        self.iBgrLbl.SetLabel("Ibgr: ")
        self.fLbl.SetLabel("F: ")
        self.fErrLbl.SetLabel("Ferr: ")
        self.ctotLbl.SetLabel("Ctot: ")
        self.aLbl.SetLabel("Alpha: ")
        self.bLbl.SetLabel("Beta: ")
        self.secLbl.SetLabel("Seconds: ")
        self.aLbl.SetLabel("transm: ")
        self.bLbl.SetLabel("filters: ")
        self.secLbl.SetLabel("corrdet: ")

    def newSelected(self, event: wx.Event) -> None:
        """Handle new tree item selection by updating plots and fields."""
        ofMe = event.GetItem()
        myParent = self.hdfTree.GetItemParent(ofMe)
        itemData = self.hdfTree.GetItemData(ofMe)
        if itemData is None:
            self.fig4.clear()
            self.canvas4.draw()
            self.rodFig.clear()
            self.rodCanvas.draw()
            self.clearFields()
            self.statusBar.SetStatusText(self.hdfTree.GetItemText(ofMe))
        else:
            pixel_map_changed = self.hdfObject[itemData]["det_0"].get("pixel_map_changed", "False")
            if safe_eval_hdf(pixel_map_changed):
                self.hdfObject[itemData]["det_0"]["pixel_map_changed"] = "False"
            if self.keepMaxToggle.GetValue():
                self.hdfObject[itemData]["det_0"]["image_max"] = str(self.imageMaxField.GetValue())
            possibilities = {
                self.colNbgrFreeze: (self.colNbgrField, "cnbgr", int),
                self.colPowerFreeze: (self.colPowerField, "cpow", float),
                self.colWidthFreeze: (self.colWidthField, "cwidth", int),
                self.rowNbgrFreeze: (self.rowNbgrField, "rnbgr", int),
                self.rowPowerFreeze: (self.rowPowerField, "rpow", float),
                self.rowWidthFreeze: (self.rowWidthField, "rwidth", int),
                self.flagFreeze: (self.flagField, "bgrflag", int),
                self.roiFreeze: (self.roiField, "roi", list),
                self.rotateFreeze: (self.rotateField, "rotangle", float),
            }
            for key in possibilities:
                if key.GetValue():
                    whatField, updateThis, ofType = possibilities[key]
                    new_value = safe_eval_hdf(whatField.GetValue())
                    if ofType is list and new_value is None:
                        new_value = []
                    elif ofType is dict and new_value is None:
                        new_value = {}
                    elif new_value is not None:
                        new_value = ofType(new_value)

                    if self.hdfObject[itemData]["det_0"][updateThis] == str(new_value):
                        pass
                    else:
                        self.hdfObject[itemData]["det_0"][updateThis] = str(new_value) if new_value is not None else ""
                        whatField.SetValue(str(new_value) if new_value is not None else "")
                        self.hdfObject[itemData]["det_0"]["image_changed"] = "True"
                        self.hdfObject[itemData]["det_0"]["F_changed"] = 1.0
            self.updateFourPlot(myParent, itemData)
            self.updateRodPlot(myParent, itemData)
            self.updateFields(itemData)
            statusText = self.hdfTreeObject.statusString(self.hdfTree, self.hdfObject, ofMe)
            self.statusBar.SetStatusText(statusText)

    def saveAttrFile(self, event: wx.Event) -> None:
        """Save current point's attributes to a tab-delimited file."""
        try:
            ofMe = self.hdfTree.GetSelection()
            itemData = self.hdfTree.GetItemData(ofMe)
        except Exception:
            print("Error getting tree selection")
            return
        if itemData is None:
            return

        saveDialog = wx.FileDialog(
            self,
            message="Save file...",
            defaultDir=self.lastDirectory,
            defaultFile="",
            wildcard="txt files (*.txt)|*.txt|" + "All files (*.*)|*",
            style=wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT,
        )
        if saveDialog.ShowModal() == wx.ID_OK:
            # Update last directory
            self.lastDirectory = os.path.dirname(saveDialog.GetPath())
            print("Saving attribute file to" + saveDialog.GetPath())
            try:
                attributeFile = open(saveDialog.GetPath(), "w")
            except Exception:
                print("Error opening attribute file")
                saveDialog.Destroy()
                raise
            try:
                for key in [
                    "bad_pixel_map",
                    "beam_slits",
                    "bgrflag",
                    "cnbgr",
                    "cpow",
                    "ctan",
                    "cwidth",
                    "det_slits",
                    "rnbgr",
                    "roi",
                    "rotangle",
                    "rpow",
                    "rtan",
                    "rwidth",
                    "sample_angles",
                    "sample_diameter",
                    "sample_polygon",
                    "scale",
                ]:
                    value = self.hdfObject[itemData]["det_0"][key]
                    value = value.decode("utf-8")
                    attributeFile.write(key + "\t" + value + "\n")
                attributeFile.write("geom\t" + bytes_to_str(self.hdfObject[itemData]["geom"]))
                print("Attribute file saved to" + saveDialog.GetPath())
            except Exception:
                print("Error writing to file")
                attributeFile.close()
                saveDialog.Destroy()
                raise
            attributeFile.close()
        saveDialog.Destroy()

    def saveHKLFFerr(self, event: wx.Event) -> None:
        """Save H, K, L, F, and Ferr values to file."""
        self.customSelection.customTree.Expand(self.customSelection.customRoot)
        self.customSelection.CenterOnParent()
        userReply = self.customSelection.ShowModal()
        if userReply == wx.ID_CANCEL:
            self.hdfTree.SetFocus()
            return
        treeSelections = self.customSelection.customTree.GetSelections()
        saveThese = []
        for selected in treeSelections:
            saveThese.extend(self.customTreeObject.getRelevantChildren(self.customSelection.customTree, self.hdfObject, selected))
        saveThese = list(set(saveThese))
        saveThese.sort()

        saveDialog = wx.FileDialog(
            self,
            message="Save file as...",
            defaultDir=self.lastDirectory,
            defaultFile="",
            wildcard="lst files (*.lst)|*.lst|" + "All files (*.*)|*",
            style=wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT,
        )
        if saveDialog.ShowModal() == wx.ID_OK:
            fname = saveDialog.GetPath()
            # Update last directory
            self.lastDirectory = os.path.dirname(fname)
            print("Saving H, K, L, F, and Ferr values to" + saveDialog.GetPath())
            try:
                allBadPs = self.hdfObject.get_all(("det_0", "bad_point"), saveThese)
                allHs = self.hdfObject.get_all("H", saveThese)
                allKs = self.hdfObject.get_all("K", saveThese)
                allLs = self.hdfObject.get_all("L", saveThese)
                allFs = self.hdfObject.get_all(("det_0", "F"), saveThese)
                allFerrs = self.hdfObject.get_all(("det_0", "Ferr"), saveThese)
                f = open(fname, "w")
                header = "  #idx %5s %5s %5s %7s %7s\n" % ("H", "K", "L", "F", "Ferr")
                f.write(header)
                for iterData in saveThese:
                    if not safe_eval_hdf(allBadPs[iterData]):
                        line = "%6s %3.2f %3.2f %6.3f %6.6g %6.6g\n" % (
                            iterData,
                            allHs[iterData],
                            allKs[iterData],
                            allLs[iterData],
                            allFs[iterData],
                            allFerrs[iterData],
                        )
                        f.write(line)
                print("Saved H, K, L, F, and Ferr values to" + saveDialog.GetPath())
                f.close()
            except Exception:
                oops = wx.MessageDialog(self, "Error saving file\n" + str(Exception))
                oops.ShowModal()
                oops.Destroy()
                raise
        saveDialog.Destroy()

    def saveHKLEFFerr(self, event: wx.Event) -> None:
        """Save H, K, L, E, F, and Ferr values to file."""
        self.customSelection.customTree.Expand(self.customSelection.customRoot)
        self.customSelection.CenterOnParent()
        userReply = self.customSelection.ShowModal()
        if userReply == wx.ID_CANCEL:
            self.hdfTree.SetFocus()
            return
        treeSelections = self.customSelection.customTree.GetSelections()
        saveThese = []
        for selected in treeSelections:
            saveThese.extend(self.customTreeObject.getRelevantChildren(self.customSelection.customTree, self.hdfObject, selected))
        saveThese = list(set(saveThese))
        saveThese.sort()

        saveDialog = wx.FileDialog(
            self,
            message="Save file as...",
            defaultDir=self.lastDirectory,
            defaultFile="",
            wildcard="lst files (*.lst)|*.lst|" + "All files (*.*)|*",
            style=wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT,
        )
        if saveDialog.ShowModal() == wx.ID_OK:
            fname = saveDialog.GetPath()
            # Update last directory
            self.lastDirectory = os.path.dirname(fname)
            print("Saving H, K, L, E, F, and Ferr values to" + saveDialog.GetPath())
            try:
                allBadPs = self.hdfObject.get_all(("det_0", "bad_point"), saveThese)
                allHs = self.hdfObject.get_all("H", saveThese)
                allKs = self.hdfObject.get_all("K", saveThese)
                allLs = self.hdfObject.get_all("L", saveThese)
                allEs = self.hdfObject.get_all("Energy", saveThese)
                allFs = self.hdfObject.get_all(("det_0", "F"), saveThese)
                allFerrs = self.hdfObject.get_all(("det_0", "Ferr"), saveThese)
                f = open(fname, "w")
                header = "  #idx %5s %5s %5s %5s %7s %7s\n" % ("H", "K", "L", "E", "F", "Ferr")
                f.write(header)
                for iterData in saveThese:
                    if not safe_eval_hdf(allBadPs[iterData]):
                        line = "%6s %3.2f %3.2f %6.3f %6.5f %6.6g %6.6g\n" % (
                            iterData,
                            allHs[iterData],
                            allKs[iterData],
                            allLs[iterData],
                            allEs[iterData],
                            allFs[iterData],
                            allFerrs[iterData],
                        )
                        f.write(line)
                print("Saved H, K, L, E, F, and Ferr values to" + saveDialog.GetPath())
                f.close()
            except Exception:
                oops = wx.MessageDialog(self, "Error saving file\n" + str(Exception))
                oops.ShowModal()
                oops.Destroy()
                raise
        saveDialog.Destroy()

    def saveRIDSdata(self, event: wx.Event) -> None:
        """Save RIDS format data with E, H, K, L, F, Ferr, alpha, beta values."""
        self.customSelection.customTree.Expand(self.customSelection.customRoot)
        self.customSelection.CenterOnParent()
        userReply = self.customSelection.ShowModal()
        if userReply == wx.ID_CANCEL:
            self.hdfTree.SetFocus()
            return
        treeSelections = self.customSelection.customTree.GetSelections()
        saveThese = []
        for selected in treeSelections:
            saveThese.extend(self.customTreeObject.getRelevantChildren(self.customSelection.customTree, self.hdfObject, selected))
        saveThese = list(set(saveThese))
        saveThese.sort()

        saveDialog = wx.FileDialog(
            self,
            message="Save file as...",
            defaultDir=self.lastDirectory,
            defaultFile="",
            wildcard="rsd files (*.rsd)|*.rsd|" + "All files (*.*)|*",
            style=wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT,
        )
        if saveDialog.ShowModal() == wx.ID_OK:
            fname = saveDialog.GetPath()
            # Update last directory
            self.lastDirectory = os.path.dirname(fname)
            print("Saving RIDS to" + saveDialog.GetPath())
            try:
                allBadPs = self.hdfObject.get_all(("det_0", "bad_point"), saveThese)
                allHs = self.hdfObject.get_all("H", saveThese)
                allKs = self.hdfObject.get_all("K", saveThese)
                allLs = self.hdfObject.get_all("L", saveThese)
                allEs = self.hdfObject.get_all("Energy", saveThese)
                allFs = self.hdfObject.get_all(("det_0", "F"), saveThese)
                allFerrs = self.hdfObject.get_all(("det_0", "Ferr"), saveThese)
                allAlphas = self.hdfObject.get_all(("det_0", "alpha"), saveThese)
                allBetas = self.hdfObject.get_all(("det_0", "beta"), saveThese)
                f = open(fname, "w")
                header = "#%5s %5s %5s %5s %7s %7s %7s %7s\n" % ("E", "H", "K", "L", "F", "Ferr", "alpha", "beta")
                f.write(header)
                for iterData in saveThese:
                    if not safe_eval_hdf(allBadPs[iterData]):
                        line = "%6.5f %3.2f %3.2f %6.2f %6.6g %6.6g %6.6g %6.6g\n" % (
                            allEs[iterData],
                            allHs[iterData],
                            allKs[iterData],
                            allLs[iterData],
                            allFs[iterData],
                            allFerrs[iterData],
                            allAlphas[iterData],
                            allBetas[iterData],
                        )
                        f.write(line)
                print("Saved RIDS to" + saveDialog.GetPath())
                f.close()
            except Exception:
                oops = wx.MessageDialog(self, "Error saving file\n" + str(Exception))
                oops.ShowModal()
                oops.Destroy()
                raise
        saveDialog.Destroy()

    def saveCTRabData(self, event: wx.Event) -> None:
        """Save CTR data with H, K, L, F, Ferr, alpha, beta values."""
        self.customSelection.customTree.Expand(self.customSelection.customRoot)
        self.customSelection.CenterOnParent()
        userReply = self.customSelection.ShowModal()
        if userReply == wx.ID_CANCEL:
            self.hdfTree.SetFocus()
            return
        treeSelections = self.customSelection.customTree.GetSelections()
        saveThese = []
        for selected in treeSelections:
            saveThese.extend(self.customTreeObject.getRelevantChildren(self.customSelection.customTree, self.hdfObject, selected))
        saveThese = list(set(saveThese))
        saveThese.sort()

        saveDialog = wx.FileDialog(
            self,
            message="Save file as...",
            defaultDir=self.lastDirectory,
            defaultFile="",
            wildcard="lst files (*.lst)|*.lst|" + "All files (*.*)|*",
            style=wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT,
        )
        if saveDialog.ShowModal() == wx.ID_OK:
            fname = saveDialog.GetPath()
            # Update last directory
            self.lastDirectory = os.path.dirname(fname)
            print("Saving CTR, Alpha, and Beta values to" + saveDialog.GetPath())
            try:
                allBadPs = self.hdfObject.get_all(("det_0", "bad_point"), saveThese)
                allHs = self.hdfObject.get_all("H", saveThese)
                allKs = self.hdfObject.get_all("K", saveThese)
                allLs = self.hdfObject.get_all("L", saveThese)
                allFs = self.hdfObject.get_all(("det_0", "F"), saveThese)
                allFerrs = self.hdfObject.get_all(("det_0", "Ferr"), saveThese)
                allAlphas = self.hdfObject.get_all(("det_0", "alpha"), saveThese)
                allBetas = self.hdfObject.get_all(("det_0", "beta"), saveThese)
                f = open(fname, "w")
                header = "#%5s %5s %5s %7s %7s %7s %7s\n" % ("H", "K", "L", "F", "Ferr", "alpha", "beta")
                f.write(header)
                for iterData in saveThese:
                    if not safe_eval_hdf(allBadPs[iterData]):
                        line = "%3.2f %3.2f %6.2f %6.6g %6.6g %6.6g %6.6g\n" % (
                            allHs[iterData],
                            allKs[iterData],
                            allLs[iterData],
                            allFs[iterData],
                            allFerrs[iterData],
                            allAlphas[iterData],
                            allBetas[iterData],
                        )
                        f.write(line)
                print("Saved CTR, Alpha, and Beta values to" + saveDialog.GetPath())
                f.close()
            except Exception:
                oops = wx.MessageDialog(self, "Error saving file\n" + str(Exception))
                oops.ShowModal()
                oops.Destroy()
                raise
        saveDialog.Destroy()

    def saveIdata(self, event: wx.Event) -> None:
        """Save intensity data with H, K, L, F, Ferr, I, Io, Ibgr values."""
        self.customSelection.customTree.Expand(self.customSelection.customRoot)
        self.customSelection.CenterOnParent()
        userReply = self.customSelection.ShowModal()
        if userReply == wx.ID_CANCEL:
            self.hdfTree.SetFocus()
            return
        treeSelections = self.customSelection.customTree.GetSelections()
        saveThese = []
        for selected in treeSelections:
            saveThese.extend(self.customTreeObject.getRelevantChildren(self.customSelection.customTree, self.hdfObject, selected))
        saveThese = list(set(saveThese))
        saveThese.sort()

        saveDialog = wx.FileDialog(
            self,
            message="Save file as...",
            defaultDir=self.lastDirectory,
            defaultFile="",
            wildcard="int files (*.int)|*.int|" + "All files (*.*)|*",
            style=wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT,
        )
        if saveDialog.ShowModal() == wx.ID_OK:
            fname = saveDialog.GetPath()
            # Update last directory
            self.lastDirectory = os.path.dirname(fname)
            print("Saving Intesity data to" + saveDialog.GetPath())
            try:
                allBadPs = self.hdfObject.get_all(("det_0", "bad_point"), saveThese)
                allHs = self.hdfObject.get_all("H", saveThese)
                allKs = self.hdfObject.get_all("K", saveThese)
                allLs = self.hdfObject.get_all("L", saveThese)
                allIs = self.hdfObject.get_all(("det_0", "I"), saveThese)
                allIos = self.hdfObject.get_all("io", saveThese)
                allIbgrs = self.hdfObject.get_all(("det_0", "Ibgr"), saveThese)
                allSecs = self.hdfObject.get_all("Seconds", saveThese)
                allFs = self.hdfObject.get_all(("det_0", "F"), saveThese)
                allFerrs = self.hdfObject.get_all(("det_0", "Ferr"), saveThese)
                f = open(fname, "w")
                header = "#%5s %5s %5s %7s %7s %7s %7s %7s %7s\n" % ("H", "K", "L", "F", "Ferr", "I", "Io", "Ibgr", "Seconds")
                f.write(header)
                for iterData in saveThese:
                    if not safe_eval_hdf(allBadPs[iterData]):
                        line = "%3.2f %3.2f %6.2f %6.6g %6.6g %6.6g %6.6g %6.6g %6.6g\n" % (
                            allHs[iterData],
                            allKs[iterData],
                            allLs[iterData],
                            allFs[iterData],
                            allFerrs[iterData],
                            allIs[iterData],
                            allIos[iterData],
                            allIbgrs[iterData],
                            allSecs[iterData],
                        )
                        f.write(line)
                print("Saving Intensity data to" + saveDialog.GetPath())
                f.close()
            except Exception:
                oops = wx.MessageDialog(self, "Error saving file\n" + str(Exception))
                oops.ShowModal()
                oops.Destroy()
                raise
        saveDialog.Destroy()

    def onSize(self, event: wx.Event | None) -> None:
        """Handle window resize events and adjust layout."""
        # Phoenix-compatible approach for nested splitters
        total_width = self.GetSize()[0]
        # Set the main split position (tree vs rest)
        self.splitWindow.SetSashPosition(180)
        # Set the right split position (data vs info)
        remaining_width = total_width - 180
        self.rightSplitter.SetSashPosition(remaining_width - 344)

        if event is not None:
            event.Skip()
        rect = self.statusBar.GetFieldRect(self.statusBar.GetFieldsCount() - 2)
        self.integrateCancel.SetPosition((rect.x + 2, rect.y + 2))
        self.integrateCancel.SetSize((rect.width - 4, rect.height - 4))


class customSelector(wx.Dialog):
    """Opens a new tree for custom selection"""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        wx.Dialog.__init__(self, args[0], -1, title="Choose points...", size=(200, 800), style=wx.DEFAULT_DIALOG_STYLE | wx.RESIZE_BORDER)

        self.customTree = myTreeCtrl(self, style=wx.TR_MULTIPLE | wx.TR_HAS_BUTTONS)

        self.okButton = wx.Button(self, wx.ID_OK, label="Ok")
        self.cancelButton = wx.Button(self, wx.ID_CANCEL, label="Cancel")

        self.choosingSizer1 = wx.BoxSizer(wx.VERTICAL)
        self.choosingSizer2 = wx.BoxSizer(wx.HORIZONTAL)

        self.choosingSizer2.Add(self.okButton, proportion=1, flag=wx.EXPAND | wx.ALL, border=4)
        self.choosingSizer2.Add(self.cancelButton, proportion=1, flag=wx.EXPAND | wx.ALL, border=4)

        self.choosingSizer1.Add(self.customTree, proportion=1, flag=wx.EXPAND | wx.ALL, border=4)
        self.choosingSizer1.Add(self.choosingSizer2, flag=wx.EXPAND)

        self.SetSizer(self.choosingSizer1)


def toggle_selector(event: Any) -> None:
    """Holds the RectangleSelector used to pick an ROI"""


labelTips = {
    "roi": """Enter the image roi. The format is [x1,y1,x2,y2] where values
        are in pixels corresponding to two corners of the box. Use the set button to
        select the roi from the image plot zoom value (Fig 1) -> raw scan data plot.""",
    "rotate": """Enter a rotation angle for the image (counter clockwise degrees)""",
    "flag": """Enter flag for image background method:
        0 determine row and column backgrounds after summation
        1 determine 2D background using 'c'olumn direction 
        2 determine 2D background using 'r'ow direction
        3 determine 2D background from the average 'r'ow and 'c'olumn directions""",
    "colNbgr": """Number of background points for linear part of column direction
        (y-direction) bgr fit. If nbgr = 0, no linear fit is included""",
    "colWidth": """Peak width for the column (y) direction bgr fit.
        The background function should fit features that are in general broader
        than the width value. Estimate cwidth using the peak width in the
        column (y) direction. Note: width = 0 corresponds to no polynomial bgr""",
    "colPower": """Power of polynomial used in row (x) direction bgr fit.
        pow = 0 results in linear background only (see nbgr).  Larger values of pow
        result in steeper polynomials.""",
    "rowNbgr": """Number of background points for linear part of row direction
        (x-direction) bgr fit. If nbgr = 0, no linear fit is included""",
    "rowWidth": """Peak width for the row (x) direction bgr fit.
        The background function should fit features that are in general broader
        than the width value. Estimate rwidth using the peak width in the
        row (x) direction. Note: width = 0 corresponds to no polynomial bgr""",
    "rowPower": """Power of polynomial used in row (x) direction bgr fit.
        pow = 0 results in linear background only (see nbgr).  Larger values of pow
        result in steeper polynomials.""",
    "beamSlit": """Enter the incident beam slit settings: beam_slits = {'horz':.6,'vert':.8}
        horz = beam horz width in mm (total width in lab-z / horz scattering plane)
        vert = beam vert hieght in mm (total width in lab-x / vert scattering plane)
        Note these dimensions should be the values at the sample position (ie measured
        with sample position scans)
        If beam slits are 'None' or {} no area correction will be done""",
    "detSlit": """Enter the detector slit settings: det_slits = {'horz':1.,'vert':1.}
        horz = det horz width in mm (total width in lab-z / horiz scattering plane)
        vert = det vert hieght in mm (total width in lab-x / vert scattering plane)
        If detector slits are 'None' or {} only a spill-off correction will be computed""",
    "geom": """Enter goniometer geometry.  Options: psic""",
    "sampleDiameter": """Enter the diameter (in mm) of a round sample mounted on center
        of the goniometer.  If this is <= 0 then use the sample polygon for computing
        the sample correction (or no sample description if a polygon is also not specified). """,
    "samplePolygon": """A list of vectors that describe a general polygon sample shape, e.g.:    
            polygon =  [[1.,1.], [.5,1.5], [-1.,1.],[-1.,-1.],[0.,.5],[1.,-1.]]
        Each entry is an [x,y] or [x,y,z] vector pointing to an apex of the sample
        polygon. These vectors should be given in general lab frame coordinates
        (x is lab frame vertical, y is positive along the direction of the beam,
        the origin is the rotation center). The vectors need to be specified at a 
        given set of angles (see 'sample angles').

        If the sample vectors are given at the flat phi and chi values and with
        the correct sample hieght (sample Z set so the sample surface is on the
        rotation center), then the z values of the sample vectors will be zero.
        If 2D vectors are passed we therefore assume these are [x,y,0].  If this
        is the case then make sure:
            angles = {'phi':flatphi,'chi':flatchi,'eta':0.,'mu':0.}

        The easiest way to determine the sample coordinate vectors is to take a picture
        of the sample with a camera mounted so that is looks directly down the omega
        axis and the gonio angles set at the sample flat phi and chi values with
        eta = mu = 0. Then find the sample rotation center and measure the position
        of each corner (in mm) with up being the +x direction, and downstream
        being the +y direction.  """,
    "sampleAngle": """Sample angles used for the description of the sample polygon.
        Use the format (note if any angles are left out they are assumed zero):
            angles = {'phi':123.5,'chi':0.3,'eta':0.,'mu':0.}
        See 'sample polygon' for more info.""",
    "scale": """Enter scale factor.  The scale factor multiplies by all the intensity
        values. e.g.if Io ~ 1million cps then using 1e6 as the scale makes the normalized
        intensity close to cps.  ie y = scale*y/norm""",
    "badMap": """The file name of the bad pixel map.
        Enter only the file name, not the entire directory path. The file must be in the 
        current directory.""",
}
