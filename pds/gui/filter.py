import datetime
import os
import sys
import time
from typing import Any, Optional

import h5py
import wx
import wx.lib.agw.customtreectrl as treemix
import wx.lib.agw.ultimatelistctrl as ULC
import wx.lib.mixins.listctrl as listmix

from pds.gui.console_capture import ConsoleCapture
from pds.utils import FileLock, FileLockException, list_intersect, list_union, master_to_project
from pds.utils.converters import bytes_to_str

POSSIBLE_ATTRIBUTES = [
    "bad_pixel_map",
    "beam_slits",
    "bgrflag",
    "cnbgr",
    "cpow",
    "ctan",
    "cwidth",
    "det_slits",
    "geom",
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
]


class Filter(wx.Frame):
    """GUI window for filtering HDF project files. Provides interface for selecting/filtering scans and creating project files."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        wx.Frame.__init__(self, args[0], -1, title="HDF Project File Builder", size=(1440, 890))
        self.SetMinSize((1280, 650))

        # Create main splitter for optional output window
        self.mainSplitter = wx.SplitterWindow(self, style=wx.SP_3D | wx.SP_LIVE_UPDATE)
        self.statusSizer = wx.BoxSizer(wx.VERTICAL)
        self.statusSizer.Add(self.mainSplitter, proportion=1, flag=wx.EXPAND)
        self.SetSizer(self.statusSizer)

        self.filterFile = None
        self.filterLock = None
        self.scanItems = []
        self.attrDict = {}
        self.allSpecs = {}
        self.allHK = {}
        self.LMin, self.LMax = (float("inf"), float("-inf"))
        self.allTypes = {}
        self.possibleYears = []
        self.dateMin, self.dateMax = (float("inf"), float("-inf"))
        self.allInfo = {}
        self.activeFilters = []
        self.specResult = None
        self.specCases = None
        self.hkResult = None
        self.hkCases = None
        self.LResult = None
        self.LCases = None
        self.typeResult = None
        self.typeCases = None
        self.dateResult = None
        self.dateCases = None
        self.projectDict = {}
        self.fullWindow = wx.Panel(self.mainSplitter)

        # Set up variables for lazy initialization of output window
        self.outputPanel = None
        self.outputText = None
        self.clearOutputBtn = None
        self.consoleCapture = None
        self.outputWindowVisible = False

        self.originalStdout = sys.stdout
        self.originalStderr = sys.stderr
        self.captureEnabled = False

        # Three main sizers: left (table/buttons), middle (info), right (project)
        self.fullSizer = wx.BoxSizer(wx.HORIZONTAL)

        self.leftSizer = wx.BoxSizer(wx.VERTICAL)
        self.leftPanel = wx.Panel(self.fullWindow)
        self.middleSizer = wx.BoxSizer(wx.VERTICAL)
        self.middlePanel = wx.Panel(self.fullWindow)
        self.rightSizer = wx.BoxSizer(wx.VERTICAL)
        self.rightPanel = wx.Panel(self.fullWindow)

        self.filterButtonSizer = wx.BoxSizer(wx.HORIZONTAL)

        self.specButton = wx.Button(self.leftPanel, label="Filter Specfiles " + "and Scan #s", size=(-1, 32))
        self.hkButton = wx.Button(self.leftPanel, label="Filter HKs", size=(-1, 32))
        self.LButton = wx.Button(self.leftPanel, label="Filter L Range", size=(-1, 32))
        self.typeButton = wx.Button(self.leftPanel, label="Filter Types", size=(-1, 32))
        self.dateButton = wx.Button(self.leftPanel, label="Filter Date Range", size=(-1, 32))

        self.filterButtonSizer.Add(self.specButton, proportion=210, flag=wx.EXPAND | wx.TOP | wx.BOTTOM, border=8)
        self.filterButtonSizer.Add(self.hkButton, proportion=79, flag=wx.EXPAND | wx.TOP | wx.BOTTOM, border=8)
        self.filterButtonSizer.Add(self.LButton, proportion=121, flag=wx.EXPAND | wx.TOP | wx.BOTTOM, border=8)
        self.filterButtonSizer.Add(self.typeButton, proportion=67, flag=wx.EXPAND | wx.TOP | wx.BOTTOM, border=8)
        self.filterButtonSizer.AddStretchSpacer(57)
        self.filterButtonSizer.Add(self.dateButton, proportion=176, flag=wx.EXPAND | wx.TOP | wx.BOTTOM, border=8)

        self.tableSizer = wx.BoxSizer(wx.HORIZONTAL)
        self.dataTable = TableDataCtrl(self.leftPanel, style=wx.LC_REPORT | wx.LC_HRULES | wx.LC_VRULES)
        self.dataTable.InsertColumn(0, heading="Specfile", width=165)
        self.dataTable.InsertColumn(1, heading="#", width=40)
        self.dataTable.InsertColumn(2, heading="H Val", width=40)
        self.dataTable.InsertColumn(3, heading="K Val", width=38)
        self.dataTable.InsertColumn(4, heading="L Start", width=60)
        self.dataTable.InsertColumn(5, heading="L Stop", width=60)
        self.dataTable.InsertColumn(6, heading="Scan Type", width=66)
        self.dataTable.InsertColumn(7, heading="Aborted", width=56)
        self.dataTable.InsertColumn(8, heading="Date")

        # Add the table to the sizer so it scales properly
        self.tableSizer.Add(self.dataTable, proportion=1, flag=wx.EXPAND | wx.BOTTOM, border=2)

        self.managementSizer = wx.BoxSizer(wx.HORIZONTAL)
        self.keepButton = wx.Button(self.leftPanel, label="Keep Selected")
        self.resetButton = wx.Button(self.leftPanel, label="Reset Filters")
        self.rereadButton = wx.Button(self.leftPanel, label="Reread Data")

        self.managementSizer.Add(self.keepButton, proportion=2, flag=wx.EXPAND | wx.BOTTOM, border=2)
        self.managementSizer.AddStretchSpacer(5)
        self.managementSizer.Add(self.resetButton, proportion=2, flag=wx.EXPAND | wx.BOTTOM | wx.RIGHT, border=2)
        self.managementSizer.Add(self.rereadButton, proportion=2, flag=wx.EXPAND | wx.BOTTOM | wx.LEFT, border=2)
        self.managementSizer.Add(wx.StaticLine(self.leftPanel, size=(2, 24)), flag=wx.LEFT, border=16)

        self.leftSizer.Add(self.filterButtonSizer, proportion=0, flag=wx.EXPAND)
        self.leftSizer.Add(self.tableSizer, proportion=1, flag=wx.EXPAND)
        self.leftSizer.Add(self.managementSizer, proportion=0, flag=wx.EXPAND | wx.TOP | wx.BOTTOM, border=4)
        self.leftPanel.SetMinSize((600, 400))
        self.leftPanel.SetSizer(self.leftSizer)

        self.fileSizer = wx.BoxSizer(wx.HORIZONTAL)
        self.fileLabel = wx.StaticText(self.middlePanel, label="File Name: ")
        self.fileButton = wx.Button(self.middlePanel, label="Load Master File...")
        self.fileSizer.Add(self.fileLabel, flag=wx.CENTER)
        self.fileSizer.Add(self.fileButton, proportion=1, flag=wx.EXPAND | wx.RIGHT, border=20)

        self.moreLabel = wx.StaticText(self.middlePanel, label="More Info:\n")
        self.moreBox = wx.TextCtrl(self.middlePanel, style=wx.TE_MULTILINE | wx.TE_READONLY)

        self.rightOne = wx.Button(self.middlePanel, label=">")
        self.leftOne = wx.Button(self.middlePanel, label="<")
        self.rightAll = wx.Button(self.middlePanel, label=">>")
        self.leftAll = wx.Button(self.middlePanel, label="<<")

        self.rightOne.SetToolTip("Move selected to project")
        self.leftOne.SetToolTip("Remove selected from project")
        self.rightAll.SetToolTip("Move all to project")
        self.leftAll.SetToolTip("Remove all from project")

        self.newAttributeSizer = wx.BoxSizer(wx.HORIZONTAL)
        self.attrSelect = wx.Choice(self.middlePanel, size=(110, -1))
        self.attrSelect.SetItems(POSSIBLE_ATTRIBUTES)
        self.attrSelect.SetToolTip("Select Attribute")
        self.attrSpecify = wx.TextCtrl(self.middlePanel)
        self.attrSpecify.SetToolTip("Enter Value")
        self.attrAdd = wx.Button(self.middlePanel, label="+", size=(40, -1))
        self.attrAdd.SetToolTip("Add Attribute")

        self.newAttributeSizer.Add(self.attrSelect)
        self.newAttributeSizer.Add(self.attrSpecify, proportion=1, flag=wx.EXPAND)
        self.newAttributeSizer.Add(self.attrAdd)

        self.attrList = ULC.UltimateListCtrl(self.middlePanel, agwStyle=wx.LC_REPORT | wx.LC_VRULES | wx.LC_HRULES | ULC.ULC_HAS_VARIABLE_ROW_HEIGHT)
        self.attrList.InsertColumn(0, "Attribute", width=108)
        self.attrList.InsertColumn(1, "Value", width=40)
        self.attrList.InsertColumn(2, "", width=36)
        self.attrList.SetColumnWidth(1, ULC.ULC_AUTOSIZE_FILL)

        self.loadSaveAttributeSizer = wx.BoxSizer(wx.HORIZONTAL)
        self.loadAttrButton = wx.Button(self.middlePanel, label="Load Attribute File...")
        self.saveAttrButton = wx.Button(self.middlePanel, label="Save Attribute File...")
        self.loadSaveAttributeSizer.Add(self.loadAttrButton, proportion=1, flag=wx.EXPAND | wx.LEFT, border=32)
        self.loadSaveAttributeSizer.AddSpacer(16)
        self.loadSaveAttributeSizer.Add(self.saveAttrButton, proportion=1, flag=wx.EXPAND | wx.RIGHT, border=32)

        self.middleSizer.Add(self.fileSizer, proportion=0, flag=wx.EXPAND | wx.TOP | wx.LEFT, border=20)
        self.middleSizer.Add(wx.StaticLine(self.middlePanel, size=(332, 2)), flag=wx.LEFT | wx.TOP | wx.RIGHT | wx.EXPAND, border=16)
        self.middleSizer.Add(self.moreLabel, proportion=0, flag=wx.TOP | wx.LEFT, border=20)
        self.middleSizer.Add(self.moreBox, proportion=1, flag=wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, border=24)
        self.middleSizer.Add(self.rightOne, proportion=0, flag=wx.CENTER | wx.TOP | wx.BOTTOM, border=4)
        self.middleSizer.Add(self.leftOne, proportion=0, flag=wx.CENTER | wx.TOP | wx.BOTTOM, border=4)
        self.middleSizer.Add(self.rightAll, proportion=0, flag=wx.CENTER | wx.TOP | wx.BOTTOM, border=4)
        self.middleSizer.Add(self.leftAll, proportion=0, flag=wx.CENTER | wx.TOP | wx.BOTTOM, border=4)
        self.middleSizer.Add(self.newAttributeSizer, proportion=0, flag=wx.EXPAND | wx.LEFT | wx.RIGHT | wx.TOP, border=24)
        self.middleSizer.Add(self.attrList, proportion=1, flag=wx.EXPAND | wx.LEFT | wx.RIGHT, border=24)
        self.middleSizer.AddSpacer(6)
        self.middleSizer.Add(self.loadSaveAttributeSizer, flag=wx.EXPAND)
        self.middleSizer.AddSpacer(6)

        self.middlePanel.SetMinSize((350, 400))
        self.middlePanel.SetSizer(self.middleSizer)

        self.projectNameSizer = wx.BoxSizer(wx.HORIZONTAL)
        self.projectNameText = wx.StaticText(self.rightPanel, label="Project Name: ")
        self.projectNameBox = wx.TextCtrl(self.rightPanel)
        self.projectNameSizer.Add(self.projectNameText, flag=wx.CENTER | wx.LEFT, border=8)
        self.projectNameSizer.Add(self.projectNameBox, proportion=1, flag=wx.EXPAND | wx.RIGHT, border=8)

        self.newProjectTree = myTreeCtrl(self.rightPanel, style=wx.TR_MULTIPLE | wx.TR_DEFAULT_STYLE)

        self.nextStepSizer = wx.BoxSizer(wx.HORIZONTAL)
        self.newIntegrator = wx.Button(self.rightPanel, label="New Project File")
        self.appendIntegrator = wx.Button(self.rightPanel, label="Append Scans To...")
        self.nextStepSizer.Add(wx.StaticLine(self.rightPanel, size=(2, 24)), flag=wx.LEFT | wx.RIGHT, border=1)
        self.nextStepSizer.Add(self.newIntegrator, proportion=1, flag=wx.EXPAND | wx.LEFT | wx.RIGHT, border=16)
        self.nextStepSizer.Add(self.appendIntegrator, proportion=1, flag=wx.EXPAND | wx.LEFT | wx.RIGHT, border=16)

        self.rightSizer.Add(self.projectNameSizer, proportion=0, flag=wx.EXPAND | wx.TOP, border=20)
        self.rightSizer.Add(self.newProjectTree, proportion=1, flag=wx.EXPAND | wx.TOP | wx.BOTTOM, border=4)
        self.rightSizer.Add(self.nextStepSizer, proportion=0, flag=wx.EXPAND | wx.TOP | wx.BOTTOM, border=4)
        self.rightPanel.SetMinSize((360, 400))
        self.rightPanel.SetSizer(self.rightSizer)

        self.fullSizer.Add(self.leftPanel, proportion=6, flag=wx.EXPAND | wx.LEFT, border=8)
        self.fullSizer.Add(self.middlePanel, proportion=3, flag=wx.EXPAND)
        self.fullSizer.Add(self.rightPanel, proportion=4, flag=wx.EXPAND | wx.RIGHT, border=8)

        self.fullWindow.SetSizer(self.fullSizer)

        self.mainSplitter.Initialize(self.fullWindow)

        self.menuBar = wx.MenuBar()
        self.fileMenu = wx.Menu()
        self.loadFile = self.fileMenu.Append(-1, "Load HDF file...")
        self.loadAttr = self.fileMenu.Append(-1, "Load attribute file...")
        self.exitWindow = self.fileMenu.Append(-1, "Exit")
        self.menuBar.Append(self.fileMenu, "File")

        self.viewMenu = wx.Menu()
        self.toggleOutputWindow = self.viewMenu.Append(-1, "Show Output Window", "Toggle stdout/stderr output window", wx.ITEM_CHECK)
        self.menuBar.Append(self.viewMenu, "View")

        self.SetMenuBar(self.menuBar)

        self.Bind(wx.EVT_MENU, self.loadHDF, self.loadFile)
        self.Bind(wx.EVT_MENU, self.loadAttributes, self.loadAttr)
        self.Bind(wx.EVT_MENU, self.onClose, self.exitWindow)

        self.Bind(wx.EVT_MENU, self.toggleOutputWindowVisibility, self.toggleOutputWindow)

        self.specButton.Bind(wx.EVT_BUTTON, self.filterSpec)
        self.hkButton.Bind(wx.EVT_BUTTON, self.filterHK)
        self.LButton.Bind(wx.EVT_BUTTON, self.filterL)
        self.typeButton.Bind(wx.EVT_BUTTON, self.filterType)
        self.dateButton.Bind(wx.EVT_BUTTON, self.filterDate)
        self.keepButton.Bind(wx.EVT_BUTTON, self.keepSelected)
        self.resetButton.Bind(wx.EVT_BUTTON, self.resetFilters)
        self.rereadButton.Bind(wx.EVT_BUTTON, self.readFile)
        self.fileButton.Bind(wx.EVT_BUTTON, self.loadHDF)
        self.rightOne.Bind(wx.EVT_BUTTON, self.moveOneRight)
        self.leftOne.Bind(wx.EVT_BUTTON, self.moveOneLeft)
        self.rightAll.Bind(wx.EVT_BUTTON, self.moveAllRight)
        self.leftAll.Bind(wx.EVT_BUTTON, self.moveAllLeft)
        self.attrAdd.Bind(wx.EVT_BUTTON, self.addAttribute)
        self.loadAttrButton.Bind(wx.EVT_BUTTON, self.loadAttributes)
        self.saveAttrButton.Bind(wx.EVT_BUTTON, self.saveAttributes)
        self.newIntegrator.Bind(wx.EVT_BUTTON, self.newProject)
        self.appendIntegrator.Bind(wx.EVT_BUTTON, self.appendTo)
        self.projectNameBox.Bind(wx.EVT_KILL_FOCUS, self.newName)
        self.dataTable.Bind(wx.EVT_LIST_ITEM_SELECTED, self.tableClick)
        self.Bind(wx.EVT_CLOSE, self.onClose)

        # Focus on static text to avoid button glow on startup
        self.moreLabel.SetFocus()
        self.Show()

    def loadHDF(self, event: wx.CommandEvent) -> None:
        """Open an HDF file and parse its contents into the filter."""
        loadDialog = wx.FileDialog(
            self,
            message="Load file...",
            defaultDir=os.getcwd(),
            defaultFile="",
            wildcard="Master files (*.mh5)|*.mh5|" + "All files (*.*)|*",
            style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST,
        )
        if loadDialog.ShowModal() == wx.ID_OK:
            if not os.path.isfile(loadDialog.GetPath()):
                print("Error: File does not exist")
                return

            # Reset all state variables for new file
            del self.filterFile
            self.filterFile = None
            self.filterFileName = None
            self.filterLock = None
            self.scanItems = []
            self.allSpecs = {}
            self.allHK = {}
            self.LMin, self.LMax = (float("inf"), float("-inf"))
            self.allTypes = {}
            self.possibleYears = []
            self.dateMin, self.dateMax = (float("inf"), float("-inf"))
            self.allInfo = {}
            self.projectNameBox.SetValue("")
            self.resetFilters(None)
            self.updateTable()
            self.projectDict = {}

            print("Loading " + loadDialog.GetPath())
            self.filterFile = loadDialog.GetPath()
            self.filterFileName = loadDialog.GetPath()
            self.filterLock = FileLock(self.filterFileName)
            self.readFile(None)

            # Update UI elements
            self.fileButton.SetLabel(os.path.split(loadDialog.GetPath())[-1])
            if self.projectNameBox.GetValue() == "":
                projName = os.path.basename(loadDialog.GetPath())
                projName = projName.rsplit(".", 1)[0] + ".ph5"
                self.projectNameBox.SetValue(projName)
            self.newProjectTree.DeleteAllItems()
        loadDialog.Destroy()
        self.dataTable.SetFocus()

    def readFile(self, event: Optional[wx.CommandEvent]) -> None:
        """Parse HDF file to extract scan information and populate filter data structures."""
        if self.filterFile is None:
            print("Error: no file selected")
            return

        # Ensure we're working with filename, not closed file handle
        self.filterFile = self.filterFileName

        # Acquire file lock to prevent concurrent access
        try:
            print("Attempting to lock file...")
            while wx.GetApp().HasPendingEvents():
                wx.GetApp().Yield(True)
            self.filterLock.acquire()
            print("Lock acquired")
            while wx.GetApp().HasPendingEvents():
                wx.GetApp().Yield(True)
        except FileLockException as e:
            print("Error: " + str(e))
            return

        try:
            self.filterFile = h5py.File(self.filterFile, "r")
        except IOError:
            print("Error opening file")
            self.filterLock.release()
            print("Lock released")
            return

        # Initialize data structures for scan information
        self.scanItems = []
        self.allSpecs = {}
        self.allHK = {}
        self.LMin, self.LMax = (float("inf"), float("-inf"))
        self.allTypes = {}
        self.possibleYears = []
        self.dateMin, self.dateMax = (float("inf"), float("-inf"))
        self.allInfo = {}

        # Parse HDF structure: specfiles contain scans with attributes
        filterItems = list(self.filterFile.items())
        filterItems.sort()
        for spec, group in filterItems:
            for number, scan in group.items():
                scanAttrs = scan.attrs
                # Format abort status for display
                sAbort = scanAttrs.get("aborted", "?")
                if sAbort == 0:
                    sAbort = ""
                elif sAbort == 1:
                    sAbort = "True"

                # Build info text for detailed scan view
                toShow = (
                    f"Command: {bytes_to_str(scanAttrs.get('cmd', 'N/A'))}\n\n"
                    + f"Attenuators: {bytes_to_str(scanAttrs.get('atten', 'N/A'))}\n\n"
                    + f"Energy: {scanAttrs.get('energy', 'N/A')}\n\n"
                    + f"Data points: {scanAttrs.get('nl_dat', 'N/A')}"
                )

                # Store scan data: [specfile, scan#, H, K, Lstart, Lstop, type, aborted, date, HK_dist, path, info]
                self.scanItems.append(
                    [
                        bytes_to_str(scanAttrs.get("spec_name", "N/A")),
                        str(scanAttrs.get("index", "0")),
                        bytes_to_str(scanAttrs.get("h_val", "--")),
                        bytes_to_str(scanAttrs.get("k_val", "--")),
                        bytes_to_str(scanAttrs.get("real_L_start", "--")),
                        bytes_to_str(scanAttrs.get("real_L_stop", "--")),
                        bytes_to_str(scanAttrs.get("s_type", "N/A")),
                        sAbort,
                        bytes_to_str(scanAttrs.get("date", "N/A")),
                        bytes_to_str(scanAttrs.get("hk_dist", "--")),
                        scan.name,
                        toShow,
                    ]
                )

        # Build filter data structures from scan information
        for scan in self.scanItems:
            # Track scan type frequencies
            if scan[6] not in self.allTypes:
                self.allTypes[scan[6]] = 1
            else:
                self.allTypes[scan[6]] += 1

            # Only show L values for scans that use them
            if scan[6] not in ["rodscan", "Escan", "hklscan"]:
                scan[4] = "--"
                scan[5] = "--"
            else:
                # Build HK pair dictionary: {(H,K): {specfile: [distance, count]}}
                specVal = scan[0]
                hVal = scan[2]
                kVal = scan[3]
                if (hVal, kVal) not in self.allHK:
                    self.allHK[(hVal, kVal)] = {specVal: [scan[9], 1]}
                elif specVal not in self.allHK[(hVal, kVal)]:
                    self.allHK[(hVal, kVal)][specVal] = [scan[9], 1]
                else:
                    self.allHK[(hVal, kVal)][specVal][1] += 1

            # Track specfiles and their scan numbers
            if scan[0] not in self.allSpecs:
                self.allSpecs[scan[0]] = [int(scan[1])]
            else:
                self.allSpecs[scan[0]].append(int(scan[1]))

            # Track L range for filtering
            try:
                self.LMin = min(float(scan[4]), self.LMin)
            except Exception:
                pass
            try:
                self.LMax = max(float(scan[5]), self.LMax)
            except Exception:
                pass

            # Extract years for date filtering
            if scan[8].split()[-1] not in self.possibleYears:
                self.possibleYears.append(scan[8].split()[-1])

            # Track date range for filtering
            try:
                self.dateMin = min(time.mktime(time.strptime(scan[8])), self.dateMin)
            except Exception:
                pass
            try:
                self.dateMax = max(time.mktime(time.strptime(scan[8])), self.dateMax)
            except Exception:
                pass

            # Store detailed info keyed by (specfile, scan#)
            self.allInfo[(scan[0], scan[1])] = scan[11]

        # Sort scans: first by number, then by specfile
        self.scanItems.sort(key=lambda scan: int(scan[1]))
        self.scanItems.sort(key=lambda scan: scan[0])

        self.updateTable()

        # Clean up file handle and release lock
        try:
            self.filterFile.close()
            self.filterLock.release()
            print("Lock released")
        except Exception:
            print("Error closing file")

    def updateTable(self) -> None:
        """Refresh scan table with current filter results."""
        self.dataTable.DeleteAllItems()

        # Collect all active filters
        self.activeFilters = []
        for filter in [self.specCases, self.hkCases, self.LCases, self.typeCases, self.dateCases]:
            if filter is not None:
                self.activeFilters.append(filter)

        if self.activeFilters:
            # Apply intersection of all active filters
            self.activeFilters = list_intersect(*self.activeFilters)
            for entry in self.scanItems:
                if entry[10] in self.activeFilters:
                    self.dataTable.Append(entry[:9])
                    # Gray out scans already in project
                    try:
                        if self.projectDict[entry[0]][entry[1]] is not None:
                            item = self.dataTable.GetItemCount() - 1
                            self.dataTable.SetItemTextColour(item, wx.Colour(128, 128, 128))
                    except Exception:
                        pass
        else:
            # No filters active - show all scans
            for entry in self.scanItems:
                self.dataTable.Append(entry[:9])
                # Gray out scans already in project
                try:
                    if self.projectDict[entry[0]][entry[1]] is not None:
                        item = self.dataTable.GetItemCount() - 1
                        self.dataTable.SetItemTextColour(item, wx.Colour(128, 128, 128))
                except Exception:
                    pass

    def tableClick(self, event: wx.ListEvent) -> None:
        """Update info panel when user clicks on a scan in the table."""
        itemId = event.GetIndex()

        # Extract specfile name and scan number from selected row
        specName = self.dataTable.GetItem(itemId, 0).Text
        scanNumber = self.dataTable.GetItem(itemId, 1).Text

        # Display detailed scan info in the text box
        toShow = self.allInfo.get((specName, scanNumber), "")
        self.moreBox.SetValue(toShow)

    def addAttribute(self, event: wx.CommandEvent) -> None:
        """Add selected attribute-value pair to the project attributes list."""
        attrText = self.attrSelect.GetStringSelection()
        # Skip if no attribute selected or already exists
        if attrText == "" or attrText in self.attrDict:
            return

        attrValue = str(self.attrSpecify.GetValue())
        if attrValue == "":
            return

        # Store attribute in dictionary and display in list
        self.attrDict[attrText] = attrValue
        self.attrList.Append([attrText, attrValue, ""])
        self.attrList.SetItemData(self.attrList.GetItemCount() - 1, attrText)

        # Add delete button for this attribute
        thisButton = wx.Button(self.attrList, label="X", size=(32, 15), name=attrText)
        thisButton.Bind(wx.EVT_BUTTON, self.deleteMe)
        self.attrList.SetItemWindow(self.attrList.GetItemCount() - 1, col=2, wnd=thisButton)

        # Select the newly added item
        if self.attrList.GetItemCount() > 0:
            self.attrList.Select(self.attrList.GetItemCount() - 1)

    def deleteMe(self, event: wx.CommandEvent) -> None:
        """Remove attribute from both dictionary and display list."""
        # Get attribute name from the delete button that was clicked
        deleteThis = event.GetEventObject().GetName()
        del self.attrDict[deleteThis]

        # Find and remove the corresponding list item
        deleteThis = self.attrList.FindItemData(-1, deleteThis)
        self.attrList.DeleteItem(deleteThis)

    def loadAttributes(self, event: wx.CommandEvent) -> None:
        """Load attribute-value pairs from a tab-delimited text file."""
        loadDialog = wx.FileDialog(
            self, message="Load file...", defaultDir=os.getcwd(), defaultFile="", wildcard="txt files (*.txt)|*.txt|" + "All files (*.*)|*", style=wx.FD_OPEN
        )
        if loadDialog.ShowModal() == wx.ID_OK:
            print("Loading attribute file " + loadDialog.GetPath())
            try:
                attributeFile = open(loadDialog.GetPath())
            except:
                print("Error opening attribute file")
                loadDialog.Destroy()
                raise

            try:
                for line in attributeFile:
                    line = line.strip().split("\t")
                    # Skip invalid lines: must be 2 columns, valid attribute, not duplicate, not empty
                    if len(line) != 2 or line[0] not in POSSIBLE_ATTRIBUTES or line[0] in list(self.attrDict.keys()) or line[1] == "":
                        continue

                    # Add attribute to dictionary and display list
                    self.attrDict[line[0]] = line[1]
                    self.attrList.Append([line[0], line[1], ""])
                    self.attrList.SetItemData(self.attrList.GetItemCount() - 1, line[0])

                    # Add delete button for this attribute
                    thisButton = wx.Button(self.attrList, label="X", size=(32, 15), name=line[0])
                    thisButton.Bind(wx.EVT_BUTTON, self.deleteMe)
                    self.attrList.SetItemWindow(self.attrList.GetItemCount() - 1, col=2, wnd=thisButton)
                attributeFile.close()
            except:
                print("Error reading attribute file")
                loadDialog.Destroy()
                raise
        loadDialog.Destroy()

    def saveAttributes(self, event: wx.CommandEvent) -> None:
        """Save current attribute list to a tab-delimited text file."""
        saveDialog = wx.FileDialog(
            self, message="Save file...", defaultDir=os.getcwd(), defaultFile="", wildcard="txt files (*.txt)|*.txt|" + "All files (*.*)|*", style=wx.FD_SAVE
        )
        if saveDialog.ShowModal() == wx.ID_OK:
            print("Saving attribute file " + saveDialog.GetPath())
            try:
                attributeFile = open(saveDialog.GetPath(), "w")
            except:
                print("Error opening attribute file")
                saveDialog.Destroy()
                raise

            try:
                # Write each attribute-value pair as tab-separated line
                for key, value in self.attrDict.items():
                    attributeFile.write(key + "\t" + value + "\n")
            except:
                print("Error writing to file")
                attributeFile.close()
                saveDialog.Destroy()
                raise
            attributeFile.close()
        saveDialog.Destroy()

    def dictToTree(self, thisDict: dict[str, Any], thisTree: wx.TreeCtrl, thisRoot: wx.TreeItemId) -> None:
        """Recursively build tree structure from nested dictionary."""
        for key, value in thisDict.items():
            if type(value) is dict:
                # Create parent node for nested dictionary
                parentItem = thisTree.AppendItem(thisRoot, key)
                self.dictToTree(value, thisTree, parentItem)
            else:
                # Create leaf node for simple key-value pair
                thisTree.AppendItem(thisRoot, str(key) + ": " + str(value))
        thisTree.SortChildren(thisRoot)

    def newName(self, event: wx.FocusEvent) -> None:
        """Ensure project name has .ph5 extension and update tree root."""
        # Auto-append .ph5 extension if missing
        if not self.projectNameBox.GetValue().endswith(".ph5"):
            self.projectNameBox.SetValue(self.projectNameBox.GetValue() + ".ph5")

        # Update project tree root with new name
        try:
            self.newProjectTree.SetItemText(self.newProjectTree.GetRootItem(), self.projectNameBox.GetValue())
        except Exception:
            pass

    def moveOneRight(self, event: wx.CommandEvent) -> None:
        """Move selected table scans to project tree with current attributes."""
        item = self.dataTable.GetFirstSelected()
        if item == -1:
            return

        # Process each selected scan
        while item != -1:
            specName = self.dataTable.GetItem(item, 0).Text
            scanNumber = self.dataTable.GetItem(item, 1).Text

            # Add to project dictionary with current attribute set
            if specName in self.projectDict:
                if scanNumber not in self.projectDict[specName]:
                    self.projectDict[specName][scanNumber] = self.attrDict.copy()
                else:
                    print("Specfile " + specName + ", scan " + scanNumber + " is already in the tree.")
            else:
                self.projectDict[specName] = {}
                self.projectDict[specName][scanNumber] = self.attrDict.copy()

            # Gray out added scans in table
            self.dataTable.SetItemTextColour(item, wx.Colour(128, 128, 128))
            item = self.dataTable.GetNextSelected(item)

        # Rebuild and display the project tree
        self.newProjectTree.DeleteAllItems()
        projectRoot = self.newProjectTree.AddRoot(self.projectNameBox.GetValue())
        self.dictToTree(self.projectDict, self.newProjectTree, projectRoot)
        self.newProjectTree.Expand(projectRoot)

    def moveAllRight(self, event: wx.CommandEvent) -> None:
        """Move all visible table scans to project tree with current attributes."""
        item = self.dataTable.GetNextItem(-1)
        if item == -1:
            return

        # Process all scans in table
        while item != -1:
            specName = self.dataTable.GetItem(item, 0).Text
            scanNumber = self.dataTable.GetItem(item, 1).Text

            # Add to project dictionary with current attribute set
            if specName in self.projectDict:
                if scanNumber not in self.projectDict[specName]:
                    self.projectDict[specName][scanNumber] = self.attrDict.copy()
                else:
                    print("Specfile " + specName + ", scan " + scanNumber + " is already in the tree.")
            else:
                self.projectDict[specName] = {}
                self.projectDict[specName][scanNumber] = self.attrDict.copy()

            # Gray out added scans in table
            self.dataTable.SetItemTextColour(item, wx.Colour(128, 128, 128))
            item = self.dataTable.GetNextItem(item)

        # Rebuild and display the project tree
        self.newProjectTree.DeleteAllItems()
        projectRoot = self.newProjectTree.AddRoot(self.projectNameBox.GetValue())
        self.dictToTree(self.projectDict, self.newProjectTree, projectRoot)
        self.newProjectTree.Expand(projectRoot)

    def moveOneLeft(self, event: wx.CommandEvent) -> None:
        """Remove selected items from project tree and restore to normal table color."""
        allSelected = self.newProjectTree.GetSelections()
        allSelected.sort(key=lambda selection: self.newProjectTree.getLevel(selection))

        for selection in allSelected:
            try:
                selectionLevel = self.newProjectTree.getLevel(selection)
            except Exception:
                selectionLevel = -1

            if selectionLevel == 0:
                # Root selected - remove everything
                self.moveAllLeft(event)
                return
            elif selectionLevel == 1:
                # Specfile level - remove entire specfile
                specName = self.newProjectTree.GetItemText(selection)

                # Restore normal color to all scans from this specfile
                item = self.dataTable.GetNextItem(-1)
                while item != -1:
                    if self.dataTable.GetItem(item, 0).Text == specName:
                        self.dataTable.SetItemTextColour(item, wx.BLACK)
                    item = self.dataTable.GetNextItem(item)

                # Remove from project dictionary and tree
                self.projectDict.pop(specName)
                self.newProjectTree.Delete(selection)
            elif selectionLevel == 2:
                # Individual scan level - remove specific scan
                scanParent = self.newProjectTree.GetItemParent(selection)
                scanNumber = self.newProjectTree.GetItemText(selection)
                specName = self.newProjectTree.GetItemText(scanParent)

                # Restore normal color to this specific scan
                item = self.dataTable.GetNextItem(-1)
                while item != -1:
                    if self.dataTable.GetItem(item, 0).Text == specName and self.dataTable.GetItem(item, 1).Text == scanNumber:
                        self.dataTable.SetItemTextColour(item, wx.BLACK)
                        break
                    item = self.dataTable.GetNextItem(item)

                # Remove scan from project dictionary and tree
                self.projectDict[specName].pop(scanNumber)
                self.newProjectTree.Delete(selection)

        # Clean up empty specfile entries
        popUs = []
        for key in self.projectDict:
            if self.projectDict[key] == {}:
                popUs.append(key)
        for key in popUs:
            self.projectDict.pop(key)

        # Clean up empty tree nodes
        projectRoot = self.newProjectTree.GetRootItem()
        if not projectRoot.IsOk():
            return
        if self.newProjectTree.GetChildrenCount(projectRoot) == 0:
            self.newProjectTree.Delete(projectRoot)
        else:
            item, cookie = self.newProjectTree.GetFirstChild(projectRoot)
            while item.IsOk():
                if self.newProjectTree.GetChildrenCount(item) == 0:
                    self.newProjectTree.Delete(item)
                item, cookie = self.newProjectTree.GetNextChild(item, cookie)

    def moveAllLeft(self, event: wx.CommandEvent) -> None:
        """Clear entire project tree and restore all table items to normal color."""
        self.newProjectTree.DeleteAllItems()
        self.projectDict = {}

        # Restore normal color to all table items
        item = self.dataTable.GetNextItem(-1)
        while item != -1:
            self.dataTable.SetItemTextColour(item, wx.BLACK)
            item = self.dataTable.GetNextItem(item)

    def newProject(self, event: wx.CommandEvent) -> None:
        """Create a new HDF project file with selected scans and current attributes."""
        if self.projectDict == {}:
            print("No scans selected")
            return

        # Show file dialog for new project file location
        self.fileDirectory, holding = os.path.split(self.filterFileName)
        file_types = "Project files (*.ph5)|*.ph5|All files (*.*)|*"
        save_dialog = wx.FileDialog(
            self,
            message="Create file...",
            defaultDir=self.fileDirectory,
            defaultFile=self.projectNameBox.GetValue(),
            wildcard=file_types,
            style=wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT,
        )

        if save_dialog.ShowModal() == wx.ID_OK:
            # Lock both master and project files during creation
            mlockFile = FileLock(self.filterFileName)
            plockFile = FileLock(save_dialog.GetPath())
            try:
                print("Attempting to lock files...")
                while wx.GetApp().HasPendingEvents():
                    wx.GetApp().Yield(True)
                mlockFile.acquire()
                plockFile.acquire()
                print("Locks acquired")
                wx.GetApp().Yield(True)
            except FileLockException as e:
                print("Error: " + str(e))
                return

            # Create project file from master file and selected scans
            print("Start: ", time.ctime(time.time()))
            out_file = save_dialog.GetPath()
            try:
                master_to_project(self.filterFileName, self.projectDict, out_file, append=False, gui=True)
            except Exception as e:
                print(f"Error generating project file: {e}")
                mlockFile.release()
                plockFile.release()
                print("Locks released")
            print("Finish: ", time.ctime(time.time()))
            mlockFile.release()
            plockFile.release()
            print("Locks released")
        save_dialog.Destroy()

    def appendTo(self, event: wx.CommandEvent) -> None:
        """Append selected scans to an existing HDF project file."""
        if self.projectDict == {}:
            print("No scans selected")
            return

        # Show file dialog for existing project file to append to
        self.fileDirectory, holding = os.path.split(self.filterFileName)
        file_types = "Project files (*.ph5)|*.ph5|All files (*.*)|*"
        save_dialog = wx.FileDialog(
            self, message="Append to...", defaultDir=self.fileDirectory, defaultFile=self.projectNameBox.GetValue(), wildcard=file_types, style=wx.FD_SAVE
        )

        if save_dialog.ShowModal() == wx.ID_OK:
            # Lock both master and project files during append operation
            mlockFile = FileLock(self.filterFileName)
            plockFile = FileLock(save_dialog.GetPath())
            try:
                print("Attempting to lock files...")
                while wx.GetApp().HasPendingEvents():
                    wx.GetApp().Yield(True)
                mlockFile.acquire()
                plockFile.acquire()
                print("Locks acquired")
            except FileLockException as e:
                print("Error: " + str(e))
                return

            # Append scans to existing project file
            print("Start: ", time.ctime(time.time()))
            out_file = save_dialog.GetPath()
            print(f"Appending {sum(len(scans) for scans in self.projectDict.values())} scans to {out_file}")

            try:
                master_to_project(self.filterFileName, self.projectDict, out_file, append=True, gui=True)
                print("\n✅ Successfully appended scans to project file")
            except Exception as e:
                import traceback

                print(f"\n❌ Error appending to project file: {e}")
                print("Full error details:")
                traceback.print_exc()
                # Show error dialog to user
                error_msg = f"Failed to append scans to project file:\n\n{str(e)}\n\nSee console output for full details."
                wx.MessageBox(error_msg, "Append Error", wx.OK | wx.ICON_ERROR)
            finally:
                mlockFile.release()
                plockFile.release()
                print("Locks released")

            print("Finish: ", time.ctime(time.time()))
        save_dialog.Destroy()

    def keepSelected(self, event: wx.CommandEvent) -> None:
        """Apply filter to show only selected scans by creating a fake spec filter."""
        item = self.dataTable.GetFirstSelected()
        if item == -1:
            return

        # Build specfile filter from selected table items
        self.specResult = {}
        self.specCases = None
        while item != -1:
            specName = self.dataTable.GetItem(item, 0).Text
            scanNumber = self.dataTable.GetItem(item, 1).Text
            if specName in self.specResult:
                self.specResult[specName].append(int(scanNumber))
            else:
                self.specResult[specName] = [int(scanNumber)]
            item = self.dataTable.GetNextSelected(item)

        # Add empty entries for unselected specfiles
        for key in self.allSpecs:
            if key not in self.specResult:
                self.specResult[key] = []

        # Create filter cases from the selection
        if self.specResult == self.allSpecs:
            self.specResult = None
        else:
            for spec in self.specResult.keys():
                bothCases = []
                if self.specResult[spec] != []:
                    thisCase = [scan[10] for scan in self.scanItems if scan[0].startswith(spec)]
                    thatCase = [scan[10] for scan in self.scanItems if int(scan[1]) in self.specResult[spec]]
                    bothCases = list_intersect(thisCase, thatCase)
                    self.specCases = list_union(bothCases, self.specCases)
        self.updateTable()

    def filterSpec(self, event: wx.CommandEvent) -> None:
        """Open specfile filter dialog."""
        filterWindow = SpecWindow(self, self.allSpecs, self.specResult)
        filterWindow.ShowModal()
        filterWindow.Destroy()

    def filterHK(self, event: wx.CommandEvent) -> None:
        """Open HK pair filter dialog."""
        filterWindow = HKWindow(self, self.allHK, self.hkResult)
        filterWindow.ShowModal()
        filterWindow.Destroy()

    def filterL(self, event: wx.CommandEvent) -> None:
        """Open L range filter dialog."""
        filterWindow = LWindow(self, (self.LMin, self.LMax), self.LResult)
        filterWindow.ShowModal()
        filterWindow.Destroy()

    def filterType(self, event: wx.CommandEvent) -> None:
        """Open scan type filter dialog."""
        filterWindow = TypeWindow(self, self.allTypes, self.typeResult)
        filterWindow.ShowModal()
        filterWindow.Destroy()

    def filterDate(self, event: wx.CommandEvent) -> None:
        """Open date range filter dialog."""
        filterWindow = DateWindow(self, self.possibleYears, (self.dateMin, self.dateMax), self.dateResult)
        filterWindow.ShowModal()
        filterWindow.Destroy()

    def resetFilters(self, event: Optional[wx.CommandEvent]) -> None:
        """Clear all active filters and show all scans."""
        self.activeFilters = []
        self.specResult = None
        self.specCases = None
        self.hkResult = None
        self.hkCases = None
        self.LResult = None
        self.LCases = None
        self.typeResult = None
        self.typeCases = None
        self.dateResult = None
        self.dateCases = None
        self.updateTable()

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

    def toggleOutputWindowVisibility(self, event: wx.CommandEvent) -> None:
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
            self.mainSplitter.SplitHorizontally(self.fullWindow, self.outputPanel, -150)
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

    def onClearOutput(self, event: wx.CommandEvent) -> None:
        """Clear the output window."""
        self.outputText.Clear()

    def onClose(self, event: wx.CloseEvent) -> None:
        """Clean up file handles and locks before closing the window."""
        self.disableCapture()
        try:
            self.filterFile.close()
            self.filterLock.release()
            print("Lock released")
        except Exception:
            pass
        self.Destroy()


class SpecWindow(wx.Dialog):
    """GUI dialog for filtering scans by specfile. Provides hierarchical tree view of specfiles and their scan numbers."""

    def __init__(self, parent: Optional[wx.Window] = None, allSpec: dict[str, list[int]] = {}, currentSpec: Optional[dict[str, list[int]]] = None) -> None:
        """Initialize specfile filter dialog with hierarchical checkable tree."""
        wx.Dialog.__init__(self, parent, -1, title="Specfiles", size=(240, 400))

        # Create checkable tree control for specfile selection
        self.list = treemix.CustomTreeCtrl(
            self, agwStyle=treemix.TR_HAS_BUTTONS | treemix.TR_MULTIPLE | treemix.TR_EXTENDED | treemix.TR_AUTO_CHECK_CHILD | treemix.TR_AUTO_CHECK_PARENT
        )

        # Store spec data and set current selection state
        self.allSpec = allSpec
        if currentSpec is not None:
            self.currentSpec = currentSpec
        else:
            self.currentSpec = allSpec

        # Build tree: root -> specfiles -> individual scans
        self.allRoot = self.list.AddRoot(self.GetParent().fileButton.GetLabel(), ct_type=1)
        specKeys = sorted(self.allSpec.keys())
        for spec in specKeys:
            specRoot = self.list.AppendItem(self.allRoot, str(spec), ct_type=1)
            allScans = self.allSpec[spec]
            allScans.sort()
            for scan in allScans:
                scanRoot = self.list.AppendItem(specRoot, str(scan), ct_type=1)
                # Check scans that are currently selected
                if scan in self.currentSpec[spec]:
                    self.list.CheckItem(scanRoot)
        self.list.Expand(self.allRoot)

        # Create dialog buttons
        self.apply = wx.Button(self, label="Apply")
        self.cancel = wx.Button(self, label="Cancel")

        # Bind button events
        self.apply.Bind(wx.EVT_BUTTON, self.onApply)
        self.cancel.Bind(wx.EVT_BUTTON, self.onCancel)

        # Layout components
        self.windowSizer = wx.BoxSizer(wx.VERTICAL)
        self.buttonSizer = wx.BoxSizer(wx.HORIZONTAL)

        self.windowSizer.Add(self.list, proportion=1, flag=wx.EXPAND | wx.ALL, border=4)

        self.buttonSizer.Add(self.apply, proportion=1, flag=wx.EXPAND | wx.ALL, border=4)
        self.buttonSizer.Add(self.cancel, proportion=1, flag=wx.EXPAND | wx.ALL, border=4)
        self.windowSizer.Add(self.buttonSizer, flag=wx.EXPAND | wx.ALL, border=4)

        self.SetSizer(self.windowSizer)
        self.CenterOnScreen()
        self.Bind(wx.EVT_CLOSE, self.onCancel)

    def onApply(self, event: wx.CommandEvent) -> None:
        """Apply specfile filter selection and update main table."""
        selectedSpec = {}

        # Extract checked scans from tree control
        specKids = self.allRoot.GetChildren()
        for spec in specKids:
            scanKids = spec.GetChildren()
            specText = spec.GetText()
            selectedSpec[specText] = []
            for scan in scanKids:
                if scan.IsChecked():
                    selectedSpec[specText].append(int(scan.GetText()))

        # Clear previous filter cases
        self.GetParent().specCases = None

        # Update parent filter with new selection
        if self.allSpec == selectedSpec:
            # All scans selected - no filtering needed
            self.GetParent().specResult = None
        else:
            # Apply specfile filter based on selection
            self.GetParent().specResult = selectedSpec
            for spec in selectedSpec.keys():
                bothCases = []
                if selectedSpec[spec] != []:
                    # Find scans matching both specfile name and scan numbers
                    thisCase = [scan[10] for scan in self.GetParent().scanItems if str(scan[0]).startswith(spec)]
                    thatCase = [scan[10] for scan in self.GetParent().scanItems if int(scan[1]) in selectedSpec[spec]]
                    bothCases = list_intersect(thisCase, thatCase)
                    self.GetParent().specCases = list_union(bothCases, self.GetParent().specCases)

        self.GetParent().updateTable()
        self.EndModal(wx.ID_OK)

    def onCancel(self, event: wx.CommandEvent) -> None:
        """Close dialog without applying changes."""
        self.EndModal(wx.ID_CANCEL)


class HKWindow(wx.Dialog):
    """GUI dialog for filtering scans by HK crystallographic pairs. Shows HK values with associated specfiles and their scan distances."""

    def __init__(
        self,
        parent: Optional[wx.Window] = None,
        allHK: dict[tuple[str, str], dict[str, list[Any]]] = {},
        currentHK: Optional[dict[tuple[str, str], dict[str, list[Any]]]] = None,
    ) -> None:
        """Initialize HK filter dialog with hierarchical tree of HK values and specfiles."""
        wx.Dialog.__init__(self, parent, -1, title="HK Pairs", size=(400, 400))

        # Create checkable tree control for HK pair selection
        self.list = treemix.CustomTreeCtrl(
            self, agwStyle=treemix.TR_HAS_BUTTONS | treemix.TR_MULTIPLE | treemix.TR_EXTENDED | treemix.TR_AUTO_CHECK_CHILD | treemix.TR_AUTO_CHECK_PARENT
        )

        # Store HK data and set current selection state
        self.allHK = allHK
        if currentHK is not None:
            self.currentHK = currentHK
        else:
            self.currentHK = allHK

        # Build tree: root -> HK pairs -> specfiles with distances
        self.allRoot = self.list.AddRoot("HK Pairs", ct_type=1)
        self.allHKKeys = list(self.allHK.keys())
        # Sort by K value, then by H value for organized display
        self.allHKKeys = sorted(self.allHKKeys, key=lambda key: float(key[1]))
        self.allHKKeys = sorted(self.allHKKeys, key=lambda key: float(key[0]))

        for hk in self.allHKKeys:
            # Create HK pair node - store original key in item data
            hkRoot = self.list.AppendItem(self.allRoot, str(tuple(map(float, hk))), ct_type=1)
            hkRoot.SetData(hk)  # Store original key for later retrieval
            allSpecs = sorted(self.allHK[hk].keys())
            for spec in allSpecs:
                # Create specfile node showing scan count and distance
                specRoot = self.list.AppendItem(
                    hkRoot, spec + ": " + str(self.allHK[hk][spec][1]) + " at a distance of " + str(self.allHK[hk][spec][0]), ct_type=1
                )
                # Check specfiles that are currently selected
                if spec in self.currentHK[hk]:
                    self.list.CheckItem(specRoot)
        self.list.Expand(self.allRoot)

        # Create dialog buttons
        self.apply = wx.Button(self, label="Apply")
        self.cancel = wx.Button(self, label="Cancel")

        # Bind button events
        self.apply.Bind(wx.EVT_BUTTON, self.onApply)
        self.cancel.Bind(wx.EVT_BUTTON, self.onCancel)

        # Layout components
        self.windowSizer = wx.BoxSizer(wx.VERTICAL)
        self.buttonSizer = wx.BoxSizer(wx.HORIZONTAL)

        self.windowSizer.Add(self.list, proportion=1, flag=wx.EXPAND | wx.ALL, border=4)

        self.buttonSizer.Add(self.apply, proportion=1, flag=wx.EXPAND | wx.ALL, border=4)
        self.buttonSizer.Add(self.cancel, proportion=1, flag=wx.EXPAND | wx.ALL, border=4)
        self.windowSizer.Add(self.buttonSizer, flag=wx.EXPAND | wx.ALL, border=4)

        self.SetSizer(self.windowSizer)
        self.CenterOnScreen()
        self.Bind(wx.EVT_CLOSE, self.onCancel)

    def onApply(self, event: wx.CommandEvent) -> None:
        """Apply HK pair filter selection and update main table."""
        selectedHK = {}

        # Extract checked specfiles from tree structure
        hkKids = self.allRoot.GetChildren()
        for hk in hkKids:
            specKids = hk.GetChildren()
            # Get original HK key from stored data
            hkText = hk.GetData()
            selectedHK[hkText] = {}
            for spec in specKids:
                if spec.IsChecked():
                    # Extract specfile name from display text (before the colon)
                    specName = spec.GetText().split(":")[0]
                    # Copy distance and count info from original data using original key
                    selectedHK[hkText][specName] = self.allHK[hkText][specName]

        # Clear previous HK filter cases
        self.GetParent().hkCases = None

        if self.allHK == selectedHK:
            # All HK pairs selected - no filtering needed
            self.GetParent().hkResult = None
        else:
            # Apply HK filter based on selection
            self.GetParent().hkResult = selectedHK
            for hk in selectedHK.keys():
                bothCases = []
                if selectedHK[hk] != {}:
                    # Find scans matching H value
                    thisCase = [scan[10] for scan in self.GetParent().scanItems if scan[2] == str(hk[0])]
                    # Find scans matching K value
                    thatCase = [scan[10] for scan in self.GetParent().scanItems if scan[3] == str(hk[1])]
                    # Find scans from selected specfiles
                    otherCase = [scan[10] for scan in self.GetParent().scanItems if scan[0] in str(selectedHK[hk].keys())]
                    # Intersection gives scans matching all three criteria
                    bothCases = list_intersect(thisCase, thatCase, otherCase)
                    self.GetParent().hkCases = list_union(bothCases, self.GetParent().hkCases)

        self.GetParent().updateTable()
        self.EndModal(wx.ID_OK)

    def onCancel(self, event: wx.CommandEvent) -> None:
        """Close dialog without applying HK filter changes."""
        self.EndModal(wx.ID_CANCEL)


class LWindow(wx.Dialog):
    """GUI dialog for filtering scans by L crystallographic range. Provides text input fields for minimum and maximum L values."""

    def __init__(self, parent: Optional[wx.Window] = None, allL: tuple[float, float] = (-100, 100), currentL: Optional[tuple[float, float]] = None) -> None:
        """Initialize L range filter dialog with input fields for From and To values."""
        wx.Dialog.__init__(self, parent, -1, title="L Range", size=(232, 120))

        # Store L range bounds and current filter values
        self.LMin, self.LMax = allL
        if currentL is not None:
            self.currentLMin, self.currentLMax = currentL
        else:
            self.currentLMin, self.currentLMax = allL

        # Create 'From' value input components
        self.fromText = wx.StaticText(self, label="From:")
        self.fromValue = wx.TextCtrl(self, size=(60, -1))
        self.fromValue.SetValue(str(self.currentLMin))

        # Create 'To' value input components
        self.toText = wx.StaticText(self, label="To:")
        self.toValue = wx.TextCtrl(self, size=(60, -1))
        self.toValue.SetValue(str(self.currentLMax))

        # Create dialog buttons
        self.apply = wx.Button(self, label="Apply")
        self.cancel = wx.Button(self, label="Cancel")

        # Bind button events
        self.apply.Bind(wx.EVT_BUTTON, self.onApply)
        self.cancel.Bind(wx.EVT_BUTTON, self.onCancel)

        # Layout components
        self.windowSizer = wx.BoxSizer(wx.VERTICAL)
        self.fromToSizer = wx.BoxSizer(wx.HORIZONTAL)
        self.buttonSizer = wx.BoxSizer(wx.HORIZONTAL)

        # Arrange From and To inputs horizontally
        self.fromToSizer.Add(self.fromText, flag=wx.EXPAND | wx.TOP | wx.BOTTOM | wx.LEFT, border=8)
        self.fromToSizer.Add(self.fromValue, flag=wx.EXPAND | wx.ALL, border=4)
        self.fromToSizer.Add(self.toText, flag=wx.EXPAND | wx.TOP | wx.BOTTOM | wx.LEFT, border=8)
        self.fromToSizer.Add(self.toValue, flag=wx.EXPAND | wx.ALL, border=4)
        self.windowSizer.Add(self.fromToSizer, flag=wx.EXPAND | wx.ALL, border=4)

        # Arrange buttons horizontally
        self.buttonSizer.Add(self.apply, proportion=1, flag=wx.EXPAND | wx.RIGHT | wx.BOTTOM | wx.LEFT, border=4)
        self.buttonSizer.Add(self.cancel, proportion=1, flag=wx.EXPAND | wx.RIGHT | wx.BOTTOM | wx.LEFT, border=4)
        self.windowSizer.Add(self.buttonSizer, flag=wx.EXPAND | wx.ALL, border=4)

        # Finalize dialog setup
        self.SetSizer(self.windowSizer)
        self.CenterOnScreen()
        self.Bind(wx.EVT_CLOSE, self.onCancel)

    def onApply(self, event: wx.CommandEvent) -> None:
        """Apply L range filter and update main table."""

        # Parse and validate From L value with fallback
        try:
            fromL = float(self.fromValue.GetValue())
        except Exception:
            print("Error: Invalid minimum L; setting to -100")
            fromL = -100.0

        # Parse and validate To L value with fallback
        try:
            toL = float(self.toValue.GetValue())
        except Exception:
            print("Error: Invalid maximum L; setting to 100")
            toL = 100.0

        # Clear previous L filter cases
        self.GetParent().LCases = None

        if fromL <= self.LMin and toL >= self.LMax:
            # Range covers entire dataset - no filtering needed
            self.GetParent().LResult = None
        else:
            # Apply L range filter
            self.GetParent().LResult = (fromL, toL)

            # Find scans with L_start >= fromL (limited to scan types that use L values)
            thisCase = [
                scan[10] for scan in self.GetParent().scanItems if scan[6] in ["rodscan", "Escan", "hklscan"] and scan[4] != "--" and float(scan[4]) >= fromL
            ]

            # Find scans with L_stop <= toL (limited to scan types that use L values)
            thatCase = [
                scan[10] for scan in self.GetParent().scanItems if scan[6] in ["rodscan", "Escan", "hklscan"] and scan[4] != "--" and float(scan[5]) <= toL
            ]

            # Intersection gives scans within the L range
            bothCases = list_intersect(thisCase, thatCase)
            self.GetParent().LCases = bothCases

        self.GetParent().updateTable()
        self.EndModal(wx.ID_OK)

    def onCancel(self, event: wx.CommandEvent) -> None:
        """Close dialog without applying L range filter changes."""
        self.EndModal(wx.ID_CANCEL)


class TypeWindow(wx.Dialog):
    """GUI dialog for filtering scans by type (e.g., rodscan, Escan, hklscan). Shows checkable list of all available scan types with occurrence counts."""

    def __init__(self, parent: Optional[wx.Window] = None, allTypes: dict[str, int] = {}, currentTypes: Optional[dict[str, int]] = None) -> None:
        """Initialize scan type filter dialog with checkable list of scan types."""
        wx.Dialog.__init__(self, parent, -1, title="Scan Types", size=(200, 400))

        # Create checkable list control with columns for checkbox, scan type name, and count
        self.list = NumberListCtrl(self, style=wx.LC_REPORT | wx.LC_NO_HEADER | wx.LC_HRULES | wx.LC_VRULES)
        self.list.InsertColumn(0, "", width=24)  # Checkbox column
        self.list.InsertColumn(1, "Scan Type", width=66)  # Scan type name
        self.list.InsertColumn(2, "Num. Scans")  # Count of scans per type

        # Store scan type data and set current selection state
        self.allTypes = allTypes
        if currentTypes is not None:
            self.currentTypes = currentTypes
        else:
            self.currentTypes = allTypes

        # Populate list with scan types and their counts
        self.allTypeKeys = sorted(self.allTypes.keys())
        for key in self.allTypeKeys:
            newType = self.list.Append(["", key, self.allTypes[key]])
            # Check scan types that are currently selected
            if key in self.currentTypes:
                self.list.CheckItem(newType)

        # Create UI components
        self.selector = wx.CheckBox(self, style=wx.CHK_3STATE)
        self.apply = wx.Button(self, label="Apply")
        self.cancel = wx.Button(self, label="Cancel")

        # Set initial state of "check all" checkbox based on current selections
        self.setCheck(None)

        # Bind events
        self.apply.Bind(wx.EVT_BUTTON, self.onApply)
        self.cancel.Bind(wx.EVT_BUTTON, self.onCancel)
        self.selector.Bind(wx.EVT_CHECKBOX, self.onCheckAll)

        # Bind checkbox events to update "check all" state (can be slow for many items)
        self.Bind(wx.EVT_CHECKBOX, self.setCheck)

        # Layout components
        self.windowSizer = wx.BoxSizer(wx.VERTICAL)
        self.columnSizer = wx.BoxSizer(wx.HORIZONTAL)
        self.buttonSizer = wx.BoxSizer(wx.HORIZONTAL)

        # Build header row with "check all" checkbox and column labels
        self.columnSizer.Add(self.selector, flag=wx.EXPAND | wx.LEFT | wx.RIGHT, border=4)
        self.columnSizer.Add(wx.StaticText(self, label="Scan Type"), flag=wx.EXPAND | wx.LEFT, border=12)
        self.columnSizer.Add(wx.StaticText(self, label="Num. Scans"), flag=wx.EXPAND | wx.LEFT, border=12)
        self.windowSizer.Add(self.columnSizer, flag=wx.EXPAND | wx.ALL, border=4)

        # Add main list control
        self.windowSizer.Add(self.list, proportion=1, flag=wx.EXPAND | wx.ALL, border=4)

        # Add buttons at bottom
        self.buttonSizer.Add(self.apply, proportion=1, flag=wx.EXPAND | wx.ALL, border=4)
        self.buttonSizer.Add(self.cancel, proportion=1, flag=wx.EXPAND | wx.ALL, border=4)
        self.windowSizer.Add(self.buttonSizer, flag=wx.EXPAND | wx.ALL, border=4)

        # Finalize dialog setup
        self.SetSizer(self.windowSizer)
        self.CenterOnScreen()
        self.Bind(wx.EVT_CLOSE, self.onCancel)

    def setCheck(self, event: Optional[wx.CommandEvent]) -> None:
        """Update the master 'check all' checkbox state based on individual item states."""
        num = self.list.GetItemCount()
        if num == 0:
            return

        # Check if all items have the same check state
        firstState = self.list.IsItemChecked(0)
        for i in range(1, num):
            if self.list.IsItemChecked(i) != firstState:
                # Mixed state - some checked, some unchecked
                self.selector.Set3StateValue(wx.CHK_UNDETERMINED)
                return

        # All items have the same state
        self.selector.SetValue(firstState)

    def onCheckAll(self, event: wx.CommandEvent) -> None:
        """Handle master 'check all' checkbox clicks."""
        if self.selector.Get3StateValue():
            self.onSelectAll(None)
        else:
            self.onDeselectAll(None)

    def onSelectAll(self, event: Optional[wx.CommandEvent]) -> None:
        """Check all scan type checkboxes in the list."""
        num = self.list.GetItemCount()
        for i in range(num):
            self.list.CheckItem(i)

    def onDeselectAll(self, event: Optional[wx.CommandEvent]) -> None:
        """Uncheck all scan type checkboxes in the list."""
        num = self.list.GetItemCount()
        for i in range(num):
            self.list.CheckItem(i, False)

    def onApply(self, event: wx.CommandEvent) -> None:
        """Apply scan type filter and update main table."""

        # Collect all checked scan types
        selectedTypes = []
        num = self.list.GetItemCount()
        for i in range(num):
            if self.list.IsItemChecked(i):
                selectedTypes.append(self.list.GetItem(i, 1).GetText())

        # Clear previous type filter cases
        self.GetParent().typeCases = None

        if set(self.allTypes.keys()) == set(selectedTypes):
            # All scan types selected - no filtering needed
            self.GetParent().typeResult = None
        else:
            # Apply scan type filter
            self.GetParent().typeResult = selectedTypes
            # Find scans matching the selected scan types
            thisCase = [scan[10] for scan in self.GetParent().scanItems if scan[6] in selectedTypes]
            self.GetParent().typeCases = thisCase

        self.GetParent().updateTable()
        self.EndModal(wx.ID_OK)

    def onCancel(self, event: wx.CommandEvent) -> None:
        """Close dialog without applying scan type filter changes."""
        self.EndModal(wx.ID_CANCEL)


class DateWindow(wx.Dialog):
    """GUI dialog for filtering scans by date range. Provides dropdown choices for From/To dates including year, month, day, hour, minute."""

    def __init__(
        self,
        parent: Optional[wx.Window] = None,
        possibleYears: list[str] = [],
        allDates: tuple[float, float] = (float("-inf"), float("inf")),
        currentDates: Optional[tuple[float, float]] = None,
    ) -> None:
        """Initialize date range filter dialog with dropdown choices for From/To dates."""
        wx.Dialog.__init__(self, parent, -1, title="Date Range", size=(288, 150))

        # Store date range bounds and current filter values
        self.dateMin, self.dateMax = allDates
        if currentDates is not None:
            self.currentDateMin, self.currentDateMax = currentDates
        else:
            self.currentDateMin, self.currentDateMax = allDates

        # Create choice lists for date/time components
        self.yearChoices = possibleYears if possibleYears else ["N/A"]
        self.monthChoices = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        self.dayChoices = [f"{i:02d}" for i in range(1, 32)]
        self.hourChoices = [f"{i:02d}" for i in range(24)]
        self.minuteChoices = [f"{i:02d}" for i in range(60)]

        # Create the colons for between the hour and minute choices, set to a larger font so they're easier to see
        self.colonText = wx.StaticText(self, label=":")
        self.colonText.SetFont(wx.Font(12, wx.DEFAULT, wx.NORMAL, wx.NORMAL))
        self.colonText2 = wx.StaticText(self, label=":")
        self.colonText2.SetFont(wx.Font(12, wx.DEFAULT, wx.NORMAL, wx.NORMAL))

        # Buttons
        self.apply = wx.Button(self, label="Apply")
        self.cancel = wx.Button(self, label="Cancel")

        # Bindings
        self.apply.Bind(wx.EVT_BUTTON, self.onApply)
        self.cancel.Bind(wx.EVT_BUTTON, self.onCancel)

        # Create the 'From' fields
        self.fromText = wx.StaticText(self, label="From:")
        self.fromMonth = wx.Choice(self, choices=self.monthChoices, size=(46, 21))
        self.fromDay = wx.Choice(self, choices=self.dayChoices, size=(36, 21))
        self.fromYear = wx.Choice(self, choices=self.yearChoices, size=(54, 21))
        self.fromHour = wx.Choice(self, choices=self.hourChoices, size=(36, 21))
        self.fromMinute = wx.Choice(self, choices=self.minuteChoices, size=(36, 21))

        # Set the initial selections for the 'From' fields
        try:
            fromDate = datetime.datetime.utcfromtimestamp(self.currentDateMin)
            self.fromMonth.SetStringSelection(fromDate.strftime("%b"))
            self.fromDay.SetStringSelection(fromDate.strftime("%d"))
            self.fromYear.SetStringSelection(fromDate.strftime("%Y"))
            self.fromHour.SetStringSelection(fromDate.strftime("%H"))
            self.fromMinute.SetStringSelection(fromDate.strftime("%M"))
        except (OverflowError, ValueError):
            self.fromMonth.SetStringSelection("Jan")
            self.fromDay.SetStringSelection("01")
            self.fromYear.SetStringSelection("1970")
            self.fromHour.SetStringSelection("00")
            self.fromMinute.SetStringSelection("00")

        # Create the 'To' fields
        self.toText = wx.StaticText(self, label="To:")
        self.toMonth = wx.Choice(self, choices=self.monthChoices, size=(46, 21))
        self.toDay = wx.Choice(self, choices=self.dayChoices, size=(36, 21))
        self.toYear = wx.Choice(self, choices=self.yearChoices, size=(54, 21))
        self.toHour = wx.Choice(self, choices=self.hourChoices, size=(36, 21))
        self.toMinute = wx.Choice(self, choices=self.minuteChoices, size=(36, 21))

        # Set the initial selections for the 'To' fields
        try:
            toDate = datetime.datetime.utcfromtimestamp(self.currentDateMax)
            self.toMonth.SetStringSelection(toDate.strftime("%b"))
            self.toDay.SetStringSelection(toDate.strftime("%d"))
            self.toYear.SetStringSelection(toDate.strftime("%Y"))
            self.toHour.SetStringSelection(toDate.strftime("%H"))
            self.toMinute.SetStringSelection(toDate.strftime("%M"))
        except (OverflowError, ValueError):
            self.toMonth.SetStringSelection("Jan")
            self.toDay.SetStringSelection("01")
            self.toYear.SetStringSelection("1970")
            self.toHour.SetStringSelection("00")
            self.toMinute.SetStringSelection("00")

        # General layout
        self.windowSizer = wx.BoxSizer(wx.VERTICAL)
        # 'From' values
        self.fromSizer = wx.BoxSizer(wx.HORIZONTAL)
        # 'To' values
        self.toSizer = wx.BoxSizer(wx.HORIZONTAL)
        # Apply and cancel buttons
        self.buttonSizer = wx.BoxSizer(wx.HORIZONTAL)

        # 'From' sizer
        self.fromSizer.Add(self.fromText, flag=wx.EXPAND | wx.TOP | wx.BOTTOM | wx.LEFT, border=4)
        self.fromSizer.Add(self.fromMonth, flag=wx.EXPAND | wx.LEFT, border=2)
        self.fromSizer.Add(self.fromDay, flag=wx.EXPAND | wx.LEFT, border=2)
        self.fromSizer.Add(self.fromYear, flag=wx.EXPAND | wx.LEFT, border=2)
        self.fromSizer.Add(self.fromHour, flag=wx.EXPAND | wx.LEFT, border=8)
        self.fromSizer.Add(self.colonText, flag=wx.EXPAND | wx.LEFT, border=0)
        self.fromSizer.Add(self.fromMinute, flag=wx.EXPAND | wx.LEFT, border=0)
        self.windowSizer.Add(self.fromSizer, flag=wx.EXPAND | wx.ALL, border=4)

        # 'To' sizer
        self.toSizer.Add(self.toText, flag=wx.EXPAND | wx.TOP | wx.BOTTOM | wx.LEFT, border=4)
        self.toSizer.Add(self.toMonth, flag=wx.EXPAND | wx.LEFT, border=14)
        self.toSizer.Add(self.toDay, flag=wx.EXPAND | wx.LEFT, border=2)
        self.toSizer.Add(self.toYear, flag=wx.EXPAND | wx.LEFT, border=2)
        self.toSizer.Add(self.toHour, flag=wx.EXPAND | wx.LEFT, border=8)
        self.toSizer.Add(self.colonText2, flag=wx.EXPAND | wx.LEFT, border=0)
        self.toSizer.Add(self.toMinute, flag=wx.EXPAND | wx.LEFT, border=0)
        self.windowSizer.Add(self.toSizer, flag=wx.EXPAND | wx.ALL, border=4)

        # Button sizer
        self.buttonSizer.Add(self.apply, proportion=1, flag=wx.EXPAND | wx.ALL, border=4)
        self.buttonSizer.Add(self.cancel, proportion=1, flag=wx.EXPAND | wx.ALL, border=4)
        self.windowSizer.Add(self.buttonSizer, flag=wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, border=4)

        # Create the window, center it, and bind the close button
        self.SetSizer(self.windowSizer)
        self.CenterOnScreen()
        self.Bind(wx.EVT_CLOSE, self.onCancel)

    def onApply(self, event: wx.CommandEvent) -> None:
        """Apply date range filter and update main table."""

        # Parse selected From and To dates from dropdown choices
        try:
            fromDateStr = f"{self.fromYear.GetStringSelection()}-{self.fromMonth.GetStringSelection()}-{self.fromDay.GetStringSelection()} {self.fromHour.GetStringSelection()}:{self.fromMinute.GetStringSelection()}:00"
            fromDate = datetime.datetime.strptime(fromDateStr, "%Y-%b-%d %H:%M:%S").timestamp()
            toDateStr = f"{self.toYear.GetStringSelection()}-{self.toMonth.GetStringSelection()}-{self.toDay.GetStringSelection()} {self.toHour.GetStringSelection()}:{self.toMinute.GetStringSelection()}:00"
            toDate = datetime.datetime.strptime(toDateStr, "%Y-%b-%d %H:%M:%S").timestamp()
        except ValueError as e:
            print(f"Error parsing date: {e}")
            self.Destroy()
            return

        # Validate that From date is not after To date
        if fromDate > toDate:
            print("Error: 'To' date precedes 'From' date")
            self.Destroy()
            return

        # Clear previous date filter cases
        self.GetParent().dateCases = None

        if fromDate <= self.dateMin and toDate >= self.dateMax:
            # Date range covers entire dataset - no filtering needed
            self.GetParent().dateResult = None
        else:
            # Apply date range filter
            self.GetParent().dateResult = (fromDate, toDate)

            # Find scans with date >= fromDate
            thisCase = [scan[10] for scan in self.GetParent().scanItems if datetime.datetime.strptime(scan[8], "%a %b %d %H:%M:%S %Y").timestamp() >= fromDate]

            # Find scans with date <= toDate
            thatCase = [scan[10] for scan in self.GetParent().scanItems if datetime.datetime.strptime(scan[8], "%a %b %d %H:%M:%S %Y").timestamp() <= toDate]

            # Intersection gives scans within the date range
            bothCases = list_intersect(thisCase, thatCase)
            self.GetParent().dateCases = bothCases

        self.GetParent().updateTable()
        self.EndModal(wx.ID_OK)

    def onCancel(self, event: wx.CommandEvent) -> None:
        """Close dialog without applying date range filter changes."""
        self.EndModal(wx.ID_CANCEL)


class TableDataCtrl(wx.ListCtrl, listmix.ListCtrlAutoWidthMixin):
    """Enhanced list control that combines standard wxPython ListCtrl with auto-width functionality."""

    def __init__(self, parent: wx.Window, ID: int = -1, pos: wx.Point = wx.DefaultPosition, size: wx.Size = wx.DefaultSize, style: int = 0) -> None:
        """Initialize the table data control with auto-width capability."""
        # Initialize base ListCtrl with standard functionality
        wx.ListCtrl.__init__(self, parent, ID, pos, size, style)

        # Add auto-width mixin to automatically size the last column
        listmix.ListCtrlAutoWidthMixin.__init__(self)


class NumberListCtrl(wx.ListCtrl, listmix.ListCtrlAutoWidthMixin):
    """Enhanced list control with checkbox support and auto-width functionality."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the checkable list control with auto-width capability."""
        # Initialize base ListCtrl with provided arguments
        wx.ListCtrl.__init__(self, *args, **kwargs)

        # Add auto-width mixin for automatic column sizing
        listmix.ListCtrlAutoWidthMixin.__init__(self)

        # Enable checkboxes for each list item
        self.EnableCheckBoxes(True)

    def OnCheckItem(self, index: int, flag: bool) -> None:
        """Handle checkbox state changes and notify parent components."""
        # Post checkbox event to parent to allow state updates
        wx.PostEvent(self.GetEventHandler(), wx.CommandEvent(wx.EVT_CHECKBOX.typeId, self.GetId()))


class myTreeCtrl(wx.TreeCtrl):
    """Enhanced tree control with custom sorting and level detection functionality."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the enhanced tree control."""
        wx.TreeCtrl.__init__(self, *args, **kwargs)

    def getLevel(self, item: wx.TreeItemId) -> int:
        """Determine the hierarchical level of a tree item."""
        level = 0
        rootItem = self.GetRootItem()

        # Traverse up the tree, counting levels until we reach the root
        while item != rootItem:
            level += 1
            item = self.GetItemParent(item)

        return level

    def OnCompareItems(self, item1: wx.TreeItemId, item2: wx.TreeItemId) -> int:
        """Custom comparison function for tree item sorting."""
        text1 = self.GetItemText(item1)
        text2 = self.GetItemText(item2)

        try:
            # Attempt numeric comparison for proper integer sorting
            int1 = int(text1)
            int2 = int(text2)
            return (int1 > int2) - (int1 < int2)
        except ValueError:
            # Fall back to string comparison if numeric conversion fails
            return (text1 > text2) - (text1 < text2)
