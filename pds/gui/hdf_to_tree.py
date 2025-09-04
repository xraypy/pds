from typing import Any

import wx

from pds.utils.converters import bytes_to_str


class HDFToTree:
    def __init__(self) -> None:
        self.reverseLookup: dict[Any, wx.TreeItemId] = {}

    def dictToTree(self, thisDict: dict[str, Any], thisTree: wx.TreeCtrl, thisRoot: wx.TreeItemId) -> None:
        """Convert a nested dictionary into a tree structure."""
        for key, value in thisDict.items():
            if type(value) is dict:
                parentItem = thisTree.AppendItem(thisRoot, str(key))
                thisTree.SetItemData(parentItem, None)
                self.dictToTree(value, thisTree, parentItem)
            else:
                childItem = thisTree.AppendItem(thisRoot, str(key))
                thisTree.SetItemData(childItem, value)
        thisTree.SortChildren(thisRoot)

    def populateTree(self, thisTree: wx.TreeCtrl, thisObject: Any) -> None:
        """Build a dictionary from the hdf_data object to populate the tree."""
        rootName = thisObject.fname.split("/")[-1].split("\\")[-1]
        treeRoot = thisTree.AddRoot(rootName)
        hdfDict = {}
        for item in thisObject.all_items:
            try:
                item0 = item[0]
                item1 = item[1]
                itemString = item1.attrs.get("name")
                itemString = bytes_to_str(itemString)

                if itemString is None or ":" not in itemString:
                    print(f"Warning: Skipping item with invalid name: {itemString}")
                    continue

                name_parts = itemString.split(":")
                if len(name_parts) < 4:
                    print(f"Warning: Skipping item with insufficient name parts: {itemString}")
                    continue

                (specName, scanNum, pointNum, epoch) = name_parts[:4]
                scanNum = "Scan " + scanNum[1:]
                pointNum = pointNum.split("/")[0]
                pointNum = "Point " + pointNum[1:]

                if specName not in hdfDict:
                    hdfDict[specName] = {scanNum: {pointNum: item0}}
                elif scanNum not in hdfDict[specName]:
                    hdfDict[specName][scanNum] = {pointNum: item0}
                elif pointNum not in hdfDict[specName][scanNum]:
                    hdfDict[specName][scanNum][pointNum] = item0
                else:
                    print("Error: Duplicate point")
                    print("Specfile: " + specName)
                    print("Scan number: " + scanNum)
                    print("Point number: " + pointNum)
                    continue
            except Exception as e:
                print(f"Warning: Error processing item {item}: {e}")
                continue

        self.dictToTree(hdfDict, thisTree, treeRoot)
        self.populateReverse(thisTree, treeRoot)

    def populateReverse(self, thisTree: wx.TreeCtrl, thisRoot: wx.TreeItemId) -> None:
        """Create a reverse lookup for hashable tree item data."""
        item, cookie = thisTree.GetFirstChild(thisRoot)
        while item:
            iterData = thisTree.GetItemData(item)
            if iterData is not None:
                try:
                    hash(iterData)
                    self.reverseLookup[iterData] = item
                except TypeError:
                    pass
            else:
                self.populateReverse(thisTree, item)
            item, cookie = thisTree.GetNextChild(thisRoot, cookie)

    def deleteItem(self, thisTree: wx.TreeCtrl, thisObject: Any, thisItem: wx.TreeItemId) -> None:
        """Remove an item and its children from the tree and object."""
        thisData = thisTree.GetItemData(thisItem)
        if thisData is not None:
            thisObject.delete(thisData)
        elif thisTree.ItemHasChildren(thisItem):
            item, cookie = thisTree.GetFirstChild(thisItem)
            while item:
                item2, cookie2 = thisTree.GetNextChild(thisItem, cookie)
                self.deleteItem(thisTree, thisObject, item)
                item, cookie = item2, cookie2
        thisTree.Delete(thisItem)

    def getRelevantChildren(self, thisTree: wx.TreeCtrl, thisObject: Any, thisItem: wx.TreeItemId) -> list[Any]:
        """Return a list of all relevant children associated with a point."""
        thisData = thisTree.GetItemData(thisItem)
        if thisData is not None:
            return [thisData]
        elif thisTree.ItemHasChildren(thisItem):
            returnList = []
            item, cookie = thisTree.GetFirstChild(thisItem)
            while item:
                returnList.extend(self.getRelevantChildren(thisTree, thisObject, item))
                item, cookie = thisTree.GetNextChild(thisItem, cookie)
            return returnList
        else:
            return []

    def statusString(self, thisTree: wx.TreeCtrl, thisObject: Any, thisItem: wx.TreeItemId) -> str:
        """Return a string description of the given tree item."""
        thisData = thisTree.GetItemData(thisItem)
        if thisData is not None:
            try:
                thisName = thisObject[thisData]["name"]
                thisName = bytes_to_str(thisName)

                name_parts = thisName.split(":")
                if len(name_parts) < 4:
                    return f"Item {thisData} (malformed name)"

                thisSpec, thisScan, thisPoint, thisTime = name_parts[:4]
                thisScan = thisScan[1:] if len(thisScan) > 0 else "unknown"
                thisPoint = thisPoint.split("/")[0][1:] if len(thisPoint) > 0 else "unknown"
                return "Scan " + thisScan + ", Point " + thisPoint
            except Exception as e:
                return f"Item {thisData} (error: {e})"
        else:
            return thisTree.GetItemText(thisItem)


class myTreeCtrl(wx.TreeCtrl):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        wx.TreeCtrl.__init__(self, *args, **kwargs)

    def OnCompareItems(self, item1: wx.TreeItemId, item2: wx.TreeItemId) -> int:
        """Custom comparison for numeric sorting of scans/points."""
        item1Str = self.GetItemText(item1)
        item2Str = self.GetItemText(item2)

        if item1Str.startswith("Scan") or item1Str.startswith("Point"):
            try:
                item1Parts = item1Str.split(" ")
                item2Parts = item2Str.split(" ")

                if len(item1Parts) < 2 or len(item2Parts) < 2:
                    return (item1Str > item2Str) - (item1Str < item2Str)

                item1Num = int(item1Parts[1])
                item2Num = int(item2Parts[1])
                return (item1Num > item2Num) - (item1Num < item2Num)
            except (ValueError, IndexError):
                return (item1Str > item2Str) - (item1Str < item2Str)
        else:
            return (item1Str > item2Str) - (item1Str < item2Str)
