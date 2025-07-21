import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import h5py
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))

from pds.gui.hdf_to_tree import HDFToTree, myTreeCtrl


class MockTreeCtrl:
    """Mock wx.TreeCtrl for testing without wx dependency."""

    def __init__(self):
        self.items = {}
        self.data = {}
        self.root_item = MockTreeItemId("root")
        self.next_id = 1

    def AddRoot(self, text):
        self.root_item.text = text
        return self.root_item

    def AppendItem(self, parent, text):
        item_id = MockTreeItemId(f"item_{self.next_id}")
        item_id.text = text
        item_id.parent = parent
        self.next_id += 1
        if parent not in self.items:
            self.items[parent] = []
        self.items[parent].append(item_id)
        return item_id

    def SetItemData(self, item, data):
        self.data[item] = data

    def GetItemData(self, item):
        return self.data.get(item)

    def GetItemText(self, item):
        return getattr(item, "text", str(item))

    def SortChildren(self, parent):
        if parent in self.items:
            self.items[parent].sort(key=lambda x: x.text)

    def GetFirstChild(self, parent):
        if parent in self.items and self.items[parent]:
            return self.items[parent][0], 0
        return None, None

    def GetNextChild(self, parent, cookie):
        if parent in self.items and cookie + 1 < len(self.items[parent]):
            return self.items[parent][cookie + 1], cookie + 1
        return None, None

    def ItemHasChildren(self, item):
        return item in self.items and len(self.items[item]) > 0

    def Delete(self, item):
        # Remove from parent's children
        for parent, children in self.items.items():
            if item in children:
                children.remove(item)
        # Remove item's own data
        if item in self.data:
            del self.data[item]
        if item in self.items:
            del self.items[item]


class MockTreeItemId:
    """Mock wx.TreeItemId for testing."""

    def __init__(self, id_val):
        self.id = id_val
        self.text = ""
        self.parent = None

    def __eq__(self, other):
        return isinstance(other, MockTreeItemId) and self.id == other.id

    def __hash__(self):
        return hash(self.id)

    def __str__(self):
        return f"MockTreeItemId({self.id})"


class MockHdfObject:
    """Mock HDF data object for testing."""

    def __init__(self, fname, all_items):
        self.fname = fname
        self.all_items = all_items
        self._deleted_items = set()

    def delete(self, item):
        self._deleted_items.add(item)

    def __getitem__(self, key):
        if key in self._deleted_items:
            raise KeyError(f"Item {key} has been deleted")
        return {"name": f"testspec:s1:p{key}:1234567890"}


class TestHDFToTree:
    """Test the HDFToTree class functionality for Python 2/3 compatibility."""

    def setup_method(self):
        """Set up test environment with mock objects and test data."""
        self.hdf_to_tree = HDFToTree()
        self.mock_tree = MockTreeCtrl()

        # Create temporary HDF5 file for testing
        self.temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".h5")
        self.temp_file.close()
        self.fname = self.temp_file.name

        # Create test HDF5 structure
        with h5py.File(self.fname, "w") as f:
            # Create multiple points with name attributes
            point1 = f.create_group("000001")
            point1.attrs["name"] = "testspec:s1:p1:1234567890"

            point2 = f.create_group("000002")
            point2.attrs["name"] = "testspec:s1:p2:1234567891"

            point3 = f.create_group("000003")
            point3.attrs["name"] = "testspec:s2:p1:1234567892"

            # Test malformed names for error handling
            point4 = f.create_group("000004")
            point4.attrs["name"] = "malformed_name_no_colons"

            point5 = f.create_group("000005")
            point5.attrs["name"] = "incomplete:s1"  # Missing parts

        # Create mock all_items structure
        with h5py.File(self.fname, "r") as f:
            self.all_items = []
            for key in f.keys():
                group = f[key]
                self.all_items.append((key, group))

        self.mock_hdf_object = MockHdfObject(self.fname, self.all_items)

    def teardown_method(self):
        """Clean up temporary files."""
        if os.path.exists(self.fname):
            os.unlink(self.fname)

    def test_init(self):
        """Test HDFToTree initialization."""
        hdf_tree = HDFToTree()
        assert hasattr(hdf_tree, "reverseLookup")
        assert isinstance(hdf_tree.reverseLookup, dict)
        assert len(hdf_tree.reverseLookup) == 0

    def test_dictToTree_simple_dict(self):
        """Test dictToTree with a simple nested dictionary."""
        test_dict = {"spec1": {"Scan 1": {"Point 1": "data1", "Point 2": "data2"}}}

        root = self.mock_tree.AddRoot("test_root")
        self.hdf_to_tree.dictToTree(test_dict, self.mock_tree, root)

        # Check that items were added
        assert root in self.mock_tree.items
        assert len(self.mock_tree.items[root]) == 1

        spec_item = self.mock_tree.items[root][0]
        assert self.mock_tree.GetItemText(spec_item) == "spec1"
        assert self.mock_tree.GetItemData(spec_item) is None

    def test_dictToTree_with_leaf_nodes(self):
        """Test dictToTree with both branch and leaf nodes."""
        test_dict = {"branch": {"leaf1": "value1"}, "direct_leaf": "value2"}

        root = self.mock_tree.AddRoot("test_root")
        self.hdf_to_tree.dictToTree(test_dict, self.mock_tree, root)

        # Verify structure
        root_children = self.mock_tree.items.get(root, [])
        assert len(root_children) == 2

        # Find items by text
        branch_item = next(item for item in root_children if self.mock_tree.GetItemText(item) == "branch")
        leaf_item = next(item for item in root_children if self.mock_tree.GetItemText(item) == "direct_leaf")

        assert self.mock_tree.GetItemData(branch_item) is None
        assert self.mock_tree.GetItemData(leaf_item) == "value2"

    def test_populateTree_normal_data(self):
        """Test populateTree with normal HDF data structure."""
        # Create a proper mock object with expected structure
        mock_hdf_object = Mock()
        mock_hdf_object.fname = "/path/to/testfile.h5"

        # Mock item structure with proper name attributes
        mock_item1 = Mock()
        mock_item1.attrs = {"name": "testspec:s1:p1:1234567890"}

        mock_item2 = Mock()
        mock_item2.attrs = {"name": "testspec:s1:p2:1234567891"}

        mock_item3 = Mock()
        mock_item3.attrs = {"name": "testspec:s2:p1:1234567892"}

        mock_hdf_object.all_items = [("000001", mock_item1), ("000002", mock_item2), ("000003", mock_item3)]

        with patch("pds.gui.hdf_to_tree.bytes_to_str", side_effect=lambda x: x):
            self.hdf_to_tree.populateTree(self.mock_tree, mock_hdf_object)

        # Check root was created
        assert self.mock_tree.root_item.text == "testfile.h5"

        # Verify tree structure was built
        assert self.mock_tree.root_item in self.mock_tree.items
        root_children = self.mock_tree.items[self.mock_tree.root_item]
        assert len(root_children) == 1  # Should have one spec

    def test_populateTree_malformed_names(self):
        """Test populateTree handles malformed item names gracefully."""
        mock_hdf_object = Mock()
        mock_hdf_object.fname = "/path/to/testfile.h5"

        # Mock items with malformed names
        mock_item1 = Mock()
        mock_item1.attrs = {"name": "no_colons_here"}

        mock_item2 = Mock()
        mock_item2.attrs = {"name": "incomplete:s1"}  # Missing parts

        mock_item3 = Mock()
        mock_item3.attrs = {"name": None}  # None name

        mock_hdf_object.all_items = [("000001", mock_item1), ("000002", mock_item2), ("000003", mock_item3)]

        with patch("pds.gui.hdf_to_tree.bytes_to_str", side_effect=lambda x: x):
            with patch("builtins.print") as mock_print:
                self.hdf_to_tree.populateTree(self.mock_tree, mock_hdf_object)

                # Should have printed warnings
                assert mock_print.call_count >= 3

    def test_populateTree_exception_handling(self):
        """Test populateTree handles exceptions during item processing."""
        mock_hdf_object = Mock()
        mock_hdf_object.fname = "/path/to/testfile.h5"

        # Mock item that raises an exception
        mock_item1 = Mock()
        mock_item1.attrs.get.side_effect = Exception("Test exception")

        mock_hdf_object.all_items = [("000001", mock_item1)]

        with patch("builtins.print") as mock_print:
            self.hdf_to_tree.populateTree(self.mock_tree, mock_hdf_object)

            # Should have printed warning about error
            assert mock_print.call_count >= 1

    def test_populateReverse_hashable_data(self):
        """Test populateReverse with hashable data items."""
        # Set up tree with some data
        root = self.mock_tree.AddRoot("root")
        child1 = self.mock_tree.AppendItem(root, "child1")
        child2 = self.mock_tree.AppendItem(root, "child2")

        # Set some hashable data
        self.mock_tree.SetItemData(child1, "hashable_string")
        self.mock_tree.SetItemData(child2, 12345)

        self.hdf_to_tree.populateReverse(self.mock_tree, root)

        # Check reverse lookup was populated
        assert "hashable_string" in self.hdf_to_tree.reverseLookup
        assert 12345 in self.hdf_to_tree.reverseLookup
        assert self.hdf_to_tree.reverseLookup["hashable_string"] == child1
        assert self.hdf_to_tree.reverseLookup[12345] == child2

    def test_populateReverse_unhashable_data(self):
        """Test populateReverse gracefully handles unhashable data."""
        root = self.mock_tree.AddRoot("root")
        child = self.mock_tree.AppendItem(root, "child")

        # Set unhashable data (dict)
        unhashable_data = {"key": "value"}
        self.mock_tree.SetItemData(child, unhashable_data)

        # Should not raise exception
        self.hdf_to_tree.populateReverse(self.mock_tree, root)

        # Should not be in reverse lookup
        assert len(self.hdf_to_tree.reverseLookup) == 0

    def test_populateReverse_recursive(self):
        """Test populateReverse handles nested structure recursively."""
        root = self.mock_tree.AddRoot("root")
        branch = self.mock_tree.AppendItem(root, "branch")
        leaf = self.mock_tree.AppendItem(branch, "leaf")

        # Branch has no data, leaf has data
        self.mock_tree.SetItemData(branch, None)
        self.mock_tree.SetItemData(leaf, "leaf_data")

        self.hdf_to_tree.populateReverse(self.mock_tree, root)

        # Should have found the leaf data
        assert "leaf_data" in self.hdf_to_tree.reverseLookup
        assert self.hdf_to_tree.reverseLookup["leaf_data"] == leaf

    def test_deleteItem_with_data(self):
        """Test deleteItem removes items with data from tree and object."""
        mock_hdf_object = Mock()

        root = self.mock_tree.AddRoot("root")
        item = self.mock_tree.AppendItem(root, "item")
        self.mock_tree.SetItemData(item, "test_data")

        self.hdf_to_tree.deleteItem(self.mock_tree, mock_hdf_object, item)

        # Should have called delete on hdf object
        mock_hdf_object.delete.assert_called_once_with("test_data")

        # Item should be removed from tree
        assert item not in self.mock_tree.data
        assert item not in self.mock_tree.items

    def test_deleteItem_with_children(self):
        """Test deleteItem recursively removes children."""
        mock_hdf_object = Mock()

        root = self.mock_tree.AddRoot("root")
        parent = self.mock_tree.AppendItem(root, "parent")
        child1 = self.mock_tree.AppendItem(parent, "child1")
        child2 = self.mock_tree.AppendItem(parent, "child2")

        self.mock_tree.SetItemData(parent, None)
        self.mock_tree.SetItemData(child1, "child1_data")
        self.mock_tree.SetItemData(child2, "child2_data")

        self.hdf_to_tree.deleteItem(self.mock_tree, mock_hdf_object, parent)

        # Should have called delete for children with data
        assert mock_hdf_object.delete.call_count == 2
        mock_hdf_object.delete.assert_any_call("child1_data")
        mock_hdf_object.delete.assert_any_call("child2_data")

    def test_getRelevantChildren_leaf_node(self):
        """Test getRelevantChildren with leaf node returns its data."""
        root = self.mock_tree.AddRoot("root")
        leaf = self.mock_tree.AppendItem(root, "leaf")
        self.mock_tree.SetItemData(leaf, "leaf_data")

        result = self.hdf_to_tree.getRelevantChildren(self.mock_tree, Mock(), leaf)

        assert result == ["leaf_data"]

    def test_getRelevantChildren_branch_node(self):
        """Test getRelevantChildren with branch node returns all descendant data."""
        mock_hdf_object = Mock()

        root = self.mock_tree.AddRoot("root")
        branch = self.mock_tree.AppendItem(root, "branch")
        leaf1 = self.mock_tree.AppendItem(branch, "leaf1")
        leaf2 = self.mock_tree.AppendItem(branch, "leaf2")

        self.mock_tree.SetItemData(branch, None)
        self.mock_tree.SetItemData(leaf1, "data1")
        self.mock_tree.SetItemData(leaf2, "data2")

        result = self.hdf_to_tree.getRelevantChildren(self.mock_tree, mock_hdf_object, branch)

        assert set(result) == {"data1", "data2"}

    def test_getRelevantChildren_empty_node(self):
        """Test getRelevantChildren with node that has no children or data."""
        root = self.mock_tree.AddRoot("root")
        empty = self.mock_tree.AppendItem(root, "empty")
        self.mock_tree.SetItemData(empty, None)

        result = self.hdf_to_tree.getRelevantChildren(self.mock_tree, Mock(), empty)

        assert result == []

    def test_statusString_with_valid_data(self):
        """Test statusString returns formatted string for valid item data."""
        mock_hdf_object = Mock()
        mock_hdf_object.__getitem__ = Mock(return_value={"name": "testspec:s5:p10:1234567890"})

        root = self.mock_tree.AddRoot("root")
        item = self.mock_tree.AppendItem(root, "item")
        self.mock_tree.SetItemData(item, "test_key")

        with patch("pds.gui.hdf_to_tree.bytes_to_str", side_effect=lambda x: x):
            result = self.hdf_to_tree.statusString(self.mock_tree, mock_hdf_object, item)

        assert result == "Scan 5, Point 10"

    def test_statusString_with_malformed_name(self):
        """Test statusString handles malformed names gracefully."""
        mock_hdf_object = Mock()
        mock_hdf_object.__getitem__ = Mock(return_value={"name": "malformed_name"})

        root = self.mock_tree.AddRoot("root")
        item = self.mock_tree.AppendItem(root, "item")
        self.mock_tree.SetItemData(item, "test_key")

        with patch("pds.gui.hdf_to_tree.bytes_to_str", side_effect=lambda x: x):
            result = self.hdf_to_tree.statusString(self.mock_tree, mock_hdf_object, item)

        assert "malformed name" in result
        assert "test_key" in result

    def test_statusString_with_exception(self):
        """Test statusString handles exceptions during data access."""
        mock_hdf_object = Mock()
        mock_hdf_object.__getitem__ = Mock(side_effect=Exception("Access error"))

        root = self.mock_tree.AddRoot("root")
        item = self.mock_tree.AppendItem(root, "item")
        self.mock_tree.SetItemData(item, "test_key")

        result = self.hdf_to_tree.statusString(self.mock_tree, mock_hdf_object, item)

        assert "error" in result.lower()
        assert "test_key" in result

    def test_statusString_without_data(self):
        """Test statusString returns item text when no data is associated."""
        root = self.mock_tree.AddRoot("root")
        item = self.mock_tree.AppendItem(root, "test_item_text")
        self.mock_tree.SetItemData(item, None)

        result = self.hdf_to_tree.statusString(self.mock_tree, Mock(), item)

        assert result == "test_item_text"

    def test_statusString_bytes_handling(self):
        """Test statusString properly handles bytes in names (Python 2/3 compatibility)."""
        mock_hdf_object = Mock()
        mock_hdf_object.__getitem__ = Mock(return_value={"name": b"testspec:s3:p5:1234567890"})

        root = self.mock_tree.AddRoot("root")
        item = self.mock_tree.AppendItem(root, "item")
        self.mock_tree.SetItemData(item, "test_key")

        with patch("pds.gui.hdf_to_tree.bytes_to_str") as mock_bytes_to_str:
            mock_bytes_to_str.return_value = "testspec:s3:p5:1234567890"
            result = self.hdf_to_tree.statusString(self.mock_tree, mock_hdf_object, item)

        assert result == "Scan 3, Point 5"
        mock_bytes_to_str.assert_called_once()


class TestMyTreeCtrl:
    """Test the myTreeCtrl class functionality."""

    def setup_method(self):
        """Set up test environment."""
        # Create a mock wx.TreeCtrl class
        with patch("pds.gui.hdf_to_tree.wx.TreeCtrl"):
            self.tree_ctrl = myTreeCtrl()

    def test_init(self):
        """Test myTreeCtrl initialization."""
        with patch("pds.gui.hdf_to_tree.wx.TreeCtrl.__init__") as mock_init:
            tree = myTreeCtrl("arg1", "arg2", kwarg1="value1")
            mock_init.assert_called_once_with(tree, "arg1", "arg2", kwarg1="value1")

    def test_OnCompareItems_scan_comparison(self):
        """Test OnCompareItems with scan items for numeric sorting."""
        mock_item1 = MockTreeItemId("item1")
        mock_item2 = MockTreeItemId("item2")

        with patch.object(self.tree_ctrl, "GetItemText") as mock_get_text:
            # Test numeric comparison: Scan 2 vs Scan 10
            mock_get_text.side_effect = ["Scan 2", "Scan 10"]
            result = self.tree_ctrl.OnCompareItems(mock_item1, mock_item2)
            assert result < 0  # Scan 2 should come before Scan 10

            # Test reverse comparison: Scan 10 vs Scan 2
            mock_get_text.side_effect = ["Scan 10", "Scan 2"]
            result = self.tree_ctrl.OnCompareItems(mock_item1, mock_item2)
            assert result > 0  # Scan 10 should come after Scan 2

    def test_OnCompareItems_point_comparison(self):
        """Test OnCompareItems with point items for numeric sorting."""
        mock_item1 = MockTreeItemId("item1")
        mock_item2 = MockTreeItemId("item2")

        with patch.object(self.tree_ctrl, "GetItemText") as mock_get_text:
            # Test numeric comparison: Point 5 vs Point 15
            mock_get_text.side_effect = ["Point 5", "Point 15"]
            result = self.tree_ctrl.OnCompareItems(mock_item1, mock_item2)
            assert result < 0  # Point 5 should come before Point 15

            # Test equal comparison
            mock_get_text.side_effect = ["Point 7", "Point 7"]
            result = self.tree_ctrl.OnCompareItems(mock_item1, mock_item2)
            assert result == 0  # Should be equal

    def test_OnCompareItems_string_fallback(self):
        """Test OnCompareItems falls back to string comparison for non-scan/point items."""
        mock_item1 = MockTreeItemId("item1")
        mock_item2 = MockTreeItemId("item2")

        with patch.object(self.tree_ctrl, "GetItemText") as mock_get_text:
            # Test alphabetical comparison
            mock_get_text.side_effect = ["apple", "banana"]
            result = self.tree_ctrl.OnCompareItems(mock_item1, mock_item2)
            assert result < 0  # "apple" comes before "banana"

            mock_get_text.side_effect = ["zebra", "apple"]
            result = self.tree_ctrl.OnCompareItems(mock_item1, mock_item2)
            assert result > 0  # "zebra" comes after "apple"

    def test_OnCompareItems_malformed_scan_point(self):
        """Test OnCompareItems handles malformed scan/point strings gracefully."""
        mock_item1 = MockTreeItemId("item1")
        mock_item2 = MockTreeItemId("item2")

        with patch.object(self.tree_ctrl, "GetItemText") as mock_get_text:
            # Test malformed scan (missing number)
            mock_get_text.side_effect = ["Scan", "Scan 5"]
            result = self.tree_ctrl.OnCompareItems(mock_item1, mock_item2)
            # Should fall back to string comparison
            assert result < 0  # "Scan" comes before "Scan 5" alphabetically

            # Test invalid number in scan
            mock_get_text.side_effect = ["Scan abc", "Scan 5"]
            result = self.tree_ctrl.OnCompareItems(mock_item1, mock_item2)
            # Should fall back to string comparison
            assert result > 0  # "Scan abc" comes after "Scan 5" alphabetically

    def test_OnCompareItems_edge_cases(self):
        """Test OnCompareItems with edge cases and error conditions."""
        mock_item1 = MockTreeItemId("item1")
        mock_item2 = MockTreeItemId("item2")

        with patch.object(self.tree_ctrl, "GetItemText") as mock_get_text:
            # Test empty strings
            mock_get_text.side_effect = ["", "Scan 1"]
            result = self.tree_ctrl.OnCompareItems(mock_item1, mock_item2)
            assert result < 0  # Empty string comes first

            # Test scan with zero
            mock_get_text.side_effect = ["Scan 0", "Scan 1"]
            result = self.tree_ctrl.OnCompareItems(mock_item1, mock_item2)
            assert result < 0  # Scan 0 comes before Scan 1

            # Test negative scan numbers (should still work)
            mock_get_text.side_effect = ["Scan -1", "Scan 1"]
            result = self.tree_ctrl.OnCompareItems(mock_item1, mock_item2)
            assert result < 0  # Scan -1 comes before Scan 1


class TestIntegration:
    """Integration tests for HDFToTree with real HDF5 data."""

    def setup_method(self):
        """Set up test environment with realistic HDF5 data."""
        self.temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".h5")
        self.temp_file.close()
        self.fname = self.temp_file.name

        # Create comprehensive HDF5 structure similar to original Python 2 data
        with h5py.File(self.fname, "w") as f:
            # Create multiple spec files and scans
            for spec_num in range(1, 3):
                for scan_num in range(1, 4):
                    for point_num in range(1, 3):
                        group_name = f"{spec_num:06d}{scan_num:02d}{point_num:02d}"
                        group = f.create_group(group_name)

                        # Create name attribute with proper format
                        name = f"spec{spec_num}:s{scan_num}:p{point_num}:123456789{point_num}"
                        # Mix bytes and string types for Python 2/3 compatibility testing
                        if point_num % 2 == 0:
                            group.attrs["name"] = name.encode("utf-8")  # Bytes
                        else:
                            group.attrs["name"] = name  # String

                        # Add some sample data
                        group.create_dataset("data", data=np.random.random(10))

        self.hdf_to_tree = HDFToTree()
        self.mock_tree = MockTreeCtrl()

        # Create mock HDF object structure
        with h5py.File(self.fname, "r") as f:
            self.all_items = []
            for key in f.keys():
                group = f[key]
                self.all_items.append((key, group))

    def teardown_method(self):
        """Clean up temporary files."""
        if os.path.exists(self.fname):
            os.unlink(self.fname)

    def test_full_populate_tree_workflow(self):
        """Test the complete workflow of populating tree from HDF5 data."""
        # Create mock items that properly simulate the HDF5 structure
        mock_items = []
        for spec_num in range(1, 3):
            for scan_num in range(1, 3):
                for point_num in range(1, 3):
                    group_name = f"{spec_num:06d}{scan_num:02d}{point_num:02d}"
                    mock_group = Mock()
                    name = f"spec{spec_num}:s{scan_num}:p{point_num}:123456789{point_num}"
                    mock_group.attrs = {"name": name}
                    mock_items.append((group_name, mock_group))

        # Create mock hdf object
        mock_hdf_object = Mock()
        mock_hdf_object.fname = self.fname
        mock_hdf_object.all_items = mock_items

        with patch("pds.gui.hdf_to_tree.bytes_to_str", side_effect=lambda x: x.decode("utf-8") if isinstance(x, bytes) else x):
            self.hdf_to_tree.populateTree(self.mock_tree, mock_hdf_object)

        # Verify root was created with filename
        assert self.fname.split("/")[-1] in self.mock_tree.root_item.text

        # Verify tree structure was built hierarchically
        root_children = self.mock_tree.items.get(self.mock_tree.root_item, [])
        assert len(root_children) > 0  # Should have spec files

        # Verify reverse lookup was populated
        assert len(self.hdf_to_tree.reverseLookup) > 0

    def test_python2_python3_compatibility(self):
        """Test that the tree handles both bytes and string data correctly."""
        # Test data with mixed bytes/string types
        test_items = []

        # Mock items with both bytes and string name attributes
        mock_item1 = Mock()
        mock_item1.attrs = {"name": b"testspec:s1:p1:1234567890"}  # Bytes

        mock_item2 = Mock()
        mock_item2.attrs = {"name": "testspec:s1:p2:1234567891"}  # String

        test_items = [("000001", mock_item1), ("000002", mock_item2)]

        mock_hdf_object = Mock()
        mock_hdf_object.fname = "/test/file.h5"
        mock_hdf_object.all_items = test_items

        with patch("pds.gui.hdf_to_tree.bytes_to_str") as mock_converter:
            # Configure mock to simulate proper bytes_to_str behavior
            def convert_bytes(value):
                if isinstance(value, bytes):
                    return value.decode("utf-8")
                return value

            mock_converter.side_effect = convert_bytes
            self.hdf_to_tree.populateTree(self.mock_tree, mock_hdf_object)

            # Verify bytes_to_str was called for bytes data
            assert mock_converter.call_count >= 2


if __name__ == "__main__":
    pytest.main([__file__])
