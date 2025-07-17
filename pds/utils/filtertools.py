from typing import Any

import h5py
import numpy as np


def is_before(am_i: str, before_me: str) -> bool:
    """Check if first date comes before second date."""
    all_months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    (test_day, test_month, test_date, test_time, test_year) = am_i.split()
    (other_day, other_month, other_date, other_time, other_year) = before_me.split()
    if int(test_year) < int(other_year):
        return True
    elif int(test_year) == int(other_year):
        if all_months.index(test_month) < all_months.index(other_month):
            return True
        elif all_months.index(test_month) == all_months.index(other_month):
            if int(test_date) < int(other_date):
                return True
            elif int(test_date) == int(other_date):
                if test_time < other_time:
                    return True
    return False


def is_after(am_i: str, after_me: str) -> bool:
    """Check if first date comes after second date."""
    return am_i == after_me or not is_before(am_i, after_me)


def list_union(*args: list[Any]) -> list[Any]:
    """Return union of given lists, filtering out None values."""
    # Filter out None values
    filtered_args = [arg for arg in args if arg is not None]
    if not filtered_args:
        return []

    union = [item for sublist in filtered_args for item in sublist]
    return list(set(union))


def list_intersect(*args: list[Any]) -> list[Any]:
    """Return intersection of given lists."""
    if not args:
        return []

    intersect = set(args[0])
    for entry in args:
        intersect = intersect.intersection(entry)
    return list(intersect)


def cases(in_here: h5py.File, of_this: str, such_that: str, this_level: str = "scan") -> list[str] | list[tuple[str, int]] | None:
    """Filter HDF5 file scans based on criteria. Returns scan paths or (scan, point) tuples."""
    if not isinstance(in_here, h5py.File):
        print("Error: input file not recognized")
        return None

    # Get all possible scan paths
    all_possible = []
    for spec, group in in_here.items():
        for number, scan in group.items():
            all_possible.append(scan.name)

    return_this = []

    if this_level == "scan":
        for scan in all_possible:
            try:
                this_value = None

                # Check scan attributes first
                if in_here[scan].attrs.get(of_this, None) is not None:
                    this_value = in_here[scan].attrs.get(of_this)
                # Check parameter labels
                elif of_this in [lab.decode("utf-8") if isinstance(lab, bytes) else lab for lab in in_here[scan]["param_labs"]]:
                    param_labs = [lab.decode("utf-8") if isinstance(lab, bytes) else lab for lab in in_here[scan]["param_labs"]]
                    this_ind = param_labs.index(of_this)
                    this_value = in_here[scan]["param_data"][this_ind]

                if this_value is None:
                    continue

                # Evaluate the filter condition
                if _evaluate_condition(this_value, such_that):
                    return_this.append(scan)

            except Exception as e:
                print(f"Warning: Error processing scan {scan}: {e}")
                continue

    elif this_level == "point":
        for scan in all_possible:
            try:
                # Check scan attributes first
                if in_here[scan].attrs.get(of_this, None) is not None:
                    this_value = in_here[scan].attrs.get(of_this)
                    if _evaluate_condition(this_value, such_that):
                        for i in range(len(in_here[scan]["point_data"])):
                            return_this.append((scan, i))

                # Check point labels
                elif of_this in [lab.decode("utf-8") if isinstance(lab, bytes) else lab for lab in in_here[scan]["point_labs"]]:
                    point_labs = [lab.decode("utf-8") if isinstance(lab, bytes) else lab for lab in in_here[scan]["point_labs"]]
                    this_ind = point_labs.index(of_this)
                    for i in range(len(in_here[scan]["point_data"])):
                        this_value = in_here[scan]["point_data"][i][this_ind]
                        if _evaluate_condition(this_value, such_that):
                            return_this.append((scan, i))

                # Check param labels for point-level filtering
                elif of_this in [lab.decode("utf-8") if isinstance(lab, bytes) else lab for lab in in_here[scan]["param_labs"]]:
                    param_labs = [lab.decode("utf-8") if isinstance(lab, bytes) else lab for lab in in_here[scan]["param_labs"]]
                    this_ind = param_labs.index(of_this)
                    this_value = in_here[scan]["param_data"][this_ind]
                    if _evaluate_condition(this_value, such_that):
                        for i in range(len(in_here[scan]["point_data"])):
                            return_this.append((scan, i))

            except Exception as e:
                print(f"Warning: Error processing scan {scan}: {e}")
                continue
    else:
        print("Error: unrecognized filter level")
        return None

    return return_this


def _evaluate_condition(this_value, such_that: str) -> bool:
    """Helper function to safely evaluate filter conditions."""
    try:
        if isinstance(this_value, (str, bytes)):
            # Handle both str and bytes for Python 2/3 compatibility
            if isinstance(this_value, bytes):
                this_value = this_value.decode("utf-8")
            try:
                return bool(eval(f'"{this_value}" {such_that}'))
            except SyntaxError:
                # Handle compound expressions by wrapping in parentheses
                return bool(eval(f'("{this_value}" {such_that})'))

        elif isinstance(this_value, (int, float, bool, np.bool_)) or isinstance(this_value, np.integer) or isinstance(this_value, np.floating):
            # Handle all numeric types including numpy types
            numeric_value = float(this_value) if isinstance(this_value, (np.integer, np.floating)) else this_value
            try:
                # Create a safe evaluation environment
                safe_globals = {"__builtins__": {}}
                safe_locals = {"value": numeric_value}
                return bool(eval(f"value {such_that}", safe_globals, safe_locals))
            except (SyntaxError, NameError):
                # Fallback to original method for simple expressions
                return bool(eval(f"{numeric_value} {such_that}"))

        elif isinstance(this_value, np.ndarray):
            # Handle numpy arrays - convert to scalar if single element
            if this_value.size == 1:
                scalar_value = this_value.item()
                return _evaluate_condition(scalar_value, such_that)
            else:
                print(f"Warning: Cannot filter on multi-element array: {this_value}")
                return False

        else:
            print(f"Warning: Unrecognized type for filtering: {type(this_value)}")
            print(f"Value: {this_value}")
            return False

    except Exception as e:
        print(f"Warning: Error evaluating condition '{such_that}' for value '{this_value}': {e}")
        return False
