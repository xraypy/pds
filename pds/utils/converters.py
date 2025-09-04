from typing import Any

import numpy as np


def bytes_to_str(value: Any) -> Any:
    """Convert bytes to string for Python 3 compatibility with h5py."""
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return value


def safe_eval_hdf(value: Any) -> Any:
    """Safely evaluate HDF object values with proper type conversion for various data types."""
    try:
        # Handle numpy arrays and similar objects
        if hasattr(value, "shape") or hasattr(value, "__array__"):
            return value

        # Handle numeric values directly
        if isinstance(value, (int, float)):
            return value

        # Convert bytes to string first
        str_value = bytes_to_str(value)

        # Handle boolean string values
        if isinstance(str_value, str):
            if str_value.lower() == "true":
                return True
            elif str_value.lower() == "false":
                return False

        # Handle bytes literal strings
        if str_value.startswith("b'") and str_value.endswith("'"):
            bytes_obj = eval(str_value)
            if isinstance(bytes_obj, bytes):
                str_value = bytes_obj.decode("utf-8")
        elif str_value.startswith('b"') and str_value.endswith('"'):
            bytes_obj = eval(str_value)
            if isinstance(bytes_obj, bytes):
                str_value = bytes_obj.decode("utf-8")

        return eval(str_value)
    except Exception as e:
        print(f"Error evaluating HDF value: {value}, error: {e}")
        return value


def extract_value(data: float | list[float] | np.ndarray, point: int) -> float:
    """Extract float value from data at given point index. Handles both scalar and array data consistently."""
    if hasattr(data, "__getitem__") and not isinstance(data, str):
        try:
            return float(data[point])
        except (IndexError, TypeError):
            # For arrays/lists, try to get the first element if point access fails
            try:
                if len(data) > 0:
                    return float(data[0])
                else:
                    raise ValueError("Empty array/list cannot be converted to float")
            except (TypeError, AttributeError):
                # If it's not a container or length is not accessible, try direct conversion
                return float(data)
    else:
        return float(data)
