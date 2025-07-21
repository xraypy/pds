from typing import Any


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
