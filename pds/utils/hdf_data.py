import h5py
import numpy

from pds.utils.converters import bytes_to_str
from pds.utils.file_locker import FileLock
from pds.utils.image_data import correct_image, read_pixel_map

# General keys mapping - excludes position/scaler values that vary by scan type
GEN_KEYS = {
    "chi": ["angle_values", 0],
    "del": ["angle_values", 1],
    "eta": ["angle_values", 2],
    "mu": ["angle_values", 3],
    "nu": ["angle_values", 4],
    "phi": ["angle_values", 5],
    "real_a": ["lattice_values", 0],
    "real_b": ["lattice_values", 1],
    "real_c": ["lattice_values", 2],
    "real_alpha": ["lattice_values", 3],
    "real_beta": ["lattice_values", 4],
    "real_gamma": ["lattice_values", 5],
    "recip_a": ["lattice_values", 6],
    "recip_b": ["lattice_values", 7],
    "recip_c": ["lattice_values", 8],
    "recip_alpha": ["lattice_values", 9],
    "recip_beta": ["lattice_values", 10],
    "recip_gamma": ["lattice_values", 11],
    "lambda": ["lattice_values", 12],
    "or0_h": ["or_values", 0],
    "or0_k": ["or_values", 1],
    "or0_L": ["or_values", 2],
    "or0_del": ["or_values", 3],
    "or0_eta": ["or_values", 4],
    "or0_chi": ["or_values", 5],
    "or0_phi": ["or_values", 6],
    "or0_nu": ["or_values", 7],
    "or0_mu": ["or_values", 8],
    "or0_lambda": ["or_values", 9],
    "or1_h": ["or_values", 10],
    "or1_k": ["or_values", 11],
    "or1_L": ["or_values", 12],
    "or1_del": ["or_values", 13],
    "or1_eta": ["or_values", 14],
    "or1_chi": ["or_values", 15],
    "or1_phi": ["or_values", 16],
    "or1_nu": ["or_values", 17],
    "or1_mu": ["or_values", 18],
    "or1_lambda": ["or_values", 19],
}

ATT_KEYS = ["date_stamp", "energy", "geom", "hist", "info", "name", "type"]

MISC_KEYS = ["Q", "haz"]

# Detector keys - excludes image_data and corrected_image to prevent overwriting
DET_KEYS = {
    "bad_pixel_map": ["det_%i/corr_values.%i", 0],
    "bad_point": ["det_%i/corr_values.%i", 1],
    "image_changed": ["det_%i/corr_values.%i", 2],
    "image_max": ["det_%i/corr_values.%i", 3],
    "pixel_map_changed": ["det_%i/corr_values.%i", 4],
    "real_image_max": ["det_%i/corr_values.%i", 5],
    "rotangle": ["det_%i/corr_values.%i", 6],
    "sample_angles": ["det_%i/corr_values.%i", 7],
    "sample_diameter": ["det_%i/corr_values.%i", 8],
    "sample_polygon": ["det_%i/corr_values.%i", 9],
    "scale": ["det_%i/corr_values.%i", 10],
    "beam_slits": ["det_%i/det_values.%i", 0],
    "det_slits": ["det_%i/det_values.%i", 1],
    "bgrflag": ["det_%i/int_values.%i", 0],
    "cnbgr": ["det_%i/int_values.%i", 1],
    "compress": ["det_%i/int_values.%i", 2],
    "cpow": ["det_%i/int_values.%i", 3],
    "ctan": ["det_%i/int_values.%i", 4],
    "cwidth": ["det_%i/int_values.%i", 5],
    "filter": ["det_%i/int_values.%i", 6],
    "integrated": ["det_%i/int_values.%i", 7],
    "nline": ["det_%i/int_values.%i", 8],
    "rnbgr": ["det_%i/int_values.%i", 9],
    "roi": ["det_%i/int_values.%i", 10],
    "rpow": ["det_%i/int_values.%i", 11],
    "rtan": ["det_%i/int_values.%i", 12],
    "rwidth": ["det_%i/int_values.%i", 13],
    "alpha": ["det_%i/result_values.%i", 0],
    "beta": ["det_%i/result_values.%i", 1],
    "ctot": ["det_%i/result_values.%i", 2],
    "F": ["det_%i/result_values.%i", 3],
    "F_changed": ["det_%i/result_values.%i", 4],
    "Ferr": ["det_%i/result_values.%i", 5],
    "I": ["det_%i/result_values.%i", 6],
    "I_c": ["det_%i/result_values.%i", 7],
    "I_r": ["det_%i/result_values.%i", 8],
    "Ibgr": ["det_%i/result_values.%i", 9],
    "Ibgr_c": ["det_%i/result_values.%i", 10],
    "Ibgr_r": ["det_%i/result_values.%i", 11],
    "Ierr": ["det_%i/result_values.%i", 12],
    "Ierr_c": ["det_%i/result_values.%i", 13],
    "Ierr_r": ["det_%i/result_values.%i", 14],
}

DET_ATT_KEYS = ["data", "name"]

VERSIONED_KEYS = []


class HdfDataFile:
    """Container for data stored in HDF files. Provides read/write access to structured data points."""

    def __init__(self, fname: str):
        self.fname = fname
        self.point = 0
        self.point_dict = {}
        self.version = 1
        self.file = None
        self.all_items = None

        self.lock_file = FileLock(self.fname)
        print("Attempting to lock file...")
        self.lock_file.acquire()
        print("Lock acquired")
        try:
            self.file = h5py.File(self.fname, "r+")
        except:
            print("Error: unable to open file")
            raise
        self.all_items = self.file.items()

    def __getitem__(self, arg: int) -> dict:
        """Read point data and return dictionary. Usage: hdf_object[point][arg]."""
        if arg == self.point:
            return self.point_dict
        if self.point != 0 and self.point_dict != {}:
            self.write_point(self.point_dict, self.point)
        self.read_point(arg)
        return self.point_dict

    def close(self) -> None:
        """Close the file after writing current point data. Releases file lock."""

        try:
            if self.point != 0 and self.point_dict != {}:
                self.write_point(self.point_dict, self.point)
        except ValueError:
            print("Error writing point; file may already be closed")

        try:
            self.point = 0
            self.point_dict = {}
            self.file.flush()
            self.file.close()
            self.lock_file.release()
            print("Lock released")

        except Exception:
            print("Error: file may not have closed cleanly,")
            print("though it may have already been closed.")
        try:
            self.lock_file.release()  # Retry lock release if file close failed
        except Exception:
            pass

    def delete(self, item: str) -> None:
        """Delete a point from the file."""

        del self.file[item]

    def get(self, num: int, default: dict = None) -> dict:
        """Return point dictionary if exists, otherwise return default. Works like dict.get()."""
        try:
            return self.__getitem__(num)
        except KeyError:
            return default
        except:
            raise

    def get_all(self, key: str or tuple, points: list = None) -> dict:
        """Get key value for all points. Use tuple for nested keys like ('det_0', 'image_data')."""

        all_results = {}

        if points is None:
            points = []
            for item in self.all_items:
                points.append(item[0])
        if isinstance(key, (str, bytes)):
            if key in GEN_KEYS:
                key_loc = GEN_KEYS[key]
                for point in points:
                    all_results[point] = self.file[point][key_loc[0]][key_loc[1]]
                if self.point in points:
                    all_results[self.point] = self.point_dict[key]
            elif key in ATT_KEYS:
                if key.startswith("hist"):
                    key = key + "." + str(self.version)
                for point in points:
                    all_results[point] = self.file[point].attrs[key]
                if self.point in points:
                    if key.startswith("hist"):
                        all_results[self.point] = self.point_dict["hist"]
                    else:
                        all_results[self.point] = self.point_dict[key]
            elif key in MISC_KEYS:
                for point in points:
                    all_results[point] = self.file[point][key]
                if self.point in points:
                    all_results[self.point] = self.point_dict[key]
            else:
                for point in points:
                    # Handle key search in labels with proper bytes/string conversion
                    if self._key_in_labels(key, self.file[point]["position_labels"]):
                        key_loc = self._find_key_in_labels(key, self.file[point]["position_labels"])
                        all_results[point] = self.file[point]["position_values"][key_loc]
                    elif self._key_in_labels(key, self.file[point]["scaler_labels"]):
                        key_loc = self._find_key_in_labels(key, self.file[point]["scaler_labels"])
                        all_results[point] = self.file[point]["scaler_values"][key_loc]
                    else:
                        print("Position/Scaler Labels Unrecognized Key Error: ", key)
                if self.point in points and key in self.point_dict.keys():
                    all_results[self.point] = self.point_dict[key]
        elif isinstance(key, tuple):
            det_name = key[0]
            key = key[1]
            if key in DET_KEYS:
                key_loc = DET_KEYS[key]
                key_loc_path = key_loc[0].split("/")[1] % self.version
                for point in points:
                    try:
                        all_results[point] = self.file[point][det_name][key_loc_path][key_loc[1]]
                    except (OSError, IOError) as e:
                        print(f"Error reading {key} for point {point}: {e}")
                        # Try to use a default value or skip this point
                        all_results[point] = None
                if self.point in points:
                    all_results[self.point] = self.point_dict[det_name][key]
            elif key in DET_ATT_KEYS:
                for point in points:
                    try:
                        all_results[point] = self.file[point][det_name].attrs[key]
                    except (OSError, IOError) as e:
                        print(f"Error reading attribute {key} for point {point}: {e}")
                        all_results[point] = None
                if self.point in points:
                    all_results[self.point] = self.point_dict[det_name][key]
            elif key.startswith("image_data"):
                for point in points:
                    try:
                        all_results[point] = self.file[point][det_name][key]
                    except Exception:
                        pass
            elif key.startswith("corrected_image"):
                for point in points:
                    try:
                        point_image = numpy.array(self.file[point][det_name]["image_data"])
                        bpm_loc = DET_KEYS["bad_pixel_map"]
                        current_corr = bpm_loc[0].split("/")[1] % self.version
                        point_mask = str(self.file[point][det_name][current_corr][bpm_loc[1]])
                        if not point_mask.startswith("(") and not point_mask.startswith("["):
                            point_mask = str(read_pixel_map(point_mask))
                        all_results[point] = correct_image(point_image, point_mask)
                    except Exception:
                        pass
                if self.point in points:
                    all_results[self.point] = self.point_dict[det_name][key]
            else:
                print("Error: unrecognized key")
        else:
            print("Error: unknown key type")
        return all_results

    def read_point(self, num: str) -> None:
        """Read data from point into dictionary. Num should be full serial string like '000328'."""
        self.point = num
        self.point_dict = {}
        for key in GEN_KEYS:
            key_loc = GEN_KEYS[key]
            self.point_dict[key] = self.file[num][key_loc[0]][key_loc[1]]
        for key in self.file[num]["position_labels"]:
            key_loc = list(self.file[num]["position_labels"]).index(key)
            dict_key = bytes_to_str(key)
            self.point_dict[dict_key] = self.file[num]["position_values"][key_loc]
            if isinstance(key, str):
                self.point_dict[key.encode("utf-8")] = self.file[num]["position_values"][key_loc]
            else:
                self.point_dict[key] = self.file[num]["position_values"][key_loc]
        for key in self.file[num]["scaler_labels"]:
            key_loc = list(self.file[num]["scaler_labels"]).index(key)
            dict_key = bytes_to_str(key)
            self.point_dict[dict_key] = self.file[num]["scaler_values"][key_loc]
            if isinstance(key, str):
                self.point_dict[key.encode("utf-8")] = self.file[num]["scaler_values"][key_loc]
            else:
                self.point_dict[key] = self.file[num]["scaler_values"][key_loc]
        for key in ATT_KEYS:
            if key.startswith("hist"):
                key = key + "." + str(self.version)
                self.point_dict["hist"] = self.file[num].attrs[key]
            else:
                self.point_dict[key] = self.file[num].attrs[key]
        for key in MISC_KEYS:
            self.point_dict[key] = self.file[num][key]
        current_det_num = 0
        while True:
            det_str = "det_%i" % current_det_num
            if det_str not in self.file[num]:
                break
            self.point_dict[det_str] = {}
            for key in DET_KEYS:
                try:
                    key_loc = DET_KEYS[key]
                    key_loc_path = key_loc[0] % (current_det_num, self.version)
                    self.point_dict[det_str][key] = self.file[num][key_loc_path][key_loc[1]]
                except (KeyError, OSError, IOError):
                    pass
            for key in DET_ATT_KEYS:
                try:
                    self.point_dict[det_str][key] = self.file[num][det_str].attrs[key]
                except (KeyError, OSError, IOError):
                    pass
            try:
                point_image = numpy.array(self.file[num][det_str]["image_data"])
                self.point_dict[det_str]["image_data"] = point_image
                point_mask = bytes_to_str(self.point_dict[det_str]["bad_pixel_map"])
                if not point_mask.startswith("(") and not point_mask.startswith("["):
                    point_mask = str(read_pixel_map(point_mask))
                    self.point_dict[det_str]["bad_pixel_map"] = point_mask
                corrected_image = correct_image(point_image, point_mask)
                self.point_dict[det_str]["corrected_image"] = corrected_image
            except Exception:
                pass
            current_det_num += 1

    def set_all(self, key: str or tuple, value: any, points: list = None) -> None:
        """Set key value for all points. Use tuple for nested keys like ('det_0', 'bad_pixel_map')."""

        if points is None:
            points = []
            for item in self.all_items:
                points.append(item[0])

        # Handle both string and bytes keys - match original Python 2 behavior
        if isinstance(key, (str, bytes)):
            # Convert bytes to string using bytes_to_str for consistency
            str_key = bytes_to_str(key) if isinstance(key, bytes) else key

            if str_key in GEN_KEYS:
                if self.point in points:
                    self.point_dict[str_key] = value
                key_loc = GEN_KEYS[str_key]
                for point in points:
                    self.file[point][key_loc[0]][key_loc[1]] = value
            elif self.point != 0 and self._key_in_labels(str_key, self.file[self.point]["position_labels"]):
                if self.point in points:
                    self.point_dict[str_key] = value
                key_loc = self._find_key_in_labels(str_key, self.file[self.point]["position_labels"])
                for point in points:
                    self.file[point]["position_values"][key_loc] = value
            elif self.point != 0 and self._key_in_labels(str_key, self.file[self.point]["scaler_labels"]):
                if self.point in points:
                    self.point_dict[str_key] = value
                key_loc = self._find_key_in_labels(str_key, self.file[self.point]["scaler_labels"])
                for point in points:
                    self.file[point]["scaler_values"][key_loc] = value
            elif str_key in ATT_KEYS:
                if self.point in points:
                    self.point_dict[str_key] = value
                if str_key.startswith("hist"):
                    str_key = str_key + "." + str(self.version)
                for point in points:
                    self.file[point].attrs[str_key] = value
            elif str_key in MISC_KEYS:
                if self.point in points:
                    self.point_dict[str_key] = value
                for point in points:
                    self.file[point][str_key] = value
            else:
                print("Error: unrecognized key")
        elif isinstance(key, tuple):
            det_name = key[0]
            key = key[1]
            if key in DET_KEYS:
                if self.point in points:
                    self.point_dict[det_name][key] = value
                key_loc = DET_KEYS[key]
                key_loc_path = key_loc[0].split("/")[1] % self.version
                for point in points:
                    try:
                        self.file[point][det_name][key_loc_path][key_loc[1]] = value
                    except (IOError, TypeError):
                        # Handle type mismatches - try string conversion first, then float
                        try:
                            self.file[point][det_name][key_loc_path][key_loc[1]] = str(value)
                        except (IOError, TypeError):
                            self.file[point][det_name][key_loc_path][key_loc[1]] = numpy.float(value)
            elif key in DET_ATT_KEYS:
                if self.point in points:
                    self.point_dict[det_name][key] = value
                for point in points:
                    self.file[point][det_name].attrs[key] = value
            elif key.startswith("image_data"):
                print("Are you sure you want to overwrite the image data?")
                print("If so, go into the hdf_data.py file and uncomment the lines following this message.")
            elif key.startswith("corrected_image"):
                pass
            else:
                print("Error: unrecognized key")
        else:
            print("Error: unknown key type")

    def _key_in_labels(self, key: str, labels) -> bool:
        """Check if key exists in labels, handling both string and bytes."""
        try:
            for label in labels:
                if bytes_to_str(label) == key or label == key:
                    return True
            return False
        except Exception:
            return False

    def _find_key_in_labels(self, key: str, labels) -> int:
        """Find index of key in labels, handling both string and bytes."""
        for i, label in enumerate(labels):
            if bytes_to_str(label) == key or label == key:
                return i
        raise ValueError(f"Key {key} not found in labels")

    def version_point(self, data: dict = {}) -> None:
        """Create new version of point with updated data. Data is dictionary with new/updated values."""

    def write_point(self, data: dict, num: str = None) -> None:
        """Write dictionary data to file. Uses current point if num not specified."""
        if num is None:
            num = self.point
        for key in data:
            # Handle both string and bytes keys - match original Python 2 behavior
            original_key = key
            str_key = bytes_to_str(key) if isinstance(key, bytes) else key

            if str_key.startswith("hist"):
                hist_key = str_key + "." + str(self.version)
                # Use 'hist' if available, otherwise use the original key's data
                hist_data = data.get("hist", data.get(original_key))
                self.file[num].attrs[hist_key] = hist_data
            elif str_key.startswith("det_"):
                if original_key in data:
                    det_dict = data[original_key]
                    for det_key in det_dict:
                        try:
                            key_loc = DET_KEYS[det_key]
                            key_loc_path = key_loc[0].split("/")[1] % self.version
                            try:
                                # Try to write the value directly first
                                try:
                                    self.file[num][str_key][key_loc_path][key_loc[1]] = data[original_key][det_key]
                                except TypeError:
                                    # If we get a TypeError about string conversion, handle bytes/string conversion
                                    value_to_write = data[original_key][det_key]
                                    if isinstance(value_to_write, bytes):
                                        value_to_write = bytes_to_str(value_to_write)
                                    else:
                                        value_to_write = str(value_to_write)
                                    self.file[num][str_key][key_loc_path][key_loc[1]] = value_to_write
                            except IOError:
                                self.file[num][str_key][key_loc_path][key_loc[1]] = numpy.float(data[original_key][det_key])
                        except KeyError:
                            pass
            else:
                pass
