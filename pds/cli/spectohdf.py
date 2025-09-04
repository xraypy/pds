import math
import time
from pathlib import Path
from typing import Any, Optional

import h5py
import numpy as np
from PIL import Image

from pds.utils.file_locker import FileLock, FileLockException
from pds.utils.converters import bytes_to_str


def summarize(lines: list[str]) -> list[dict[str, Any]]:
    """Parse SPEC file and extract scan data organized by scan number."""

    # G-value parameter labels for SPEC version 6.03.03+
    g_labs = [
        "g_prefer",
        "g_sect",
        "g_frz",
        "g_haz",
        "g_kaz",
        "g_laz",
        "g_zh0",
        "g_zk0",
        "g_z10",
        "g_zh1",
        "g_zk1",
        "g_zl1",
        "g_kappa",
        "g_13",
        "g_14",
        "g_sigtau",
        "g_mode1",
        "g_mode2",
        "g_mode3",
        "g_mode4",
        "g_mode5",
        "g_21",
        "g_new1",
        "g_aa",
        "g_bb",
        "g_cc",
        "g_al",
        "g_be",
        "g_ga",
        "g_aa_s",
        "g_bb_s",
        "g_cc_s",
        "g_al_s",
        "g_be_s",
        "g_ga_s",
        "g_h0",
        "g_k0",
        "g_l0",
        "g_h1",
        "g_k1",
        "g_l1",
        "g_u00",
        "g_u01",
        "g_u02",
        "g_u03",
        "g_u04",
        "g_u05",
        "g_u10",
        "g_u11",
        "g_u12",
        "g_u13",
        "g_u14",
        "g_u15",
        "g_lambda0",
        "g_lambda1",
        "g_new2",
        "g_new3",
        "g_54",
        "g_55",
        "g_56",
        "g_57",
        "g_58",
        "g_59",
        "g_60",
        "g_61",
        "g_62",
        "g_H",
        "g_K",
        "g_L",
        "g_LAMBDA",
        "g_ALPHA",
        "g_BETA",
        "g_OMEGA",
        "g_TTH",
        "g_PSI",
        "g_TAU",
        "g_QAZ",
        "g_NAZ",
        "g_SIGMA_AZ",
        "g_TAU_AZ",
        "g_F_ALPHA",
        "g_F_BETA",
        "g_F_OMEGA",
        "g_F_PSI",
        "g_F_NAZ",
        "g_F_QAZ",
        "g_F_DEL",
        "g_F_ETA",
        "g_F_CHI",
        "g_F_PHI",
        "g_F_NU",
        "g_F_MU",
        "g_F_CHI_Z",
        "g_F_PHI_Z",
        "CUT_DEL",
        "CUT_ETA",
        "CUT_CHI",
        "CUT_PHI",
        "CUT_NU",
        "CUT_MU",
        "CUT_KETA",
        "CUT_KAP",
        "CUT_KPHI",
        "g_100",
        "g_101",
        "g_102",
        "g_103",
        "g_104",
        "g_105",
        "g_106",
        "g_107",
        "g_108",
        "g_109",
        "g_110",
        "g_111",
    ]

    summary = []
    lineno = 0

    # Initialize scan variables
    spec_name = epoch = mnames = cmnd = date = xtime = g_vals = q = p_vals = atten = energy = lab = None
    aborted = False
    point_data = []
    index = ncols = n_sline = 0

    for i, line in enumerate(lines):
        lineno = i + 1
        line = line.rstrip("\n\r")

        # Parse different SPEC file sections
        if line.startswith("#F"):
            spec_name = Path(line[3:]).name
        elif line.startswith("#E "):
            epoch = int(line[3:])
        elif line.startswith("#O"):
            if line[2] == "0":
                mnames = ""
            mnames = (mnames or "") + line[3:]
        elif line.startswith("#S "):
            parts = line[3:].split()
            index = int(parts[0])
            cmnd = line[4 + len(parts[0]) :]
            n_sline = lineno
        elif line.startswith("#D "):
            date = line[3:]
        elif line.startswith("#T "):
            xtime = line[3:]
        elif line.startswith("#G"):
            if line[2] == "0":
                g_vals = ""
            g_vals = (g_vals or "") + line[3:]
        elif line.startswith("#Q "):
            q = line[3:]
        elif line.startswith("#P"):
            if line[2] == "0":
                p_vals = ""
            p_vals = (p_vals or "") + line[3:]
        elif line.startswith("#N "):
            ncols = int(line[3:])
        elif line.startswith("#AT"):
            atten = line[6:].strip()
        elif line.startswith("#EN"):
            energy = float(line[8:])
        elif line.startswith("#L "):
            lab = line[3:].split()

            # Parse data points and check for aborted scans
            remaining_lines = lines[lineno:]
            nl_dat = 0
            aborted = False

            for data_line in remaining_lines:
                if data_line.startswith("#S "):
                    break
                elif data_line.startswith("#"):
                    if "aborted" in data_line:
                        aborted = True
                elif len(data_line.strip()) > 0:
                    nl_dat += 1
                    try:
                        point_data.append(list(map(float, data_line.split())))
                    except ValueError:
                        aborted = True
                        break

            # Calculate L values if available
            L_pos = -1
            L_start = L_stop = "--"
            try:
                L_pos = lab.index("L")
                if L_pos >= 0 and point_data:
                    L_start = point_data[0][L_pos]
                    L_stop = point_data[-1][L_pos]
            except (ValueError, IndexError):
                pass

            # Create scan dictionary
            scan_data = {
                "index": index,
                "spec_name": spec_name,
                "init_epoch": epoch,
                "nl_start": n_sline,
                "cmd": cmnd,
                "date": date,
                "time": xtime,
                "mnames": (mnames or "").split(),
                "P": list(map(float, (p_vals or "").split())) if p_vals else [],
                "g_labs": g_labs,
                "G": list(map(float, (g_vals or "").split())) if g_vals else [],
                "Q": q,
                "ncols": ncols,
                "labels": lab,
                "atten": atten,
                "energy": energy,
                "lineno": lineno,
                "aborted": aborted,
                "point_data": point_data,
                "real_L_start": L_start,
                "real_L_stop": L_stop,
                "nl_dat": nl_dat,
            }

            summary.append(scan_data)

            # Reset for next scan
            cmnd = date = xtime = g_vals = q = p_vals = atten = energy = lab = None
            aborted = False
            point_data = []
            index = ncols = n_sline = 0

    return summary


def read_image(image_path: Path) -> Optional[np.ndarray]:
    """Read and return image data as numpy array."""
    try:
        with Image.open(image_path) as im:
            if im.mode == "I":
                arr = np.frombuffer(im.tobytes(), dtype=np.int32)
            elif im.mode == "I;16":
                arr = np.frombuffer(im.tobytes(), dtype=np.int16)
            else:
                print(f"Warning: Unsupported image mode '{im.mode}' for {image_path}")
                return None
            return arr.reshape((im.size[1], im.size[0]))
    except Exception as e:
        print(f"Error reading image {image_path}: {e}")
        return None


def process_scan_attributes(scan_group: h5py.Group, scan: dict[str, Any]) -> None:
    """Process and set scan-specific attributes based on scan type."""

    cmd_parts = scan.get("cmd", "").split()
    if not cmd_parts:
        return

    scan_type = cmd_parts[0]
    scan_group.attrs["cmd"] = scan["cmd"]
    scan_group.attrs["s_type"] = scan_type

    # Parse different scan types
    if scan_type == "a2scan" and len(cmd_parts) >= 9:
        scan_group.attrs["motor1"] = cmd_parts[1]
        scan_group.attrs["m1_start"] = float(cmd_parts[2])
        scan_group.attrs["m1_stop"] = float(cmd_parts[3])
        scan_group.attrs["motor2"] = cmd_parts[4]
        scan_group.attrs["m2_start"] = float(cmd_parts[5])
        scan_group.attrs["m2_stop"] = float(cmd_parts[6])
        scan_group.attrs["num_points"] = float(cmd_parts[7])
        scan_group.attrs["count_time"] = float(cmd_parts[8])

    elif scan_type == "a4scan" and len(cmd_parts) >= 15:
        scan_group.attrs["motor1"] = cmd_parts[1]
        scan_group.attrs["m1_start"] = float(cmd_parts[2])
        scan_group.attrs["m1_stop"] = float(cmd_parts[3])
        scan_group.attrs["motor2"] = cmd_parts[4]
        scan_group.attrs["m2_start"] = float(cmd_parts[5])
        scan_group.attrs["m2_stop"] = float(cmd_parts[6])
        scan_group.attrs["motor3"] = cmd_parts[7]
        scan_group.attrs["m3_start"] = float(cmd_parts[8])
        scan_group.attrs["m3_stop"] = float(cmd_parts[9])
        scan_group.attrs["motor4"] = cmd_parts[10]
        scan_group.attrs["m4_start"] = float(cmd_parts[11])
        scan_group.attrs["m4_stop"] = float(cmd_parts[12])
        scan_group.attrs["num_points"] = float(cmd_parts[13])
        scan_group.attrs["count_time"] = float(cmd_parts[14])

    elif scan_type == "ascan" and len(cmd_parts) >= 6:
        scan_group.attrs["motor1"] = cmd_parts[1]
        scan_group.attrs["m1_start"] = float(cmd_parts[2])
        scan_group.attrs["m1_stop"] = float(cmd_parts[3])
        scan_group.attrs["num_points"] = float(cmd_parts[4])
        scan_group.attrs["count_time"] = float(cmd_parts[5])

    elif scan_type == "Escan" and len(cmd_parts) >= 5:
        scan_group.attrs["energy_start"] = float(cmd_parts[1])
        scan_group.attrs["energy_stop"] = float(cmd_parts[2])
        scan_group.attrs["num_points"] = float(cmd_parts[3])
        scan_group.attrs["count_time"] = float(cmd_parts[4])

        # Parse Q values for energy scans
        if scan.get("Q"):
            q_parts = scan["Q"].split()
            if len(q_parts) >= 2:
                scan_group.attrs["h_val"] = round(float(q_parts[0]), 1)
                scan_group.attrs["k_val"] = round(float(q_parts[1]), 1)

    elif scan_type == "hklscan" and len(cmd_parts) >= 9:
        scan_group.attrs["h_val"] = round(float(cmd_parts[1]), 1)
        scan_group.attrs["h_start"] = float(cmd_parts[1])
        scan_group.attrs["h_stop"] = float(cmd_parts[2])
        scan_group.attrs["k_val"] = round(float(cmd_parts[3]), 1)
        scan_group.attrs["k_start"] = float(cmd_parts[3])
        scan_group.attrs["k_stop"] = float(cmd_parts[4])
        scan_group.attrs["L_start"] = float(cmd_parts[5])
        scan_group.attrs["L_stop"] = float(cmd_parts[6])
        scan_group.attrs["num_points"] = float(cmd_parts[7])
        scan_group.attrs["count_time"] = float(cmd_parts[8])

    elif scan_type == "rodscan" and len(cmd_parts) >= 9:
        scan_group.attrs["h_val"] = float(cmd_parts[1])
        scan_group.attrs["k_val"] = float(cmd_parts[2])
        scan_group.attrs["L_start"] = float(cmd_parts[3])
        scan_group.attrs["L_stop"] = float(cmd_parts[4])
        scan_group.attrs["L_space"] = float(cmd_parts[5])
        scan_group.attrs["max_time"] = float(cmd_parts[6])
        scan_group.attrs["peak_pos"] = float(cmd_parts[7])
        scan_group.attrs["peak_space"] = float(cmd_parts[8])

        # Calculate HK distance for rodscan
        h_val = scan_group.attrs["h_val"]
        k_val = scan_group.attrs["k_val"]

        try:
            param_labs_list = [bytes_to_str(x) for x in list(scan_group["param_labs"])]
            h_val_d = float(list(scan_group["param_data"])[param_labs_list.index("g_aa_s")])
            k_val_d = float(list(scan_group["param_data"])[param_labs_list.index("g_bb_s")])
            t_val_d = float(list(scan_group["param_data"])[param_labs_list.index("g_ga_s")])
            t_val_d = math.radians(180 - t_val_d)

            hk_dist = round(((h_val * h_val_d) ** 2 + (k_val * k_val_d) ** 2 - 2 * h_val * h_val_d * k_val * k_val_d * math.cos(t_val_d)) ** 0.5, 8)
            scan_group.attrs["hk_dist"] = hk_dist
        except (ValueError, IndexError, KeyError):
            pass  # Skip HK distance calculation if parameters are missing

    elif scan_type == "timescan" and len(cmd_parts) >= 3:
        scan_group.attrs["count_time"] = float(cmd_parts[1])
        scan_group.attrs["time_space"] = float(cmd_parts[2])


def spec_to_hdf5(spec_file: Path, output_file: Path, mode: str = "a", image_dir: Optional[Path] = None, verbose: bool = True) -> bool:
    """Convert SPEC file to HDF5 format."""
    if not spec_file.exists():
        print(f"Error: SPEC file not found: {spec_file}")
        return False

    start_time = time.time()

    # Determine image directory
    if image_dir is None:
        spec_dir = spec_file.parent
        spec_name_base = spec_file.stem
        if spec_name_base.endswith(".spc"):
            spec_name_base = spec_name_base[:-4]
        image_dir = spec_dir / "images" / spec_name_base

    # Read and parse SPEC file
    try:
        with open(spec_file, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
    except Exception as e:
        print(f"Error reading SPEC file: {e}")
        return False

    if verbose:
        print(f"Parsing SPEC file: {spec_file}")

    summary = summarize(lines)
    if not summary:
        print("Warning: No scans found in SPEC file")
        return False

    # Acquire file lock
    lock = FileLock(str(output_file), timeout=10)
    try:
        if verbose:
            print("Acquiring file lock...")
        lock.acquire()
        if verbose:
            print("Lock acquired")
    except FileLockException as e:
        print(f"Error acquiring lock: {e}")
        return False

    try:
        # Open HDF5 file
        with h5py.File(output_file, mode) as master_file:
            spec_group = master_file.require_group(spec_file.name)

            for scan in summary:
                scan_index = str(scan["index"])
                scan_group = spec_group.require_group(scan_index)

                # Skip if scan already complete
                if "point_data" in scan_group and len(scan_group["point_data"]) == scan["nl_dat"]:
                    if verbose:
                        print(f"Skipping scan {scan_index} (already complete)")
                    continue

                if verbose:
                    print(f"Processing scan {scan_index}")

                # Clean up existing datasets
                for dataset_name in ["point_data", "point_labs", "param_labs", "param_data"]:
                    if dataset_name in scan_group:
                        del scan_group[dataset_name]

                # Create datasets
                scan_group.create_dataset("point_data", data=scan["point_data"])

                # Ensure string datasets are properly encoded for h5py compatibility
                scan_group.create_dataset("point_labs", data=scan["labels"], dtype=h5py.string_dtype(encoding="utf-8"))
                scan_group.create_dataset("param_labs", data=list(scan["g_labs"] + scan["mnames"]), dtype=h5py.string_dtype(encoding="utf-8"))
                scan_group.create_dataset("param_data", data=list(scan["G"]) + list(scan["P"]))

                # Set scan attributes
                for key, value in scan.items():
                    if key not in ["labels", "point_data", "g_labs", "mnames", "G", "P"]:
                        if value is None:
                            value = "None"
                        # Handle bytes/string conversion for attributes
                        if isinstance(value, bytes):
                            value = value.decode("utf-8", errors="replace")
                        scan_group.attrs[key] = value

                # Process scan-specific attributes
                process_scan_attributes(scan_group, scan)

                # Process image data if directory exists
                scan_image_dir = image_dir / f"S{scan['index']:03d}"
                if scan_image_dir.exists():
                    if verbose:
                        print(f"Processing images in: {scan_image_dir}")

                    image_data = []
                    image_files = sorted([f for f in scan_image_dir.iterdir() if f.suffix.lower() == ".tif"])

                    for image_file in image_files:
                        image_array = read_image(image_file)
                        if image_array is not None:
                            image_data.append(image_array)
                        elif image_data:  # Fill with -1 if read fails but we have previous data
                            placeholder = np.full_like(image_data[-1], -1)
                            image_data.append(placeholder)

                    if image_data:
                        if "image_data" in scan_group:
                            del scan_group["image_data"]
                        scan_group.create_dataset("image_data", data=image_data, compression="szip")

    except Exception as e:
        print(f"Error during HDF5 conversion: {e}")
        return False

    finally:
        lock.release()
        if verbose:
            print("Lock released")

    elapsed_time = (time.time() - start_time) / 60
    if verbose:
        print(f"Conversion completed in {elapsed_time:.2f} minutes")

    return True
