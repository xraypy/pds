import re
from typing import Any

import h5py
import numpy as np

from pds.utils.converters import bytes_to_str

INTEGRATION_PARAMETERS = {
    "bgrflag": 1,
    "cnbgr": 5,
    "colormap": "None",
    "compress": 1,
    "cpow": 2,
    "ctan": False,
    "cwidth": 15,
    "filter": False,
    "geom": "psic",
    "integrated": False,
    "nline": 1,
    "rnbgr": 5,
    "roi": [],
    "rpow": 0,
    "rtan": False,
    "rwidth": 15,
}
CORRECTION_PARAMETERS = {
    "bad_pixel_map": [],
    "bad_point": False,
    "image_changed": True,
    "image_max": -1,
    "pixel_map_changed": True,
    "real_image_max": -1,
    "rotangle": 0,
    "sample_angles": [],
    "sample_diameter": 10,
    "sample_polygon": [],
    "scale": 1.0e6,
}
DETECTOR_PARAMETERS = {"beam_slits": {}, "det_slits": {}, "name": "pilatus"}


def read_pixel_map(fname: str) -> tuple[list[list[int]], list[list[int]]] | list[Any]:
    """Read the pixel map file assuming (bad pixel) (good pixel) format."""
    bad_pixels = []
    good_pixels = []
    try:
        bad = []
        good = []
        f = open(fname)
        for line in f.readlines():
            tmp = line.strip().split()
            bad.append(tmp[0])
            if len(tmp) > 1:
                good.append(tmp[1])
        f.close()
        for p in bad:
            pp = p.split(",")
            bad_pixels.append(list(map(int, pp)))
        for p in good:
            pp = p.split(",")
            good_pixels.append(list(map(int, pp)))
        return bad_pixels, good_pixels
    except Exception:
        print("Error reading file: %s" % fname)
        return []


def master_to_project(master_file: str, desired_scans: dict[str, dict[str, dict[str, Any]]], project_file: str, append: bool = True, gui: bool = False) -> None:
    """Convert scans from a master file to a project file. Handles both appending and overwriting."""

    read_this = h5py.File(master_file, "r")
    # Data type for variable-length string arrays
    var_len_strs = h5py.vlen_dtype(str)

    if append:
        # Append mode: add new points without overwriting, may overwrite past 999999 points.
        write_this = h5py.File(project_file, "a")
        this_items = list(write_this.items())
        point_counter = int(this_items[-1][0]) + 1
        all_names = {}
        for item in this_items:
            all_names[item[1].attrs.get("name")] = item[1]
    else:
        # Write mode: overwrite existing data, reset naming counter.
        write_this = h5py.File(project_file, "w")
        point_counter = 1
        all_names = {}

    # Ensure sequentially numbered points from 1.

    progress_continue = True
    if gui:
        key_count = 0
        project_progress = 0
        for spec_name in desired_scans.keys():
            for scan_number in desired_scans[spec_name].keys():
                key_count += len(read_this[spec_name][scan_number]["point_data"])
        import wx

        progress_box = wx.ProgressDialog(
            "Building file...", "Progress:", key_count, style=wx.PD_CAN_ABORT | wx.PD_APP_MODAL | wx.PD_AUTO_HIDE | wx.PD_ELAPSED_TIME | wx.PD_REMAINING_TIME
        )

    try:
        for spec_name in desired_scans.keys():
            if not progress_continue:
                break
            for scan_number in desired_scans[spec_name].keys():
                if not progress_continue:
                    break
                read_head = read_this[spec_name][scan_number]
                num_points = len(read_head["point_data"])
                specified_attrs = desired_scans[spec_name][scan_number]
                file_epoch = read_head.attrs.get("init_epoch", 0)

                point_labs_list = [bytes_to_str(x) for x in read_head["point_labs"]]
                epoch_loc = point_labs_list.index("Epoch")

                param_labs_list = [bytes_to_str(x) for x in read_head["param_labs"]]

                # add_time = 0
                for i in range(num_points):
                    # Epoch offset of the point
                    point_epoch = read_head["point_data"][i][epoch_loc]

                    uniq_name = str(spec_name + ":S" + scan_number + ":P" + str(i + 1) + "/" + str(num_points) + ":" + str(point_epoch + file_epoch))
                    uniq_pattern = str(spec_name + ":S" + scan_number + ":P" + str(i + 1) + "/[0-9]+:" + str(point_epoch + file_epoch))

                    matches = [word for word in all_names.keys() if re.match(uniq_pattern, word)]
                    if matches:
                        all_names[matches[0]].attrs["name"] = uniq_name
                        if gui:
                            print(matches[0], " already in ", project_file)
                            if matches[0] != uniq_name:
                                print("Renamed to " + uniq_name)
                            project_progress += 1
                            progress_continue, holding = progress_box.Update(project_progress)
                            wx.GetApp().Yield(True)
                        continue

                    point_name = "%6.6i" % point_counter
                    point_group = write_this.create_group(point_name)
                    point_counter += 1

                    # Unique identifier
                    point_group.attrs["name"] = uniq_name
                    all_names[uniq_name] = point_group
                    # Scan type
                    point_group.attrs["type"] = read_head.attrs.get("s_type", "NA")
                    # Point info initialized to the spec command
                    point_group.attrs["info"] = read_head.attrs.get("cmd", "NA")
                    # Geometry of the point
                    point_group.attrs["geom"] = specified_attrs.get("geom", INTEGRATION_PARAMETERS["geom"])
                    # Comment field initialized to 'Initialized'
                    point_group.attrs["hist.1"] = "Initialized"
                    # Time the point was taken as epoch.
                    point_group.attrs["date_stamp"] = point_epoch + file_epoch
                    # Energy
                    point_group.attrs["energy"] = read_head.attrs.get("energy", "NA")

                    # Angles
                    ang_labels = ["chi", "del", "eta", "mu", "nu", "phi"]
                    point_group.create_dataset("angle_labels", data=ang_labels)

                    pang_labels = ["chi", "TwoTheta", "theta", "Psi", "Nu", "phi"]
                    ang_values = []
                    for j in range(6):
                        ang_lbl = ang_labels[j]
                        pang_lbl = pang_labels[j]

                        if ang_lbl in point_labs_list:
                            ang_pos = list(point_labs_list).index(ang_lbl)
                            ang_val = read_head["point_data"][i][ang_pos]
                        elif pang_lbl in point_labs_list:
                            ang_pos = list(point_labs_list).index(pang_lbl)
                            ang_val = read_head["point_data"][i][ang_pos]
                        else:
                            ang_pos = list(param_labs_list).index(pang_lbl)
                            ang_val = read_head["param_data"][ang_pos]
                        ang_values.append(ang_val)
                    point_group.create_dataset("angle_values", data=ang_values)

                    # Real-space lattice followed by recip-space lattice
                    lattice_start = list(param_labs_list).index("g_aa")
                    # lattice_stop = list(param_labs_list).index('g_ga_s')
                    ltc_lbls = [
                        "real_a",
                        "real_b",
                        "real_c",
                        "real_alpha",
                        "real_beta",
                        "real_gamma",
                        "recip_a",
                        "recip_b",
                        "recip_c",
                        "recip_alpha",
                        "recip_beta",
                        "recip_gamma",
                        "lambda",
                    ]
                    point_group.create_dataset("lattice_labels", data=ltc_lbls)
                    lattice_data = list(read_head["param_data"][lattice_start : lattice_start + 12])
                    lattice_start = list(param_labs_list).index("g_LAMBDA")
                    lattice_data.append(read_head["param_data"][lattice_start])
                    point_group.create_dataset("lattice_values", data=lattice_data)

                    # Q
                    h_loc = list(point_labs_list).index("H")
                    k_loc = list(point_labs_list).index("K")
                    L_loc = list(point_labs_list).index("L")
                    h_val = read_head["point_data"][i][h_loc]
                    k_val = read_head["point_data"][i][k_loc]
                    L_val = read_head["point_data"][i][L_loc]
                    point_group.create_dataset("Q", data=[h_val, k_val, L_val])

                    # Or's: all of or0 followed by all of or1
                    or_start = list(param_labs_list).index("g_h0")
                    or_zero = list(read_head["param_data"][or_start : or_start + 3])
                    or_one = list(read_head["param_data"][or_start + 3 : or_start + 6])
                    # Or angles
                    or_start += 6
                    or_zero.extend(read_head["param_data"][or_start : or_start + 6])
                    or_one.extend(read_head["param_data"][or_start + 6 : or_start + 12])
                    # Or lambdas
                    or_start += 12
                    or_zero.extend(read_head["param_data"][or_start : or_start + 1])
                    or_one.extend(read_head["param_data"][or_start + 1 : or_start + 2])
                    or_zero.extend(or_one)
                    or_labs = [
                        "or0_h",
                        "or0_k",
                        "or0_L",
                        "or0_del",
                        "or0_eta",
                        "or0_chi",
                        "or0_phi",
                        "or0_nu",
                        "or0_mu",
                        "or0_lambda",
                        "or1_h",
                        "or1_k",
                        "or1_L",
                        "or1_del",
                        "or1_eta",
                        "or1_chi",
                        "or1_phi",
                        "or1_nu",
                        "or1_mu",
                        "or1_lambda",
                    ]
                    point_group.create_dataset("or_labels", data=or_labs)
                    point_group.create_dataset("or_values", data=or_zero)

                    # Azimuth vector
                    haz_start = list(param_labs_list).index("g_haz")
                    haz_data = read_head["param_data"][haz_start : haz_start + 3]
                    point_group.create_dataset("haz", data=haz_data)

                    # Position values
                    split_index = point_labs_list.index("Epoch")
                    pos_lbls = point_labs_list[:split_index]
                    point_group.create_dataset("position_labels", data=pos_lbls)
                    pos_values = read_head["point_data"][i][:split_index]
                    point_group.create_dataset("position_values", data=pos_values)

                    # Scaler values
                    sclr_lbls = point_labs_list[split_index:]
                    point_group.create_dataset("scaler_labels", data=sclr_lbls)
                    sclr_values = read_head["point_data"][i][split_index:]
                    point_group.create_dataset("scaler_values", data=sclr_values)

                    # Detector
                    det_group = point_group.create_group("det_0")
                    # Detector name
                    no_show = DETECTOR_PARAMETERS.get("name", "NA")
                    det_group.attrs["name"] = specified_attrs.get("name", no_show)
                    # Detector data
                    det_group.attrs["data"] = read_head["point_data"][i]
                    # Image data if it exists
                    try:
                        det_group.create_dataset("image_data", data=read_head["image_data"][i], compression="szip")
                    except Exception:
                        pass
                    # Integration parameters
                    int_labels = [
                        "bgrflag",
                        "cnbgr",
                        "compress",
                        "cpow",
                        "ctan",
                        "cwidth",
                        "filter",
                        "integrated",
                        "nline",
                        "rnbgr",
                        "roi",
                        "rpow",
                        "rtan",
                        "rwidth",
                    ]
                    det_group.create_dataset("int_labels", data=int_labels)
                    int_values = []
                    for label in int_labels:
                        no_show = INTEGRATION_PARAMETERS.get(label, "NA")
                        int_values.append(str(specified_attrs.get(label, no_show)))
                    det_group.create_dataset("int_values.1", data=int_values, dtype=var_len_strs)
                    # Correction parameters
                    corr_labels = [
                        "bad_pixel_map",
                        "bad_point",
                        "image_changed",
                        "image_max",
                        "pixel_map_changed",
                        "real_image_max",
                        "rotangle",
                        "sample_angles",
                        "sample_diameter",
                        "sample_polygon",
                        "scale",
                    ]
                    det_group.create_dataset("corr_labels", data=corr_labels)
                    corr_values = []
                    for label in corr_labels:
                        no_show = CORRECTION_PARAMETERS.get(label, "NA")
                        if label.startswith("bad_pixel_map"):
                            this_map = str(specified_attrs.get(label, no_show))
                            if not this_map.startswith("(") and not this_map.startswith("["):
                                this_map = str(read_pixel_map(this_map))
                            corr_values.append(this_map)
                        else:
                            corr_values.append(str(specified_attrs.get(label, no_show)))
                    det_group.create_dataset("corr_values.1", data=corr_values, dtype=var_len_strs)
                    # Detector parameters
                    det_labels = ["beam_slits", "det_slits"]
                    det_group.create_dataset("det_labels", data=det_labels)
                    det_values = []
                    for label in det_labels:
                        no_show = DETECTOR_PARAMETERS.get(label, "NA")
                        det_values.append(str(specified_attrs.get(label, no_show)))
                    det_group.create_dataset("det_values.1", data=det_values, dtype=var_len_strs)
                    # Results
                    res_labels = ["alpha", "beta", "ctot", "F", "F_changed", "Ferr", "I", "I_c", "I_r", "Ibgr", "Ibgr_c", "Ibgr_r", "Ierr", "Ierr_c", "Ierr_r"]
                    res_values = [0, 0, 0, 0, 1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  # F_changed: 1.0=True, 0.0=False for float64 compatibility
                    det_group.create_dataset("result_labels", data=res_labels)
                    det_group.create_dataset("result_values.1", data=res_values, dtype=np.float64)

                    if gui:
                        project_progress += 1
                        progress_continue, holding = progress_box.Update(project_progress)
                        wx.GetApp().Yield(True)
    except Exception as e:
        print("Error generating file:", e)
        if gui:
            progress_continue = False
            progress_box.Destroy()
        raise

    if gui:
        progress_continue = False
        progress_box.Destroy()

    read_this.close()
    write_this.close()
