import shutil
from pathlib import Path

from pyshortcuts import make_shortcut


def make_shortcuts() -> None:
    """Create desktop shortcuts for PDS applications."""
    print("Creating PDS desktop shortcuts...")

    # Find the pds executable
    pds_exe = shutil.which("pds")
    if not pds_exe:
        raise FileNotFoundError("Could not find pds executable in PATH")

    # Get the package directory to find the icons
    package_dir = Path(__file__).parent.parent  # Go up from cli/ to pds/
    filter_icon = str(package_dir / "icons" / "filter.png")
    integrator_icon = str(package_dir / "icons" / "integrator.png")

    # Create shortcuts using the full path to pds executable
    # Use folder parameter to create them in a PDS subfolder
    filter_cmd = f"{pds_exe} --filter"
    integrator_cmd = f"{pds_exe} --integrator"
    make_shortcut(filter_cmd, name="Filter", folder="PDS", terminal=True, icon=filter_icon)
    make_shortcut(integrator_cmd, name="Integrator", folder="PDS", terminal=True, icon=integrator_icon)
