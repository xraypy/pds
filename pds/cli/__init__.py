import shutil
import sys
from pathlib import Path
from typing import List

from pyshortcuts import make_shortcut

from pds.cli.auto_master import AutoMasterMonitor
from pds.cli.spectohdf import spec_to_hdf5

__all__ = ["run_converter", "print_help", "create_shortcuts", "create_installers"]

CONVERT_HELP_TEXT = """
Convert Command Usage:

Single file conversion:
  pds --convert input.spec [output.mh5] [options]
  
Directory monitoring (auto-master):
  pds --convert /path/to/spec/directory [options]

Options for single file conversion:
  -w                    Overwrite existing output file
  -a                    Append to existing output file (default)
  -q                    Quiet mode (suppress verbose output)
  -i <image_dir>        Custom image directory path

Options for directory monitoring:
  -i <seconds>          Monitoring interval in seconds (default: 20)
  -q                    Quiet mode (suppress verbose output)
  --process-existing    Process all existing files before monitoring
  --once                Process existing files once and exit (no monitoring)

Examples:
  pds --convert data.spec                    # Convert single file
  pds --convert data.spec output.mh5 -w      # Convert with specific output, overwrite
  pds --convert /path/to/specs                # Monitor directory
  pds --convert /path/to/specs -i 30          # Monitor with 30-second interval
  pds --convert /path/to/specs --once         # Process existing files once
"""


def run_converter(args: list[str]) -> None:
    """Smart convert command that automatically detects if input is file or directory."""

    # Handle no arguments
    if not args:
        print("Error: No path specified for conversion")
        print("Usage: pds --convert <file_or_directory> [options]")
        return

    input_path = Path(args[0])

    # Handle path does not exist
    if not input_path.exists():
        print(f"Error: Path does not exist: {input_path}")
        return

    try:
        # Determine if input is file or directory
        if input_path.is_file():
            _convert_single_file(input_path, args[1:])
        elif input_path.is_dir():
            _monitor_directory(input_path, args[1:])
        else:
            print(f"Error: Path is neither a file nor directory: {input_path}")
            return
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        return
    except Exception as e:
        print(f"Unexpected error: {e}")
        return


def _convert_single_file(spec_file: Path, remaining_args: list[str]) -> int:
    """Convert a single SPEC file to HDF5."""
    # Parse additional arguments
    output_file = None
    mode = "a"  # Default to append
    verbose = True
    image_dir = None

    i = 0
    while i < len(remaining_args):
        arg = remaining_args[i]
        if arg == "-w":
            mode = "w"
        elif arg == "-a":
            mode = "a"
        elif arg == "-q":
            verbose = False
        elif arg == "-i" and i + 1 < len(remaining_args):
            image_dir = Path(remaining_args[i + 1])
            i += 1
        elif not arg.startswith("-"):
            # Assume it's the output file if no output specified yet
            if output_file is None:
                output_file = Path(arg)
        i += 1

    # Generate output filename if not provided
    if output_file is None:
        output_file = spec_file.with_suffix(".mh5")
    elif not output_file.suffix:
        output_file = output_file.with_suffix(".mh5")

    # Check if output file exists and handle interactive mode
    if output_file.exists() and mode == "a":
        if sys.stdin.isatty():  # Interactive terminal
            while True:
                response = input(f"File '{output_file}' exists. (A)ppend, over(W)rite, or (C)ancel? [A/w/c]: ").lower().strip()

                if response in ("", "a"):
                    mode = "a"
                    break
                elif response == "w":
                    mode = "w"
                    break
                elif response == "c":
                    print("Operation cancelled")
                    return 1
                else:
                    print("Please enter 'A', 'W', or 'C'")

    if verbose:
        print(f"Converting SPEC file: {spec_file}")
        print(f"Output file: {output_file}")
        if image_dir:
            print(f"Image directory: {image_dir}")
        print(f"Mode: {'Overwrite' if mode == 'w' else 'Append'}")

    # Perform conversion
    success = spec_to_hdf5(spec_file=spec_file, output_file=output_file, mode=mode, image_dir=image_dir, verbose=verbose)

    return 0 if success else 1


def _monitor_directory(spec_dir: Path, remaining_args: List[str]) -> int:
    """Monitor directory for SPEC files (auto-master mode)."""
    # Parse additional arguments
    interval = 20
    verbose = True
    process_existing = False
    once = False

    i = 0
    while i < len(remaining_args):
        arg = remaining_args[i]
        if arg == "-i" and i + 1 < len(remaining_args):
            try:
                interval = int(remaining_args[i + 1])
                i += 1
            except ValueError:
                print(f"Error: Invalid interval value: {remaining_args[i + 1]}")
                return 1
        elif arg == "-q":
            verbose = False
        elif arg == "--process-existing":
            process_existing = True
        elif arg == "--once":
            once = True
        i += 1

    if verbose:
        print(f"Auto-master mode: monitoring directory {spec_dir}")
        print(f"Interval: {interval} seconds")
        if process_existing:
            print("Will process existing files first")
        if once:
            print("Will process existing files once and exit")

    try:
        monitor = AutoMasterMonitor(spec_dir=spec_dir, interval=interval, verbose=verbose)

        if once:
            # Process existing files once and exit
            monitor.process_existing_files()
        else:
            # Process existing files if requested
            if process_existing:
                monitor.process_existing_files()

            # Start monitoring
            monitor.monitor()

        return 0

    except FileNotFoundError as e:
        print(f"Error: {e}")
        return 1
    except Exception as e:
        print(f"Error: {e}")
        return 1


def print_help() -> None:
    """Print help information for the convert command."""
    help_text = CONVERT_HELP_TEXT
    print(help_text.strip())


def create_shortcuts() -> None:
    """Create desktop shortcuts for PDS applications."""
    print("Creating PDS desktop shortcuts...")

    # Find the pds executable: first in PATH, then next to this Python (e.g. post-install)
    pds_exe = shutil.which("pds")
    if not pds_exe:
        prefix = Path(sys.executable).resolve().parent
        if sys.platform == "win32":
            candidate = prefix / "Scripts" / "pds.exe"
        else:
            candidate = prefix / "pds"
        if candidate.exists():
            pds_exe = str(candidate)
        else:
            raise FileNotFoundError("Could not find pds executable in PATH or next to Python")

    # Get the package directory to find the icons
    package_dir = Path(__file__).parent.parent  # Go up from cli/ to pds/
    filter_icon = str(package_dir / "icons" / "filter.png")
    integrator_icon = str(package_dir / "icons" / "integrator.png")

    # Create shortcuts using the full path to pds executable
    # Use folder parameter to create them in a PDS subfolder
    filter_cmd = f"{pds_exe} --filter"
    integrator_cmd = f"{pds_exe} --integrator"
    make_shortcut(filter_cmd, name="Filter", folder="PDS", terminal=False, icon=filter_icon)
    make_shortcut(integrator_cmd, name="Integrator", folder="PDS", terminal=False, icon=integrator_icon)
