import argparse
import sys
from pathlib import Path

from pyshortcuts import make_shortcut

from pds import run_filter, run_integrator
from tests import run_all_tests


def main() -> None:
    """Main entry point for the PDS tools."""
    parser = argparse.ArgumentParser(description="PDS Command Line Interface")
    parser.add_argument("-c", "--convert", nargs=argparse.REMAINDER, help="run the spec_to_hdf CLI tool with arguments")
    parser.add_argument("-i", "--integrator", action="store_true", help="run the integrator GUI")
    parser.add_argument("-f", "--filter", action="store_true", help="run the filter GUI")
    parser.add_argument("-t", "--test", action="store_true", help="run all tests")
    parser.add_argument("-m", "--make_icon", action="store_true", help="create desktop shortcuts")
    args = parser.parse_args()

    if args.convert is not None:
        print("Not implemented yet")
    elif args.integrator:
        run_integrator()
    elif args.filter:
        run_filter()
    elif args.test:
        run_all_tests()
    elif args.make_icon:
        python_exe = sys.executable
        pds_script = str(Path(__file__).resolve())
        filter_cmd = f'"{python_exe}" "{pds_script}" --filter'
        integrator_cmd = f'"{python_exe}" "{pds_script}" --integrator'
        make_shortcut(filter_cmd, name="PDS Filter", terminal=True)
        make_shortcut(integrator_cmd, name="PDS Integrator", terminal=True)


if __name__ == "__main__":
    main()
