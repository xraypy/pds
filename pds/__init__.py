import argparse

from pds.cli import create_shortcuts, print_help, run_converter
from pds.gui import run_filter, run_integrator
from tests import run_all_tests

__all__ = ["run_integrator", "run_filter", "main"]

EXAMPLES = """
examples:
  %(prog)s --convert data.spec                    # Convert single SPEC file
  %(prog)s --convert /path/to/specs --once        # Convert all SPEC files in directory
  %(prog)s --convert /path/to/specs               # Monitor directory for new files
  %(prog)s --convert help                         # Show detailed convert options
  %(prog)s --integrator                           # Launch integrator GUI
  %(prog)s --filter                               # Launch filter GUI
  %(prog)s --test                                 # Run all tests
  %(prog)s --make_icon                            # Create desktop shortcuts
"""


def main() -> None:
    """Main entry point for the PDS tools."""
    parser = argparse.ArgumentParser(
        description="PDS (Surface Scattering Integration) Command Line Interface",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=EXAMPLES,
    )
    parser.add_argument("-c", "--convert", nargs=argparse.REMAINDER, help="convert SPEC files to HDF5. Use 'pds --convert help' for detailed options")
    parser.add_argument("-i", "--integrator", action="store_true", help="run the integrator GUI")
    parser.add_argument("-f", "--filter", action="store_true", help="run the filter GUI")
    parser.add_argument("-t", "--test", action="store_true", help="run all tests")
    parser.add_argument("-m", "--make_icon", action="store_true", help="create desktop shortcuts")

    args = parser.parse_args()

    if args.convert is not None:
        # Check if user is asking for convert help
        if len(args.convert) > 0 and args.convert[0].lower() in ["help", "--help", "-h"]:
            print_help()
        else:
            run_converter(args.convert)
    elif args.integrator:
        run_integrator()
    elif args.filter:
        run_filter()
    elif args.test:
        run_all_tests()
    elif args.make_icon:
        create_shortcuts()
    else:
        parser.print_help()
