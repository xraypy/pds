import signal
import time
from pathlib import Path
from typing import Dict, Optional, Set

from pds.cli.spectohdf import spec_to_hdf5


class AutoMasterMonitor:
    """Monitor SPEC files and automatically convert them to HDF5."""

    def __init__(self, spec_dir: Path, scripts_dir: Optional[Path] = None, interval: int = 20, verbose: bool = True) -> None:
        """Initialize the auto master monitor."""
        self.spec_dir = Path(spec_dir)
        self.scripts_dir = Path(scripts_dir) if scripts_dir else None
        self.interval = interval
        self.verbose = verbose
        self.spec_sizes: Dict[str, int] = {}
        self.running = True

        # Validate directories
        if not self.spec_dir.exists():
            raise FileNotFoundError(f"SPEC directory not found: {self.spec_dir}")
        if not self.spec_dir.is_dir():
            raise NotADirectoryError(f"Not a directory: {self.spec_dir}")

    def _find_spec_files(self) -> Set[Path]:
        """Find all SPEC files in the monitored directory."""
        spec_files = set()
        for pattern in ["*.spc", "*.spec"]:
            spec_files.update(self.spec_dir.glob(pattern))
        return spec_files

    def _get_file_size(self, file_path: Path) -> int:
        """Get file size safely."""
        try:
            return file_path.stat().st_size
        except OSError:
            return 0

    def _should_process_file(self, spec_file: Path) -> bool:
        """Check if file should be processed (new or modified)."""
        file_name = spec_file.name
        current_size = self._get_file_size(spec_file)

        if file_name not in self.spec_sizes:
            # New file
            self.spec_sizes[file_name] = current_size
            return True
        elif self.spec_sizes[file_name] < current_size:
            # Modified file (grown in size)
            self.spec_sizes[file_name] = current_size
            return True

        return False

    def _convert_spec_file(self, spec_file: Path) -> bool:
        """Convert a single SPEC file to HDF5."""
        try:
            output_file = spec_file.with_suffix(".mh5")

            if self.verbose:
                print(f"Converting {spec_file.name} -> {output_file.name}")

            success = spec_to_hdf5(
                spec_file=spec_file,
                output_file=output_file,
                mode="a",
                verbose=self.verbose,
            )

            if success and self.verbose:
                print(f"Successfully converted {spec_file.name}")
            elif not success:
                print(f"Failed to convert {spec_file.name}")

            return success

        except Exception as e:
            print(f"Error converting {spec_file.name}: {e}")
            return False

    def _setup_signal_handlers(self) -> None:
        """Setup signal handlers for graceful shutdown."""

        def signal_handler(signum, frame):
            print("\nReceived interrupt signal. Shutting down gracefully...")
            self.running = False

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

    def monitor(self) -> None:
        """Main monitoring loop."""
        self._setup_signal_handlers()

        if self.verbose:
            print("Starting auto master monitor...")
            print(f"Monitoring directory: {self.spec_dir}")
            print(f"Update interval: {self.interval} seconds")
            print("Press Ctrl+C to stop\n")

        try:
            while self.running:
                try:
                    spec_files = self._find_spec_files()

                    if not spec_files and self.verbose:
                        print("No SPEC files found in directory")

                    for spec_file in spec_files:
                        if not self.running:
                            break

                        if self._should_process_file(spec_file):
                            self._convert_spec_file(spec_file)

                    if self.running:
                        if self.verbose:
                            print(f"Waiting {self.interval} seconds...")
                        time.sleep(self.interval)

                except KeyboardInterrupt:
                    break
                except Exception as e:
                    print(f"Error during monitoring: {e}")
                    if self.running:
                        time.sleep(self.interval)

        finally:
            if self.verbose:
                print("Auto master monitor stopped")

    def process_existing_files(self) -> None:
        """Process all existing SPEC files once."""
        spec_files = self._find_spec_files()

        if self.verbose:
            print(f"Processing {len(spec_files)} existing SPEC files...")

        for spec_file in spec_files:
            self._convert_spec_file(spec_file)

        if self.verbose:
            print("Finished processing existing files")
