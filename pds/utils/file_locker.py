import errno
import getpass
import os
import socket
import time
from pathlib import Path
from typing import Any


class FileLockException(Exception):
    """Exception raised when file locking operations fail."""

    pass


class FileLock:
    """
    Cross-platform file locking with context manager support using pathlib.
    Portable across operating systems without platform-specific dependencies.
    """

    def __init__(self, file_name: str, timeout: float = 10, delay: float = 0.05) -> None:
        """Initialize file locker with specified timeout and retry delay."""
        self.is_locked = False
        self.file_path = Path(file_name)
        self.lockfile = str(self.file_path.with_suffix(self.file_path.suffix + ".lock"))
        self.file_name = file_name
        self.timeout = timeout
        self.delay = delay
        self.fd: int | None = None

    def acquire(self) -> None:
        """Acquire lock with timeout and retry logic."""
        # If already locked, return immediately
        if self.is_locked:
            return

        start_time = time.time()
        while True:
            try:
                self.fd = os.open(self.lockfile, os.O_CREAT | os.O_EXCL | os.O_RDWR)

                # Write lock information for debugging
                username = self._get_username()
                hostname = self._get_hostname()
                timestamp = time.ctime(time.time())

                os.write(self.fd, f"{username}\n".encode("utf-8"))
                os.write(self.fd, f"{hostname}\n".encode("utf-8"))
                os.write(self.fd, f"{timestamp}\n".encode("utf-8"))
                break

            except OSError as e:
                if e.errno not in (errno.EEXIST, errno.EACCES):
                    raise

                if (time.time() - start_time) >= self.timeout:
                    if e.errno == errno.EEXIST:
                        raise FileLockException("Timeout occurred while waiting for lock.")
                    else:
                        raise FileLockException("Access denied for lock file.")

                time.sleep(self.delay)

        self.is_locked = True

    def release(self) -> None:
        """Release lock by deleting lockfile."""
        if self.is_locked and self.fd is not None:
            try:
                os.close(self.fd)
                Path(self.lockfile).unlink()
            except OSError:
                # Lock file may have been removed by another process
                pass
            finally:
                self.is_locked = False
                self.fd = None

    def _get_username(self) -> str:
        """Get current username in cross-platform way."""
        return os.environ.get("USERNAME") or os.environ.get("USER") or getpass.getuser()

    def _get_hostname(self) -> str:
        """Get current hostname in cross-platform way."""
        return os.environ.get("COMPUTERNAME") or os.environ.get("HOSTNAME") or socket.gethostname()

    def __enter__(self) -> "FileLock":
        """Context manager entry point."""
        if not self.is_locked:
            self.acquire()
        return self

    def __exit__(self, exc_type: type | None, exc_val: Exception | None, exc_tb: Any) -> None:
        """Context manager exit point."""
        if self.is_locked:
            self.release()

    def __del__(self) -> None:
        """Cleanup lockfiles on object deletion."""
        self.release()

    def __repr__(self) -> str:
        """Return string representation of FileLock."""
        status = "locked" if self.is_locked else "unlocked"
        return f"FileLock(file='{self.file_name}', status='{status}')"
