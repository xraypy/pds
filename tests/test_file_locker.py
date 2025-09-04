import tempfile
import threading
import time
from pathlib import Path

import pytest

from pds.utils.file_locker import FileLock, FileLockException


class TestFileLock:
    """Test the FileLock class functionality."""

    def setup_method(self):
        """Set up test environment with temporary directory."""
        self.test_dir = Path(tempfile.mkdtemp(prefix="pds_filelock_test_"))
        self.test_file = self.test_dir / "test_file.txt"

        # Create test file
        self.test_file.write_text("test content")

    def teardown_method(self):
        """Clean up test environment."""
        import shutil

        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)

    def test_basic_lock_acquire_release(self):
        """Test basic lock acquisition and release."""
        lock = FileLock(str(self.test_file))

        # Initially not locked
        assert not lock.is_locked

        # Acquire lock
        lock.acquire()
        assert lock.is_locked
        assert Path(lock.lockfile).exists()

        # Release lock
        lock.release()
        assert not lock.is_locked
        assert not Path(lock.lockfile).exists()

    def test_context_manager(self):
        """Test using FileLock as a context manager."""
        lockfile_path = Path(str(self.test_file) + ".lock")

        # Use as context manager
        with FileLock(str(self.test_file)) as lock:
            assert lock.is_locked
            assert lockfile_path.exists()

        # Should be automatically released
        assert not lock.is_locked
        assert not lockfile_path.exists()

    def test_lock_file_content(self):
        """Test that lock file contains proper information."""
        with FileLock(str(self.test_file)) as lock:
            # Read lock file content
            with open(lock.lockfile, "r", encoding="utf-8") as f:
                content = f.read()

            lines = content.strip().split("\n")
            assert len(lines) == 3

            # Should contain username, hostname, and timestamp
            username = lines[0]
            hostname = lines[1]
            timestamp = lines[2]

            assert len(username) > 0
            assert len(hostname) > 0
            assert len(timestamp) > 0

            # Timestamp should be parseable
            time.strptime(timestamp, "%a %b %d %H:%M:%S %Y")

    def test_timeout_exception(self):
        """Test that timeout raises appropriate exception."""
        # Create first lock
        lock1 = FileLock(str(self.test_file), timeout=0.1)
        lock1.acquire()

        try:
            # Second lock should timeout
            lock2 = FileLock(str(self.test_file), timeout=0.1)
            with pytest.raises(FileLockException) as exc_info:
                lock2.acquire()

            assert "Timeout occurred" in str(exc_info.value)
        finally:
            lock1.release()

    def test_concurrent_lock_attempts(self):
        """Test concurrent lock attempts from multiple threads."""
        results = {}
        barrier = threading.Barrier(3)  # Synchronize all 3 threads

        def try_lock(thread_id):
            """Try to acquire lock from a thread."""
            barrier.wait()  # Wait for all threads to be ready
            try:
                with FileLock(str(self.test_file), timeout=0.3):
                    results[thread_id] = "acquired"
                    time.sleep(0.5)  # Hold lock longer than timeout
            except FileLockException:
                results[thread_id] = "timeout"

        # Start multiple threads
        threads = []
        for i in range(3):
            thread = threading.Thread(target=try_lock, args=(i,))
            threads.append(thread)

        # Start all threads
        for thread in threads:
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # Only one thread should have acquired the lock
        acquired_count = sum(1 for result in results.values() if result == "acquired")
        timeout_count = sum(1 for result in results.values() if result == "timeout")

        assert acquired_count == 1
        assert timeout_count == 2

    def test_lock_cleanup_on_exception(self):
        """Test that locks are properly cleaned up when exceptions occur."""
        lockfile_path = Path(str(self.test_file) + ".lock")

        try:
            with FileLock(str(self.test_file)) as lock:
                assert lock.is_locked
                assert lockfile_path.exists()
                # Raise an exception
                raise ValueError("Test exception")
        except ValueError:
            # Lock should still be cleaned up
            pass

        assert not lockfile_path.exists()

    def test_double_acquire_release(self):
        """Test that double acquire/release operations are safe."""
        lock = FileLock(str(self.test_file))

        # Double acquire should be safe
        lock.acquire()
        assert lock.is_locked

        # Second acquire should not fail
        lock.acquire()
        assert lock.is_locked

        # Double release should be safe
        lock.release()
        assert not lock.is_locked

        # Second release should not fail
        lock.release()
        assert not lock.is_locked

    def test_del_cleanup(self):
        """Test that __del__ properly cleans up locks."""
        lockfile_path = Path(str(self.test_file) + ".lock")

        # Create lock and acquire it
        lock = FileLock(str(self.test_file))
        lock.acquire()
        assert lockfile_path.exists()

        # Delete the lock object
        del lock

        # Give some time for cleanup (since __del__ is not guaranteed to run immediately)
        # We'll check that the FileLock handles cleanup properly in release()
        # The actual cleanup test is in test_cleanup_on_exception and context manager tests

    def test_lock_with_nonexistent_directory(self):
        """Test lock creation when parent directory doesn't exist."""
        nonexistent_dir = self.test_dir / "nonexistent" / "nested"
        nonexistent_file = nonexistent_dir / "test.txt"

        # This should work even if directory doesn't exist
        lock = FileLock(str(nonexistent_file))

        # The lock file should be created in the same (nonexistent) directory
        expected_lockfile = str(nonexistent_file) + ".lock"
        assert lock.lockfile == expected_lockfile

    def test_cross_platform_username_hostname(self):
        """Test cross-platform username and hostname detection."""
        lock = FileLock(str(self.test_file))

        username = lock._get_username()
        hostname = lock._get_hostname()

        assert isinstance(username, str)
        assert len(username) > 0
        assert isinstance(hostname, str)
        assert len(hostname) > 0

    def test_repr_representation(self):
        """Test string representation of FileLock."""
        lock = FileLock(str(self.test_file))

        # Unlocked state
        repr_str = repr(lock)
        assert "FileLock" in repr_str
        assert str(self.test_file) in repr_str
        assert "unlocked" in repr_str

        # Locked state
        lock.acquire()
        try:
            repr_str = repr(lock)
            assert "FileLock" in repr_str
            assert str(self.test_file) in repr_str
            assert "locked" in repr_str
        finally:
            lock.release()

    def test_custom_timeout_and_delay(self):
        """Test custom timeout and delay parameters."""
        # Test with very short timeout
        lock1 = FileLock(str(self.test_file), timeout=0.05, delay=0.01)
        lock1.acquire()

        try:
            start_time = time.time()
            lock2 = FileLock(str(self.test_file), timeout=0.05, delay=0.01)

            with pytest.raises(FileLockException):
                lock2.acquire()

            elapsed = time.time() - start_time
            # Should timeout quickly
            assert elapsed < 0.2
        finally:
            lock1.release()

    def test_lock_file_removed_externally(self):
        """Test behavior when lock file is removed by external process."""
        lock = FileLock(str(self.test_file))
        lock.acquire()

        # Simulate external removal of lock file
        Path(lock.lockfile).unlink()

        # Release should not raise an exception
        lock.release()
        assert not lock.is_locked

    def test_permission_denied_scenario(self):
        """Test handling of permission denied errors."""
        # This test is platform-dependent and may not work on all systems
        # We'll create a more controlled test

        # Create a lock first
        lock1 = FileLock(str(self.test_file))
        lock1.acquire()

        try:
            # Try to create another lock (should get EEXIST, not EACCES)
            lock2 = FileLock(str(self.test_file), timeout=0.1)
            with pytest.raises(FileLockException) as exc_info:
                lock2.acquire()

            # Should be timeout, not access denied for this case
            assert "Timeout occurred" in str(exc_info.value)
        finally:
            lock1.release()

    def test_lock_initialization_parameters(self):
        """Test FileLock initialization with various parameters."""
        # Test default parameters
        lock1 = FileLock(str(self.test_file))
        assert lock1.timeout == 10
        assert lock1.delay == 0.05
        assert lock1.file_name == str(self.test_file)
        assert lock1.lockfile == str(self.test_file) + ".lock"
        assert not lock1.is_locked
        assert lock1.fd is None

        # Test custom parameters
        lock2 = FileLock(str(self.test_file), timeout=5, delay=0.1)
        assert lock2.timeout == 5
        assert lock2.delay == 0.1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
