import os
import platform
import sys
import warnings

import wx

from pds.gui.filter import Filter
from pds.gui.integrator import Integrator

__all__ = ["run_integrator", "run_filter"]


# Suppress macOS NSOpenPanel warning for wxPython file dialogs
if platform.system() == "Darwin":
    import logging

    os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"
    os.environ["OBJC_SILENCE_ROOT_WARNING"] = "1"

    warnings.filterwarnings("ignore", message=".*NSOpenPanel.*")
    warnings.filterwarnings("ignore", message=".*NSWindow.*")
    warnings.filterwarnings("ignore", message=".*identifier.*")
    warnings.filterwarnings("ignore", message=".*overrides.*")

    logging.getLogger().addFilter(
        lambda record: not any(
            term in str(record.getMessage()) if hasattr(record, "getMessage") else str(record)
            for term in ["NSOpenPanel", "NSWindow", "identifier", "overrides"]
        )
    )

    original_stderr = sys.stderr

    class NSWarningFilter:
        def __init__(self, original_stream):
            self.original_stream = original_stream

        def write(self, text):
            # Filter out the specific NSOpenPanel warning
            if not any(term in text for term in ["NSOpenPanel", "identifier", "overrides the method identifier"]):
                self.original_stream.write(text)

        def flush(self):
            self.original_stream.flush()

        def __getattr__(self, name):
            return getattr(self.original_stream, name)

    # Apply the filter to stderr
    sys.stderr = NSWarningFilter(original_stderr)


def run_integrator() -> None:
    """Run the PDS integrator GUI."""
    app = wx.App()
    Integrator(None)
    app.MainLoop()


def run_filter() -> None:
    """Run the PDS filter GUI."""
    app = wx.App()
    Filter(None)
    app.MainLoop()
