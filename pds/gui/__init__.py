import wx

from pds.gui.filter import Filter
from pds.gui.integrator import Integrator

__all__ = ["run_integrator", "run_filter"]


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
