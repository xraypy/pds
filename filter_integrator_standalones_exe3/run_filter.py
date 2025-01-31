import wx
from wxFilter import filterGUI

app = wx.App()
myIntegrator = filterGUI(None)
app.MainLoop()    

