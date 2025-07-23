import re

import wx


class ConsoleCapture:
    """Enhanced console capture with color support and better functionality"""

    def __init__(self, text_ctrl, is_stderr=False):
        """Initialize the console capture."""
        self.text_ctrl = text_ctrl
        self.is_stderr = is_stderr
        self.buffer = ""
        self.ansi_escape = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")

        # Color mapping for different message types
        self.colors = {
            "error": wx.Colour(220, 50, 47),  # Red
            "warning": wx.Colour(255, 193, 7),  # Orange/Yellow
            "info": wx.Colour(38, 139, 210),  # Blue
            "success": wx.Colour(133, 153, 0),  # Green
            "debug": wx.Colour(108, 113, 196),  # Purple
            "stderr": wx.Colour(220, 50, 47),  # Red for stderr
            "stdout": wx.Colour(0, 0, 0),  # Black for stdout
        }

        # Text attributes for styles
        self.styles = {}
        self._create_text_styles()

    def _create_text_styles(self):
        """Create text styles for different message types"""
        if not self.text_ctrl:
            return

        default_font = self.text_ctrl.GetFont()

        for style_name, color in self.colors.items():
            attr = wx.TextAttr()
            attr.SetTextColour(color)
            attr.SetFont(default_font)
            self.styles[style_name] = attr

    def write(self, string):
        """Write string to the text control with color support"""
        if not string or not self.text_ctrl:
            return

        # Buffer the string to handle partial writes
        self.buffer += string

        # Process complete lines
        while "\n" in self.buffer:
            line, self.buffer = self.buffer.split("\n", 1)
            wx.CallAfter(self._write_line_to_ctrl, line + "\n")

        # If there's remaining content and it looks complete, write it
        if self.buffer and (len(self.buffer) > 100 or self.buffer.endswith(" ")):
            wx.CallAfter(self._write_line_to_ctrl, self.buffer)
            self.buffer = ""

    def _write_line_to_ctrl(self, line):
        """Write a line to the text control with appropriate coloring"""
        try:
            if not self.text_ctrl:
                return

            # Remove ANSI escape sequences
            clean_line = self.ansi_escape.sub("", line)

            # Determine the appropriate style based on content and source
            style = self._determine_style(clean_line)

            # Get current position for applying style
            start_pos = self.text_ctrl.GetLastPosition()

            # Append the text
            self.text_ctrl.AppendText(clean_line)

            # Apply styling to the newly added text
            end_pos = self.text_ctrl.GetLastPosition()
            if style and end_pos > start_pos:
                self.text_ctrl.SetStyle(start_pos, end_pos, style)

            # Auto-scroll to bottom
            self.text_ctrl.SetInsertionPointEnd()

        except Exception:
            # If the control is destroyed or there's an error, ignore it
            pass

    def _determine_style(self, text):
        """Determine the appropriate style based on text content"""
        text_lower = text.lower().strip()

        # Check for specific patterns
        if any(keyword in text_lower for keyword in ["error:", "exception:", "traceback", "failed"]):
            return self.styles.get("error")
        elif any(keyword in text_lower for keyword in ["warning:", "warn:", "deprecated"]):
            return self.styles.get("warning")
        elif any(keyword in text_lower for keyword in ["info:", "loading", "saved", "completed"]):
            return self.styles.get("info")
        elif any(keyword in text_lower for keyword in ["success:", "done:", "finished", "ok"]):
            return self.styles.get("success")
        elif any(keyword in text_lower for keyword in ["debug:", "trace:"]):
            return self.styles.get("debug")
        elif self.is_stderr:
            return self.styles.get("stderr")
        else:
            return self.styles.get("stdout")

    def flush(self):
        """Flush any remaining buffer content"""
        if self.buffer and self.text_ctrl:
            wx.CallAfter(self._write_line_to_ctrl, self.buffer)
            self.buffer = ""

    def clear(self):
        """Clear the text control"""
        if self.text_ctrl:
            wx.CallAfter(self.text_ctrl.Clear)

    def close(self):
        """Close the capture and flush remaining content"""
        self.flush()
        self.text_ctrl = None
