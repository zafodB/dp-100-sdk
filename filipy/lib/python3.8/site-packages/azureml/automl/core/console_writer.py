# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------
"""Wrapper for file objects to ensure flushing."""
from typing import Any, Optional, TextIO
import os


class ConsoleWriter:
    """Wrapper for file objects to ensure flushing."""

    def __init__(self, f: Optional[TextIO] = None, show_output: bool = True) -> None:
        """
        Construct a ConsoleWriter.

        :param f: the underlying file stream
        :return:
        """
        if f is None:
            import atexit

            devnull = open(os.devnull, "w", encoding="utf-8")
            atexit.register(devnull.close)
            self._file = devnull  # type: TextIO
        else:
            self._file = f

        self.show_output = show_output

    def print(self, text: str, carriage_return: bool = False) -> None:
        """
        Write to the underlying file. The file is flushed.

        :param text: the text to write
        :param carriage_return: Add the carriage return.
        :return:
        """
        if carriage_return:
            self.write(text + "\r")
        else:
            self.write(text)
        self.flush()

    def println(self, text: Optional[str] = None) -> None:
        """
        Write to the underlying file. A newline character is also written and the file is flushed.

        If the text provided is None, just a new line character will be written.

        :param text: the text to write
        :return:
        """
        if text is not None:
            self.write(text)
        self.write("\n")
        self.flush()

    def write(self, text: str) -> None:
        """
        Write directly to the underlying file.

        :param text: the text to write
        :return:
        """
        if self.show_output:
            self._file.write(text)

    def flush(self) -> None:
        """
        Flush the underlying file.

        :return:
        """
        self._file.flush()
