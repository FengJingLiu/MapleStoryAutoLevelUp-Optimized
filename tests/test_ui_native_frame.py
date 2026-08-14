import os
import subprocess
import sys
import textwrap
import unittest
from pathlib import Path


class NativeFrameQtTests(unittest.TestCase):
    def test_odd_width_and_non_contiguous_frames_do_not_crash_qt(self):
        """Run Qt out of process because the old bug was a native crash."""
        repo_root = Path(__file__).resolve().parents[1]
        script = textwrap.dedent(
            """
            import os
            os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

            from types import SimpleNamespace

            import numpy as np
            from PySide6.QtWidgets import QApplication, QLabel

            from src.ui.ui import MainWindow

            app = QApplication.instance() or QApplication([])
            debug_canvas = QLabel()
            debug_canvas.resize(1280, 650)
            route_canvas = QLabel()
            route_canvas.resize(800, 800)
            holder = SimpleNamespace(
                debug_canvas=debug_canvas,
                route_map_canvas=route_canvas,
            )

            # 3579 * 3 == 10737, which is not four-byte aligned. The previous
            # QImage overload inferred 10740 bytes per line and crashed Qt.
            native = np.zeros((2013, 3579, 3), dtype=np.uint8)
            MainWindow.update_debug_canvas(holder, native)
            assert debug_canvas.pixmap() is not None
            assert not debug_canvas.pixmap().isNull()

            # Also exercise the defensive contiguous copy used for array ROIs.
            non_contiguous = native[::2, ::2]
            assert not non_contiguous.flags.c_contiguous
            MainWindow.update_route_map_canvas(holder, non_contiguous)
            assert route_canvas.pixmap() is not None
            assert not route_canvas.pixmap().isNull()
            """
        )
        env = os.environ.copy()
        env["QT_QPA_PLATFORM"] = "offscreen"
        completed = subprocess.run(
            [sys.executable, "-c", script],
            cwd=repo_root,
            env=env,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        self.assertEqual(
            completed.returncode,
            0,
            msg=(
                f"Qt subprocess exited with {completed.returncode}\n"
                f"stdout:\n{completed.stdout}\n"
                f"stderr:\n{completed.stderr}"
            ),
        )


if __name__ == "__main__":
    unittest.main()
