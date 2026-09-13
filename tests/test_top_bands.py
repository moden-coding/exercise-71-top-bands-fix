#!/usr/bin/env python3

import unittest
from unittest.mock import patch

import pandas as pd

from src.top_bands import top_bands, main


def spy_decorator(method_to_decorate, name):
    """Wrap a bound method so calls are recorded without losing behavior.

    Copied from the assignment's former tmc.utils helper (originally from
    https://stackoverflow.com/questions/25608107) so the test suite no
    longer depends on the vendored tmc package.
    """
    from unittest.mock import MagicMock

    mock = MagicMock(name="%s method" % name)

    def wrapper(self, *args, **kwargs):
        mock(*args, **kwargs)
        return method_to_decorate(self, *args, **kwargs)

    wrapper.mock = mock
    return wrapper


class TestTopBands(unittest.TestCase):
    """top_bands() -> merged top-40/band-lineup DataFrame."""

    def test_shape(self):
        df = top_bands()
        self.assertEqual(
            df.shape,
            (9, 13),
            msg="top_bands() should return a DataFrame with shape (9, 13): "
            "one row per matching chart entry and 13 columns. Incorrect "
            "shape!",
        )

    def test_columns(self):
        df = top_bands()
        cols = [
            "Pos",
            "LW",
            "Title",
            "Artist",
            "Publisher",
            "Peak Pos",
            "WoC",
            "Band",
            "Singer",
            "Lead guitar",
            "Rhythm guitar",
            "Bass",
            "Drums",
        ]
        self.assertCountEqual(
            df.columns,
            cols,
            msg="top_bands() should return a DataFrame with exactly these "
            "columns (chart columns plus band lineup columns). Incorrect "
            "columns!",
        )

    def test_calls(self):
        merge_method = spy_decorator(pd.core.frame.DataFrame.merge, "merge")
        with patch(
            "src.top_bands.top_bands", wraps=top_bands
        ) as ptop, patch(
            "src.top_bands.pd.read_csv", wraps=pd.read_csv
        ) as prc, patch.object(
            pd.core.frame.DataFrame, "merge", new=merge_method
        ), patch(
            "src.top_bands.pd.merge", wraps=pd.merge
        ) as pmerge:
            main()
            ptop.assert_called_once()
            self.assertEqual(
                prc.call_count,
                2,
                msg="You should have called pd.read_csv exactly twice: "
                "once for the top-40 chart and once for the band lineups.",
            )
            self.assertTrue(
                pmerge.call_count == 1 or merge_method.mock.call_count == 1,
                msg="Call merge exactly once, either as pd.merge(...) or as "
                "a DataFrame.merge(...) method call!",
            )

            if pmerge.call_count >= 1:
                args, kwargs = pmerge.call_args
            else:
                args, kwargs = merge_method.mock.call_args

            self.assertTrue(
                "left_on" in kwargs,
                msg="You should have used the 'left_on' argument of "
                "pd.merge!",
            )
            self.assertTrue(
                "right_on" in kwargs,
                msg="You should have used the 'right_on' argument of "
                "pd.merge!",
            )
            params = [kwargs["left_on"], kwargs["right_on"]]
            self.assertTrue(
                ("Artist" in params or ["Artist"] in params)
                and ("Band" in params or ["Band"] in params),
                msg="You should have merged on the 'Artist' and 'Band' "
                "columns! Got left_on=%r, right_on=%r."
                % (kwargs["left_on"], kwargs["right_on"]),
            )


if __name__ == "__main__":
    unittest.main()
