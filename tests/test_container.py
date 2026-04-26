import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hdf5_container import HDF5Container


class TestHDF5Container(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.path = str(Path(self.tmpdir.name) / "test.hdf5")
        self.container = HDF5Container.from_path(path=self.path, flush_interval=1)

    def tearDown(self) -> None:
        self.container.close()
        self.tmpdir.cleanup()

    def test_store_and_access_scalar(self) -> None:
        self.container.store(keys=["group1"], name="x", data=123)
        value = self.container.access_value(keys=["group1"], name="x")
        self.assertEqual(value, 123)

    def test_store_and_access_string(self) -> None:
        self.container.store(keys=["group1"], name="message", data="hello")
        value = self.container.access_value(keys=["group1"], name="message")
        self.assertEqual(value, "hello")

    def test_reject_dtype_change_by_default(self) -> None:
        self.container.store(keys=["group1"], name="arr", data=np.array([1, 2, 3], dtype=np.int32))
        with self.assertRaises(TypeError):
            self.container.store(
                keys=["group1"],
                name="arr",
                data=np.array([1.1, 2.2, 3.3], dtype=np.float64),
            )

    def test_allow_dtype_change_when_enabled(self) -> None:
        self.container.store(keys=["group1"], name="arr", data=np.array([1, 2, 3], dtype=np.int32))
        self.container.store(
            keys=["group1"],
            name="arr",
            data=np.array([1.1, 2.2, 3.3], dtype=np.float64),
            is_dtype_change_enabled=True,
        )
        value = self.container.access_value(keys=["group1"], name="arr")
        np.testing.assert_allclose(value, np.array([1.1, 2.2, 3.3], dtype=np.float64))

    def test_flush_interval_accumulates_across_subgroup_accesses(self) -> None:
        container = HDF5Container.from_path(path=self.path, flush_interval=2)
        with patch.object(HDF5Container, "flush", autospec=True) as flush_mock:
            container.store(keys=["group1"], name="a", data=1)
            container.store(keys=["group1"], name="b", data=2)
            self.assertEqual(flush_mock.call_count, 1)
        container.close()


if __name__ == "__main__":
    unittest.main()
