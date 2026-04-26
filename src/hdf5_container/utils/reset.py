import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..container import HDF5Container


def reset_hdf5(
    hdf5_path: str,
    flush_interval: int = 100
) -> "HDF5Container":
    """
    Reset HDF5 container.
    * HDF5 is temporary file for storing tracklets on the process.

    Parameters:
    ----------
    hdf5_path: str
        The path to the HDF5 file.
    """
    if os.path.exists(hdf5_path):
        os.remove(hdf5_path)
    parent_dir = os.path.dirname(os.path.abspath(hdf5_path))
    os.makedirs(parent_dir, exist_ok=True)
    from ..container import HDF5Container

    return HDF5Container.from_path(
        path=hdf5_path,
        flush_interval=flush_interval
    )
