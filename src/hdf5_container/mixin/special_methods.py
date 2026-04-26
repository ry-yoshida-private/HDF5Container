from __future__ import annotations

from types import TracebackType
from typing import Iterator

from .protocols import HDF5ContainerProtocol


class SpecialMethodsMixin:
    """Special method implementations for container ergonomics."""

    def __enter__(self: HDF5ContainerProtocol) -> HDF5ContainerProtocol:
        """Return the container for context manager usage.

        Returns
        -------
        HDF5ContainerProtocol
            The current container instance.
        """
        return self

    def __exit__(
        self: HDF5ContainerProtocol,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Close resources on context manager exit.

        Parameters
        ----------
        exc_type : type[BaseException] | None
            Raised exception type, if an exception occurred.
        exc_val : BaseException | None
            Raised exception value, if an exception occurred.
        exc_tb : TracebackType | None
            Raised exception traceback, if an exception occurred.
        """
        self.close()

    def __iter__(self: HDF5ContainerProtocol) -> Iterator[str]:
        """Iterate over keys in the current HDF5 group.

        Returns
        -------
        Iterator[str]
            Key iterator for datasets/groups under the current node.
        """
        return iter(self.data.keys())
