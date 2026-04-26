from dataclasses import dataclass


@dataclass
class FlushCounter:
    """
    Shared mutable counter for periodic flushing across subcontainers.
    
    Parameters:
    ----------
    value: int
        The current value of the counter.
    """

    value: int = 0

    def increment(self) -> int:
        """
        Increment the counter and return the updated value.

        Returns:
        -------
        int
            The updated value of the counter.
        """
        self.value += 1
        return self.value

    def is_flush_timing(self, flush_interval: int) -> bool:
        """
        Return whether current counter reached the flush boundary.

        Parameters:
        ----------
        flush_interval: int
            The interval at which to flush the counter.

        Returns:
        -------
        bool
            True if the counter reached the flush boundary, False otherwise.
        """
        return self.value % flush_interval == 0
