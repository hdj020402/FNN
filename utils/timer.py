"""Timer utility for tracking elapsed time."""
import time


class Timer:
    """Simple timer for measuring elapsed time.

    Usage::

        timer = Timer()
        timer.start()
        # ... do work ...
        timer.end()
        days, hours, minutes, seconds = timer.get_tot_time()
    """

    def __init__(self) -> None:
        self.start_time: float | None = None
        self.end_time: float | None = None

    def start(self) -> None:
        self.start_time = time.perf_counter()

    def end(self) -> None:
        self.end_time = time.perf_counter()

    def get_tot_time(self) -> tuple[int, int, int, float]:
        """Return total elapsed time as (days, hours, minutes, seconds)."""
        if self.start_time is None or self.end_time is None:
            return (-1, -1, -1, -1.0)
        tot_time = self.end_time - self.start_time
        return Timer.convert_time(tot_time)

    def get_average_time(self, divisor: int) -> tuple[int, int, int, float]:
        """Return average time per iteration as (days, hours, minutes, seconds)."""
        if divisor <= 0:
            raise ValueError("Divisor must be greater than 0.")
        if self.start_time is None or self.end_time is None:
            return (-1, -1, -1, -1.0)
        avg_time = (self.end_time - self.start_time) / divisor
        return Timer.convert_time(avg_time)

    @staticmethod
    def convert_time(seconds: float) -> tuple[int, int, int, float]:
        """Convert seconds to (days, hours, minutes, seconds)."""
        days, remainder = divmod(seconds, 86400)
        hours, remainder = divmod(remainder, 3600)
        minutes, seconds = divmod(remainder, 60)
        return int(days), int(hours), int(minutes), seconds
