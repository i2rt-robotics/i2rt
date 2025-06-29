import logging
import time


class RateRecorder:
    def __init__(self, name: str | None = None, report_interval: float = 10):
        """
        Initialize the rate recorder.
        :param report_interval: Interval in seconds at which the rate should be reported.
        """
        self.report_interval = report_interval
        self.start_time = None
        self.last_report_time = None
        self.iteration_count = 0
        self.message = ""
        self.name = name

    def __enter__(self):
        return self.start()

    def start(self) -> None:
        # Record the start time and initialize variables when the context manager is entered
        self.start_time = time.time()
        self.last_report_time = self.start_time
        self.iteration_count = 0
        return self

    def __exit__(self, exc_type, exc_value, traceback):  # noqa: ANN001
        # Final report when exiting the context manager
        self._report_rate()

    def _report_rate(self) -> None:
        # Calculate and print the rate of iterations per second
        assert self.start_time is not None, "RateRecorder must be started before reporting."
        elapsed_time = time.time() - self.start_time
        rate = self.iteration_count / elapsed_time if elapsed_time > 0 else 0
        logging.info(
            f"{self.name} Total rate: {rate:.2f} iterations per second over {elapsed_time:.2f} seconds. User message: {self.message}"
        )

    def track(self, message: str = "") -> None:
        """
        This method should be called once every loop iteration. It tracks and reports the rate
        every `report_interval` seconds.
        """
        self.iteration_count += 1
        current_time = time.time()
        self.message = message
        # Report the rate every `self.report_interval` seconds
        assert self.start_time is not None and self.last_report_time is not None
        if current_time - self.last_report_time >= self.report_interval:
            self.last_report_time = current_time
            self._report_rate()
            # reset the iteration count
            self.iteration_count = 0
            self.start_time = time.time()


def override_log_level(level: int = logging.INFO) -> None:
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(level=level)
