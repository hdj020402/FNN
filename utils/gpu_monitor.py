"""GPU monitoring via nvidia-smi."""
import subprocess
import logging
import time
import threading


class GPUMonitor:
    """Periodically log GPU stats (utilization, memory, temperature).

    Args:
        logger: Logger to write GPU stats to.
        interval: Seconds between logging (default 60).
    """

    def __init__(self, logger: logging.Logger, interval: int = 60) -> None:
        self.logger = logger
        self.interval = interval
        self.running = False

    def gpu_monitor(self) -> None:
        """Background thread target — polls nvidia-smi."""
        while self.running:
            try:
                result = subprocess.run(
                    [
                        'nvidia-smi',
                        '--query-gpu=utilization.gpu,memory.total,memory.used,temperature.gpu',
                        '--format=csv,noheader',
                    ],
                    stdout=subprocess.PIPE,
                )
                if result.returncode == 0:
                    output = result.stdout.decode('utf-8')
                    util, mem_tot, mem_used, temp = output.replace('\n', '').split(', ')
                    self.logger.info(
                        f'utilization: {util}; memory.used: {mem_used}; '
                        f'memory.total: {mem_tot}; temperature: {temp}'
                    )
                else:
                    self.logger.error(f"Error running nvidia-smi: {result.stderr}")
            except Exception as e:
                self.logger.error(f"Exception occurred while trying to run nvidia-smi: {e}")
            time.sleep(self.interval)

    def start(self) -> None:
        """Start the GPU monitor thread."""
        self.running = True
        self.thread = threading.Thread(target=self.gpu_monitor)
        self.thread.daemon = True
        self.thread.start()

    def stop(self) -> None:
        """Stop the GPU monitor thread."""
        self.running = False
        self.thread.join(timeout=self.interval + 5)
