import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def log_runtime(start_time: float,
                end_time: float,
                task: str) -> None:
    elapsed_time = end_time - start_time
    hours, rem = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(rem, 60)
    logging.info(f"Time Taken for {task}: {int(hours):02}:{int(minutes):02}:{seconds:05.2f}")