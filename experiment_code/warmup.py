import multiprocessing
import time
import os
import random

WARMUP_SECONDS = 120

def work(stop_time):
    value = 1

    while time.monotonic() < stop_time:
        value = (value * random.uniform(1, 10)) % random.uniform(1, 10)
        time.sleep(1)


def main():
    print("Starting warmup", flush=True)

    stop_time = time.monotonic() + WARMUP_SECONDS
    #warmup all the cpus
    workers = [
        multiprocessing.Process(
            target=work,
            args=(stop_time,)
        )
        for _ in range(os.cpu_count() or 1)
    ]

    for worker in workers:
        worker.start()

    for worker in workers:
        worker.join()

    print("Warm-up complete.", flush=True)


if __name__ == "__main__":
    main()