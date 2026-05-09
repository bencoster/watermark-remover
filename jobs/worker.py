"""Background worker for video processing jobs."""

_running = False


def start_worker():
    global _running
    _running = True


def stop_worker():
    global _running
    _running = False
