import argparse
import time
from sdb.metrics import start_metrics_server


def main() -> None:
    parser = argparse.ArgumentParser(description="Start Prometheus metrics server")
    parser.add_argument("--port", type=int, default=None, help="Port to bind")
    args = parser.parse_args()
    start_metrics_server(args.port)
    print("Metrics server running. Press Ctrl-C to stop.")
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
