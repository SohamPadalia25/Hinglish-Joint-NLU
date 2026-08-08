"""Start the backend API.

    python run_api.py [--host 127.0.0.1] [--port 8000]

Equivalent to:
    python -m uvicorn backend.api:app --host 127.0.0.1 --port 8000
"""

import argparse

import uvicorn


def main() -> None:
    parser = argparse.ArgumentParser(description="Hinglish Joint NLU API")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Auto-reload on code changes (reloads the 2.2 GB checkpoint each time)",
    )
    args = parser.parse_args()

    # One worker only: each would load its own ~2.2 GB copy of the checkpoint.
    uvicorn.run("backend.api:app", host=args.host, port=args.port, reload=args.reload, workers=1)


if __name__ == "__main__":
    main()
