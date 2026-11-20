# src/hometheater/__main__.py
#
# This file makes the package directly executable via:
#   uv run python -m hometheater
#
# Python looks for __main__.py when a package is invoked with -m.
# Without it, Python finds the package but has no entry point to run.

from hometheater.main import main

if __name__ == "__main__":
    main()
