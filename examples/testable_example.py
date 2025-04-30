"""A testable example."""

import logging
import os
from pathlib import Path

import example_1
import example_2
import example_3
import example_4
import example_5
import example_6
import example_7
import example_8


def main() -> None:
    """Run the example."""
    init_dir = Path.cwd()
    os.chdir("examples/")

    try:
        example_1.main()
        example_2.main()
        example_3.main()
        example_4.main()
        example_5.main()
        example_6.main()
        example_7.main()
        example_8.main()
        logging.info("all examples ran, at least!")

    finally:
        os.chdir(init_dir)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    main()
