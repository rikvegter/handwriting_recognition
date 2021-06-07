import os
from dataclasses import dataclass
from simple_parsing import ArgumentParser, field
from simple_parsing.helpers.fields import flag


def main():
    pass


@dataclass
class General:
    """General options for the tool"""

    # The path of the image to process
    input_path: str = field(alias="-i")

    # The path to save output to
    output_path: str = field(default="./", alias="-o")

    # Save the results of intermediate steps to a debug directory in the output
    # path
    debug: bool = flag(default=False, alias="-d")


if __name__ == "__main__":
    # CLI argument parsing
    parser = ArgumentParser()
    parser.add_arguments(General, dest="general_options")

    args = parser.parse_args()

    main()
