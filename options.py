from dataclasses import dataclass
from typing import Optional

from simple_parsing import ArgumentParser, field
from simple_parsing.helpers.fields import choice, flag

@dataclass
class GeneralOptions:
    """General options for the tool"""

    # The path of the images to process
    input_path: str = field(alias="-i")

    # The path to save output to
    output_path: str = field(default="results", alias="-o")

    # Whether to stop the process after a given step. Useful for debugging.

    # [Step 1]: Line segmentation
    # [Step 2]: Character segmentation
    # [Step 3]: Character recognition
    # [Step 4]: Style classification (final step)
    # [Step 5]: Just keep going until the end.
    stop_after: int = choice(1, 2, 3, 4, 5, default=5, action="store", type=int)

    # Save the results of intermediate steps to a debug directory in the output
    # path
    debug: bool = flag(default=False, alias="-d")

    # Process only a single image
    single: bool = flag(default=False, alias="-s")
