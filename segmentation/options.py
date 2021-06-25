from dataclasses import dataclass
from simple_parsing import field
from simple_parsing.helpers.fields import choice


@dataclass
class LineSegmentationOptions:
    """Parameters for line segmentation"""

    # No options yet
    empty: str = ""


@dataclass
class CharacterSegmentationOptions:
    """Parameters for character segmentation"""

    # No options yet
    empty: str = ""


@dataclass
class SegmentationOptions:
    """Parameters for the different segmentation steps
    """

    line: LineSegmentationOptions = LineSegmentationOptions()

    character: CharacterSegmentationOptions = CharacterSegmentationOptions()
