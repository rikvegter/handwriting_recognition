from dataclasses import dataclass
from simple_parsing import field

from .character_segmentation_method import CharacterSegmentationMethod


@dataclass
class LineSegmentationOptions:
    """Parameters for line segmentation"""

    option1: str = ""


@dataclass
class CharacterSegmentationOptions:
    """Parameters for character segmentation"""

    # The method to use for character segmentation
    charseg_method: CharacterSegmentationMethod = field(
        default=CharacterSegmentationMethod.PROJECTION_PROFILE,
        choices=list(CharacterSegmentationMethod),
        alias="-c"
    )


@dataclass
class SegmentationOptions:
    """Parameters for the different segmentation steps
    """

    line: LineSegmentationOptions = LineSegmentationOptions(option1="a")

    character: CharacterSegmentationOptions = CharacterSegmentationOptions()
