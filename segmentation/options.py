from dataclasses import dataclass


@dataclass
class LineSegmentationOptions:
    """Parameters for line segmentation"""

    option1: str = ""


@dataclass
class CharacterSegmentationOptions:
    """Parameters for character segmentation"""

    option2: str = ""
    pass


@dataclass
class SegmentationOptions:
    """Parameters for the different segmentation steps
    """

    line: LineSegmentationOptions = LineSegmentationOptions(option1="a")

    character: CharacterSegmentationOptions = CharacterSegmentationOptions(
        option2="b")
