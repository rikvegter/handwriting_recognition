from enum import Enum


class CharacterSegmentationMethod(Enum):
    PROJECTION_PROFILE = 1,
    CONNECTED_COMPONENTS = 2,
    THINNING = 3

    def __str__(self) -> str:
        return str.lower(self.name)

    def __repr__(self):
        return str(self)

    @staticmethod
    def argparse(s):
        try:
            return CharacterSegmentationMethod[s.upper()]
        except KeyError:
            return s