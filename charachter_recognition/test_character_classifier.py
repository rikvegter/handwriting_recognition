from unittest import TestCase

import numpy as np
from PIL import Image

from charachter_recognition.character_classifier import CharacterClassifier


class TestCharacterClassifier(TestCase):
    classifier = None

    def setUp(self):
        if self.classifier is None:
            self.classifier = CharacterClassifier("data/classification_model/")
        self.test_image: str = "data/original/Alef/navis-QIrug-Qumran_extr09_0001-line-008-y1=400-y2=515-zone-HUMAN-x=1650-y=0049-w=0035-h=0042-ybas=0027-nink=631-segm=COCOS5cocos.pgm"
        self.test_image2: str = "data/original/Tet/navis-QIrug-Qumran_extr09_0310-line-022-y1=927-y2=1051-zone-HUMAN-x=2593-y=0024-w=0035-h=0065-ybas=0100-nink=777-segm=COCOS5cocos.pgm"

        self.test_image_img_gs: Image.Image = Image.open(self.test_image).convert(mode="L")
        self.test_image2_img_rgb: Image.Image = Image.open(self.test_image2).convert(mode="P")

        self.test_image_np_gs: np.ndarray = np.asarray(Image.open(self.test_image).convert(mode="L"), dtype=np.uint8)
        self.test_image2_np_rgb: np.ndarray = np.asarray(Image.open(self.test_image2).convert(mode="P"), dtype=np.uint8)


class TestCharacterClassifierSingleImage(TestCharacterClassifier):

    def test_classify_image_name(self):
        assert self.classifier.classify_image(self.test_image) == 11
        assert self.classifier.classify_image(self.test_image2) == 5

    def test_classify_image_image(self):
        # Test greyscale images loaded with PIL
        assert self.classifier.classify_image(self.test_image_img_gs) == 11
        # Test RGB images loaded with PIL
        assert self.classifier.classify_image(self.test_image2_img_rgb) == 5

    def test_classify_image_numpy(self):
        # Test greyscale images loaded with PIL converted to Numpy arrays
        assert self.classifier.classify_image(self.test_image_np_gs) == 11
        # Test RGB images loaded with PIL converted to Numpy arrays
        assert self.classifier.classify_image(self.test_image2_np_rgb) == 5


class TestCharacterClassifierMultipleImages(TestCharacterClassifier):
    def test_classify_images_names(self):
        np.testing.assert_array_equal([11, 5], self.classifier.classify_images([self.test_image, self.test_image2]))
        np.testing.assert_array_equal([5, 11], self.classifier.classify_images([self.test_image2, self.test_image]))

    def test_classify_images_images(self):
        np.testing.assert_array_equal([11, 5], self.classifier.classify_images([self.test_image_img_gs,
                                                                                self.test_image2_img_rgb]))

    def test_classify_images_numpy(self):
        np.testing.assert_array_equal([11, 5], self.classifier.classify_images([self.test_image_np_gs,
                                                                                self.test_image2_np_rgb]))

    def test_classify_images_mixed_types(self):
        np.testing.assert_array_equal([11, 11, 11, 5, 5, 5], self.classifier.classify_images(
            [self.test_image_img_gs, self.test_image_img_gs, self.test_image_np_gs,
             self.test_image2_img_rgb, self.test_image2_img_rgb, self.test_image2_np_rgb]))

    def test_classify_images_empty_input(self):
        np.testing.assert_array_equal([], self.classifier.classify_images([]))

    def test_classify_images_single_array(self):
        with self.assertRaises(ValueError):
            self.classifier.classify_images(self.test_image_np_gs)
        with self.assertRaises(ValueError):
            self.classifier.classify_images(self.test_image2_np_rgb)
        np.testing.assert_array_equal([11],
                                      self.classifier.classify_images(np.expand_dims(self.test_image_np_gs, axis=0)))
        np.testing.assert_array_equal([5],
                                      self.classifier.classify_images(np.expand_dims(self.test_image2_np_rgb, axis=0)))
