import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from scipy.signal import savgol_filter
from scipy.signal import argrelextrema
import numpy as np
PATH = 'data/image-data/'
MINIMA_THRESHOLD = 900000


def crop_image(area, image):
    cropped_image = image.crop(area)
    import pdb; pdb.set_trace()
    return cropped_image


def find_line_segmentations(vert_projection_profile):
    upper_line = 0
    lower_line = 0
    line_segments = []
    in_line = 0
    for i in range(0, len(vert_projection_profile)):
        #Start of a peak
        if vert_projection_profile[i] < (MINIMA_THRESHOLD + 1) and in_line == 0:
            in_line = 1
            upper_line = i
        if vert_projection_profile[i] == (MINIMA_THRESHOLD + 1) and in_line == 1:
            lower_line = i
            in_line = 0

            line_segments.append([upper_line, lower_line])

    arr = np.array(line_segments)
    return arr
    #flatten_line_segments = arr.flatten()


def get_image(im_path):
    im = Image.open(im_path)
    return im

#Draw line segmentations
def draw_line_segmentations(image, local_minima):
    draw = ImageDraw.Draw(image)
    width, height = image.size
    for minima in local_minima:
        shape = [(0, minima), (width, minima)]
        draw.line(shape, fill = 'red', width = 0)
    return draw

#Returns the summed pixel values per pixel height
def get_vertical_projection_profile(image):
    width, height = image.size
    vert_profile = []
    for i in range(0, height):
        sum_pixel_value = 0
        for j in range(0, width):
            sum_pixel_value += image.getpixel((j, i))
        vert_profile.append(sum_pixel_value)

    #Smooth the profile to avoid getting small local minima
    smooth_profile = savgol_filter(vert_profile, 51, 3)
    return smooth_profile

def create_cropped_images(line_segmentations, image):
    for lines in line_segmentations:
        #area = left, upper, right, lower
        left = 1
        top = lines[0]
        right = image.size[0]
        bottom = lines[1]
        cropped_image = image.crop((left, top, right, bottom))
        cropped_image.show()
        
        break


def main():
    im_path = PATH + 'P106-Fg002-R-C01-R01-binarized.jpg'
    image = get_image(im_path)
    vert_projection_profile = get_vertical_projection_profile(image)

    #Cap the ink per row in order to prevent flunctuations
    vert_projection_profile = np.where(vert_projection_profile > MINIMA_THRESHOLD, MINIMA_THRESHOLD+1, vert_projection_profile)

    line_segmentations = find_line_segmentations(vert_projection_profile)
    create_cropped_images(line_segmentations, image)


    #line_segmented_image = draw_line_segmentations(image, local_minima)
    image.show()

    plt.plot(vert_projection_profile)

    plt.show()



if __name__ == "__main__":
    main()
