# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
from IPython.display import HTML

import math
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2

def read_image(filename):
    return mpimg.imread(filename)

def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices):
    """
    Applies an image mask.

    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)

    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    #filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def draw_lines(img, lines, color=[255, 0, 0], thickness=10):
    """
    NOTE: this is the function you might want to use as a starting point once you want to
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).

    Think about things like separating line segments by their
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of
    the lines and extrapolate to the top and bottom of the lane.

    This function draws `lines` with `color` and `thickness`.
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """

    # Register the half of the screen to classify coordinates based on their x position, larger than middle of screen belong to the right lane and, smaller belong to the left lane
    height = img.shape[0]
    width = img.shape[1]
    image_x_center = width / 2

    # Store all the coordinates on y and x axis for the left line
    left_line_coordinates_y_axis = []
    left_line_coordinates_x_axis = []

    # Store all the coordinates on y and x axis for the right line
    right_line_coordinates_y_axis = []
    right_line_coordinates_x_axis = []

    for line in lines:
        for x1,y1,x2,y2 in line:
            if x2 - x1 == 0.:  # for perpendicular lines give an infinite scope
                slope = 9999.
            else:
                slope = (y2 - y1) / (x2 - x1)
            if abs(slope) > 0.5:
                if slope > 0 and x1 > image_x_center and x2 > image_x_center:
                    right_line_coordinates_x_axis.append(x1)
                    right_line_coordinates_x_axis.append(x2)
                    right_line_coordinates_y_axis.append(y1)
                    right_line_coordinates_y_axis.append(y2)
                elif slope < 0 and x1 < image_x_center and x2 < image_x_center:
                    left_line_coordinates_x_axis.append(x1)
                    left_line_coordinates_x_axis.append(x2)
                    left_line_coordinates_y_axis.append(y1)
                    left_line_coordinates_y_axis.append(y2)

    # If there are coordinates on the right line
    if right_line_coordinates_y_axis:
        right_m, right_b = np.polyfit(right_line_coordinates_x_axis, right_line_coordinates_y_axis, 1)

        # Find the top most y coordinate - the higher in the frame, the lower its value - and its correspondent x coordinate
        right_y2 = min(right_line_coordinates_y_axis)
        array_index = right_line_coordinates_y_axis.index(min(right_line_coordinates_y_axis))
        right_x2 = right_line_coordinates_x_axis[array_index]

        # slope or left_m = (y2-y1)/(x2-x1) then x1 = (y1 - y2) / m + x2
        right_x1 = ((height - right_y2) / right_m + right_x2)
        right_x1 = int(right_x1)

        # Draw the line starting from the top to the bottom
        cv2.line(img, (right_x2,right_y2), (right_x1,height),color,thickness)

    # If there are coordinates on the left line
    if left_line_coordinates_y_axis:
        left_m, left_b = np.polyfit(left_line_coordinates_x_axis, left_line_coordinates_y_axis, 1)

        # Find the top most y coordinate - the higher in the frame, the lower its value - and its correspondent x coordinate
        left_y2 = min(left_line_coordinates_y_axis)
        array_index = left_line_coordinates_y_axis.index(min(left_line_coordinates_y_axis))
        left_x2 = left_line_coordinates_x_axis[array_index]

        # slope or left_m = (y2-y1)/(x2-x1) then x1 = (y1 - y2) / m + x2
        left_x1 = ((height - left_y2) / left_m + left_x2)
        left_x1 = int(left_x1)

        # Draw the line starting from the top to the bottom
        cv2.line(img, (left_x2,left_y2), (left_x1,height),color,thickness)


    #for line in lines:
        #for x1,y1,x2,y2 in line:
            #cv2.line(img, (x1, y1), (x2, y2), color, thickness)

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.

    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), min_line_len,max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img

# Python 3 has support for cool math symbols.

def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.

    `initial_img` should be the image before any processing.

    The result image is computed as follows:

    initial_img * α + img * β + λ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, λ)


def process_image(image):

    # Turn the image into gray
    #grayed_image = grayscale(image)

    # Capture the x and y size
    height = image.shape[0]
    width = image.shape[1]

    image_x_center = width / 2
    image_y_center = height / 2

    # Define kernel size
    kernel_size = 5

    # Apply Gaussian smoothing
    #blur_gray = gaussian_blur(grayed_image,kernel_size)
    blur_gray = gaussian_blur(image,kernel_size)

    # Define parameters for Canny
    low_threshold = 50
    high_threshold = 150

    # Apply Canny
    edges = canny(blur_gray,low_threshold,high_threshold)

    # Next we create our polygon to define which area in the image we are interested in finding lines
    imshape = image.shape
    #
    offset_y = image_y_center * 0.24
    offset_x = image_x_center * 0.10
    #polygon_vertices = np.array([[(0,height),(450, 325), (515, 325), (width,height)]], dtype=np.int32)
    polygon_vertices = np.array([[(0,height),(image_x_center-offset_x, image_y_center+offset_y), (image_x_center+offset_x, image_y_center+offset_y), (width,height)]], dtype=np.int32)
    masked_edges = region_of_interest(edges, polygon_vertices)

    # Define the Hough transform parameters
    rho = 1 # distance resolution in pixels of the Hough grid
    theta = np.pi/180 # angular resolution in radians of the Hough grid
    threshold = 30     # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 45 #minimum number of pixels making up a line
    max_line_gap = 80  # maximum gap in pixels between connectable line segments

    # Create a blank image to draw lines on
    line_image = np.copy(image)*0

    # Run Hough on edge detected image
    #lines = hough_lines(masked_edges, rho, theta, threshold, min_line_length, max_line_gap)
    lines = cv2.HoughLinesP(masked_edges, rho, theta, threshold, np.array([]),
                                min_line_length, max_line_gap)

    # Draw red lines on the blank image
    draw_lines(line_image,lines)

    # Now apply the semi-transparent effect on the lines that were drawn on the blank image
    result = weighted_img(line_image,image)

    return result

def apply_lanes_on_image():
    # Read the original images
    image1 = mpimg.imread('test_images/solidWhiteRight.jpg')
    image2 = mpimg.imread('test_images/solidWhiteCurve.jpg')
    image3 = mpimg.imread('test_images/solidYellowCurve.jpg')
    image4 = mpimg.imread('test_images/solidYellowCurve2.jpg')
    image5 = mpimg.imread('test_images/solidYellowLeft.jpg')
    image6 = mpimg.imread('test_images/whiteCarLaneSwitch.jpg')

    line_image1 = process_image(image1)
    mpimg.imsave("test_images/solidWhiteRight_with_lanes.jpg", line_image1)
    plt.imshow(image1)

    line_image2 = process_image(image2)
    mpimg.imsave("test_images/solidWhiteCurve_with_lanes.jpg", line_image2)
    plt.imshow(image2)

    line_image3 = process_image(image3)
    mpimg.imsave("test_images/solidYellowCurve_with_lanes.jpg", line_image3)
    plt.imshow(image3)

    line_image4 = process_image(image4)
    mpimg.imsave("test_images/solidYellowCurve2_with_lanes.jpg", line_image4)
    plt.imshow(image4)

    line_image5 = process_image(image5)
    mpimg.imsave("test_images/solidYellowLeft_with_lanes.jpg", line_image5)
    plt.imshow(image5)

    line_image6 = process_image(image6)
    mpimg.imsave("test_images/whiteCarLaneSwitch_with_lanes.jpg", line_image6)
    plt.imshow(image6)

def apply_lanes_on_video():
    white_output = 'white.mp4'
    clip1 = VideoFileClip("solidWhiteRight.mp4")
    white_clip = clip1.fl_image(process_image)
    white_clip.write_videofile(white_output, audio=False)

    yellow_output = 'yellow.mp4'
    clip2 = VideoFileClip('solidYellowLeft.mp4')
    yellow_clip = clip2.fl_image(process_image)
    yellow_clip.write_videofile(yellow_output, audio=False)

    challenge_output = 'extra.mp4'
    clip3 = VideoFileClip('challenge.mp4')
    challenge_clip = clip3.fl_image(process_image)
    challenge_clip.write_videofile(challenge_output, audio=False)

apply_lanes_on_image()
apply_lanes_on_video()
