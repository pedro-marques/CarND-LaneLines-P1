#**Finding Lane Lines on the Road**

##Writeup Pedro Marques

---

**Finding Lane Lines on the Road**

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report


[//]: # (Image References)

[image1]: ./test_images/solidWhiteRight_with_lanes.jpg "Lane Lines identified"

---

### Reflection

###1. Describe your pipeline. As part of the description, explain how you modified the draw_lines() function.

My pipeline consisted of these steps. First, I stored the image size (height and width) so that I could easily find center while building my trapeze (region of interest). I did not convert the image to grayscale because on the challenge video it would not detect a segment of line.
Next I apply the Gaussian filtering to give me the gradient of the grayed image. With this gradient image I can then use the Canny edge detection algorithm to identify the edges on the given image, by defining a region of interest and a low and high threshold, so that we don't pick up every edge on the image.
After having all the edges in the desired region then I used Hough Transform to identify the lines, based on a specific set of parameters.
Next I just had to draw the lines onto the image.

In order to draw a single line on the left and right lanes, I modified the draw_lines() function by calculating the slope for each line, and grouping the coordinates from each line based on their slope and position on the screen, if the slope was positive and its x axis coordinates where to right of the screen, it belonged to the right lane and vice-versa.
After having all of the coordinates from one line and the other, I would then run linear regression to find the best line for all those coordinates, using the polyfit function, it would return the slope of that line and the point where x is 0 (b) in the function y=mx+b.
Now I just need to draw the line -- It is a requirement that the line starts from the bottom of the screen, so I already have my y1, has I have a lot of coordinates from the left and right line I just needed the topmost (x2,y2) point to find the x1 coordinate and have the (x1,y1) point to draw the line.

![alt text][image1]


###2. Identify potential shortcomings with your current pipeline


One potential shortcoming would be what would happen when/if the car goes downhill, I am unsure as if the trapeze I have designed as the region of interest would still be pointing in the road. I need to test it.

Another one is that currently I have not managed to make the lines steady.


###3. Suggest possible improvements to your pipeline

A possible improvement would be to have steadier lane lines, my lane lines are currently flickering a lot.
