import numpy as np
import cv2


# preprocessing from road detection
def preprocess(frame):
    grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(grayscale, (5, 5), 1)
    canny_edge = cv2.Canny(blur, 50, 150)

    return canny_edge


def region(frame):
    height, width = frame.shape
    triangle1 = np.array([[(0, height), (0, 600), (960, 500), (width, 620), (width, height)]])
    triangle2 = np.array([[(460, height), (460, 790), (1510, 790), (1510, height)]])

    mask = np.zeros_like(frame)

    mask = cv2.fillPoly(mask, triangle1, 255)
    mask = cv2.bitwise_and(frame, mask)
    mask = cv2.fillPoly(mask, triangle2, 0)

    return mask


def hough(frame):
    rho = 1
    theta = np.pi / 180
    threshold = 20
    minlength = 20
    maxlegth = 500
    lines = cv2.HoughLinesP(frame, rho=rho, theta=theta, threshold=threshold, minLineLength=minlength,
                            maxLineGap=maxlegth)

    return lines


def average(frame, lines):
    left = []
    right = []
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        parameters = np.polyfit((x1, x2), (y1, y2), 1)
        slope = parameters[0]
        y_int = parameters[1]
        if slope < 0:
            left.append((slope, y_int))
        else:
            right.append((slope, y_int))

    right_avg = np.average(right, axis=0)
    left_avg = np.average(left, axis=0)
    left_line = make_points(frame, left_avg)
    right_line = make_points(frame, right_avg)

    return np.array([left_line, right_line])


def make_points(frame, average):
    try:
        slope, y_int = average
    except TypeError:
        slope, y_int = np.array([0, 0])
    y1 = frame.shape[0]
    y2 = int(y1 * (3 / 5))
    x1 = int((y1 - y_int) // slope)
    x2 = int((y2 - y_int) // slope)

    return np.array([x1, y1, x2, y2])


def average_slope_intercept(lines):
    """
       Find the slope and intercept of the left and right lanes of each image.
       Parameters:
           lines: output from Hough Transform
       """
    left_lines = []
    left_weights = []
    right_lines = []
    right_weights = []

    for line in lines:
        for x1, y1, x2, y2 in line:
            if x1 == x2:
                continue

            slope = (y2 - y1) / (x2 - x1)
            intercept = y1 - (slope * x1)
            length = np.sqrt(((y2 - y1) ** 2) + ((x2 - x1) ** 2))

            if slope < 0:
                left_lines.append((slope, intercept))
                left_weights.append((length))
            else:
                right_lines.append((slope, intercept))
                right_weights.append((length))

    left_lane = np.dot(left_weights, left_lines) / np.sum(left_weights) if len(left_weights) > 0 else None
    right_lane = np.dot(right_weights, right_lines) / np.sum(right_weights) if len(right_weights) > 0 else None
    return left_lane, right_lane


def pixel_points(y1, y2, line):
    """
        Converts the slope and intercept of each line into pixel points.
            Parameters:
                y1: y-value of the line's starting point.
                y2: y-value of the line's end point.
                line: The slope and intercept of the line.
        """
    if line is None:
        return None
    slope, intercept = line
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    y1 = int(y1)
    y2 = int(y2)
    return ((x1, x2), (x2, y2))


def lane_lines(image, lines):
    """
        Create full lenght lines from pixel points.
            Parameters:
                image: The input test image.
                lines: The output lines from Hough Transform.
        """
    left_lane, right_lane = average_slope_intercept(lines)
    y1 = image.shape[0]
    y2 = y1 * 3/5
    left_line = pixel_points(y1, y2, left_lane)
    right_line = pixel_points(y1, y2, right_lane)
    return left_line, right_line


def display_lines(frame, lines):
    lines_image = np.zeros_like(frame)
    if lines is not None:
        for line in lines:
            try:
                cv2.line(lines_image, *line, (255, 0, 0), 10)
            except cv2.error:
                continue
    return lines_image


frame = cv2.imread('gtpic1.png')

canny = preprocess(frame)
area = region(canny)
points = hough(area)
avg_lines = lane_lines(frame, points)
lanesb = display_lines(frame, avg_lines)
lanes = cv2.addWeighted(frame, 0.8, lanesb, 1, 1)
cv2.imshow('video', frame)
cv2.imshow('video1', area)
cv2.imshow('result', lanes)
key = cv2.waitKey(0)
