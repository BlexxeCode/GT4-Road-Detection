import numpy as np
import cv2

video = cv2.VideoCapture('2024-02-11 17-01-35.mkv')


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
    rho = 2
    theta = np.pi / 180
    threshold = 150
    minlength = 20
    maxlegth = 100
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


def display_lines(frame, lines):
    lines_image = np.zeros_like(frame)
    if lines is not None:
        for line in lines:
            try:
                x1, y1, x2, y2 = line
                cv2.line(lines_image, (x1, y1), (x2, y2), (255, 0, 0), 10)
            except cv2.error:
                continue
    return lines_image


while True:
    ret, frames = video.read()

    canny = preprocess(frames)
    area = region(canny)
    points = hough(area)
    avg_lines = average(area, points)
    lanesb = display_lines(frames, avg_lines)
    lanes = cv2.addWeighted(frames, 0.8, lanesb, 1, 1)
    cv2.imshow('video', frames)
    cv2.imshow('video1', area)
    cv2.imshow('result', lanes)
    key = cv2.waitKey(11)
    if key == 13:
        break
