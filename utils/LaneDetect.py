import numpy as np
import cv2

# Global parameters

# Gaussian smoothing
kernel_size = 3

# Canny Edge Detector
low_threshold = 110
high_threshold = 150

# Region-of-interest vertices
# We want a trapezoid shape, with bottom edge at the bottom of the image
trap_bottom_width = 0.85  # width of bottom edge of trapezoid, expressed as percentage of image width
trap_top_width = 0.07  # ditto for top edge of trapezoid
trap_height = 0.3  # height of the trapezoid expressed as percentage of image height

# Hough Transform
rho = 2  # distance resolution in pixels of the Hough grid
theta = 1 * np.pi / 180  # angular resolution in radians of the Hough grid
threshold = 15  # minimum number of votes (intersections in Hough grid cell)
min_line_len = 10  # minimum number of pixels making up a line
max_line_gap = 20  # maximum gap in pixels between connectable line segments


# Helper functions
def grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def canny(img, low_threshold, high_threshold):
    return cv2.Canny(img, low_threshold, high_threshold)


def gaussian_blur(img, kernel_size):
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len,
                            maxLineGap=max_line_gap)
    line_img = np.zeros((*img.shape, 3), dtype=np.uint8)  # 3-channel RGB image
    draw_lines(line_img, lines)
    return line_img


def weighted_img(img, initial_img, α=1, β=0.4, λ=0.):
    return cv2.addWeighted(initial_img, α, img, β, λ)

def display_lines(img):
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len,
                            maxLineGap=max_line_gap)

    line_arr = np.squeeze(lines)
    slope_degree = (np.arctan2(line_arr[:, 1] - line_arr[:, 3], line_arr[:, 0] - line_arr[:, 2]) * 180) / np.pi

    # 수평 기울기 제한
    line_arr = line_arr[np.abs(slope_degree) < 160]
    slope_degree = slope_degree[np.abs(slope_degree) < 160]
    # 수직 기울기 제한
    line_arr = line_arr[np.abs(slope_degree) > 120]
    slope_degree = slope_degree[np.abs(slope_degree) > 120]


    # 필터링된 직선 버리기
    L_lines, R_lines = line_arr[(slope_degree > 0),:], line_arr[(slope_degree < 0), :]
    L_line_x,L_line_y,R_line_x,R_line_y = [],[],[],[]

    for i in L_lines:
        L_line_x.append(i[0]),L_line_x.append(i[2])
        L_line_y.append(i[1]),L_line_y.append(i[3])

    for i in R_lines:
        R_line_x.append(i[0]),R_line_x.append(i[2])
        R_line_y.append(i[1]),R_line_y.append(i[3])

    if len(L_line_x) > 0:
        left_m, left_b = np.polyfit(L_line_x, L_line_y, 1)
    else:
        left_m, left_b=1,1
    if len(R_line_x) > 0:
        right_m, right_b = np.polyfit(R_line_x, R_line_y, 1)
    else:
        right_m, right_b = 1,1

    y1 = img.shape[0]
    y2 = img.shape[0] * (1 - trap_height)

    right_x1 = (y1 - right_b) / right_m
    right_x2 = (y2 - right_b) / right_m

    left_x1 = (y1 - left_b) / left_m
    left_x2 = (y2 - left_b) / left_m

    # Convert calculated end points from float to int
    y1 = int(y1)
    y2 = int(y2)
    right_x1 = int(right_x1)
    right_x2 = int(right_x2)
    left_x1 = int(left_x1)
    left_x2 = int(left_x2)

    temp = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    # L_lines, R_lines = L_lines[:, None], R_lines[:, None]



    cv2.line(temp, (right_x1, y1), (right_x2, y2), (0, 255, 0), 6)
    cv2.line(temp, (left_x1, y1), (left_x2, y2), (0, 255, 0), 6)

    vertices = np.int32([[(right_x1, y1), (right_x2, y2), (left_x2, y2), (left_x1, y1)]])
    cv2.fillPoly(temp, vertices, (0, 255, 0))
    return temp

def draw_lines(img,lines): # 선 그리기
    for x in lines:
        x1, y1, x2, y2 = x[0]
        cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 5)


def roi(img,vertices):
    mask=np.zeros_like(img)
    vertices=np.int32(vertices)
    cv2.fillPoly(mask,vertices,(255,255,255))
    ROI_image=cv2.bitwise_and(img,mask)
    return ROI_image

# Main script

def LaneDetect(img):

    height, width = img.shape[:2]

    vertices = np.float32(
        [[(200, height),
          (width / 2 - 50, height / 2 + 170),
          (width / 2 + 50, height / 2 + 170),
          (width - 200, height)]])

    gray_img = grayscale(img)
    gaussian_blur_img = gaussian_blur(gray_img, kernel_size=9)
    canny_img = canny(gaussian_blur_img, low_threshold, high_threshold)
    roi_img = roi(canny_img, vertices)

    lane_img = display_lines(roi_img)
    result = weighted_img(lane_img, img)



    return  result



