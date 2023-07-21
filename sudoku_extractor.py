import cv2 as cv
import numpy as np

img_path = "./Sudoku_Boards/sudoku_1.jpg"

def read_img(img_path):
    sudoku = cv.imread(img_path)
    if len(sudoku.shape) == 3:
        sudoku = cv.cvtColor(sudoku.copy(), cv.COLOR_BGR2GRAY)

    return sudoku

def preprocess_board(sudoku):
    sudoku_processed = cv.GaussianBlur(sudoku.copy(), (9, 9), 0)
    # sudoku_processed = cv.equalizeHist(sudoku_processed)
    sudoku_processed = cv.adaptiveThreshold(sudoku_processed, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2)
    sudoku_processed = cv.bitwise_not(sudoku_processed)
    kernel = np.ones((3,3), dtype=np.uint8)
    sudoku_processed = cv.dilate(sudoku_processed, kernel, iterations=1)
    sudoku_processed = cv.erode(sudoku_processed, kernel, iterations=2)

    return sudoku_processed

def get_sudoku_outline(sudoku_processed):
    contours, _ = cv.findContours(sudoku_processed, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    sorted_contours = sorted(contours, key=cv.contourArea, reverse=True)
    polygon = sorted_contours[0]

    return polygon

def get_corner_points(outline):
    bottom_right, _ = max(enumerate([pt[0][0] + pt[0][1] for pt in
                      outline]), key=lambda x: x[1])
    top_left, _ = min(enumerate([pt[0][0] + pt[0][1] for pt in
                    outline]), key=lambda x: x[1])
    bottom_left, _ = min(enumerate([pt[0][0] - pt[0][1] for pt in
                        outline]), key=lambda x: x[1])
    top_right, _ = max(enumerate([pt[0][0] - pt[0][1] for pt in
                    outline]), key=lambda x: x[1])
    
    return outline[top_left].ravel().tolist(), outline[top_right].ravel().tolist(), outline[bottom_left].ravel().tolist(), outline[bottom_right].ravel().tolist()

def distance(pt1, pt2):
    return int(abs(np.hypot(pt1[0] - pt2[0], pt1[1] - pt2[1])))

def warp(img, corner_points):
    points = {
        "top_left": corner_points[0],
        "top_right": corner_points[1],
        "bottom_left": corner_points[2],
        "bottom_right": corner_points[3]
    }

    width = max(distance(points["top_left"], points["top_right"]), distance(points["bottom_left"], points["bottom_right"]))
    height = max(distance(points["top_left"], points["bottom_left"]), distance(points["top_right"], points["bottom_right"]))

    print(width, height)
    pts1 = np.float32([points["top_left"], points["top_right"], points["bottom_left"], points["bottom_right"]])
    pts2 = np.float32([[0,0], [width, 0], [0, height], [width, height]])

    mat = cv.getPerspectiveTransform(pts1, pts2, )

    output_size = (width, height)

    warped = cv.warpPerspective(img, mat, output_size)

    return warped
def show_image(img, contour=None, corner_points = []):
    if type(contour) == np.ndarray:
        img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
        cv.drawContours(img, [contour], -1, (255, 0, 0), 2)
    
    if corner_points != None:
        for pt in corner_points:
            cv.circle(img, pt, 10, (0, 0, 255), -1)
    cv.imshow("Sudoku Board", img)
    cv.waitKey(0)
    cv.destroyAllWindows()

sudoku = read_img(img_path)
sudoku_processed = preprocess_board(sudoku)
outline = get_sudoku_outline(sudoku_processed)
corner_points = get_corner_points(outline)
warped = warp(sudoku.copy(), corner_points=corner_points)
show_image(warped)
show_image(sudoku_processed)
show_image(sudoku.copy(), contour=outline, corner_points=corner_points)
show_image(sudoku.copy())