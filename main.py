import sys
import os
import cv2
import numpy as np

def generate_smoothing_kernel(variance):
    pass

class LaneDetection:
    def __init__(self,smoothing_kernel):
        self.smoothing_kernel = smoothing_kernel

    def smooth_image(self):
        pass

    def calculate_gradients(self):
        pass

    def canny_edge_detection(self):
        pass

    def hough_line_transform(self):
        pass

    def polynomial_curve_fitting(self):
        pass

    def output_image(self):
        pass

    def generate_centroid_for_lanes(self):
        pass

    def run_detection(self, path):
        image_raw = cv2.imread(path)



if __name__ == '__main__':
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]

    images = os.listdir(input_dir)

    lane_detector = LaneDetection()

    for img in images:
        lane_detector.run_detection(input_dir + "\\" + img)




