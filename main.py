import sys
import os
import time
import matplotlib.pyplot as plt
import cv2
import numpy as np
import cProfile

sobel_x = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])/8
sobel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])/8


def generate_smoothing_kernel(n, std_dev):

    kernel = np.zeros((n, n))
    c = n // 2
    for x in range(0, n):
        for y in range(0, n):
            dist = np.sqrt((x - c)**2 + (y - c)**2)
            kernel[x][y] = np.exp(-(dist**2) / (2 * std_dev**2))

    kernel /= np.sum(kernel)
    return kernel


def convolve(img:np.ndarray, kernel:np.ndarray):
    ksize = kernel.shape
    isize = img.shape
    k = isize[0]+(ksize[0]-1)*2
    l = isize[1]+(ksize[1]-1)*2
    temp = np.zeros((k,l))
    temp[ksize[0]-1: ksize[0]-1 + isize[0], ksize[1]-1: ksize[1]-1 + isize[1]] = img

    res = np.zeros((isize[0] + ksize[0] - 1, isize[1] + ksize[1]-1))

    for i in range(0, isize[0] + ksize[0]-1):
        for j in range(0, isize[1] + ksize[1]-1):
            # print(i+ksize[0], j+ksize[1])
            t = temp[i:i+ksize[0], j:j+ksize[1]]
            res[i][j] = np.sum(t*kernel)

    # print(res)
    return res[ksize[0]-1: ksize[0]-1 + isize[0], ksize[1]-1: ksize[1]-1 + isize[1]]


def convert_to_grayscale(img: np.ndarray):
    weights = [0.2989, 0.5870, 0.1140]
    gimg = np.zeros((img.shape[0], img.shape[1]))
    for i, row in enumerate(img):
        for j, pix in enumerate(row):
            gimg[i][j] = np.dot(pix, weights)
    return gimg


class LaneDetection:
    def __init__(self, smoothing_kernel, local_area_length=20, local_area_width=1):
        self.smoothing_kernel = smoothing_kernel
        self.gimg = None
        self.image_shape = None
        self.x_grad = None
        self.y_grad = None
        self.grad_mag = None
        self.grad_angle = None
        self.local_area_length = local_area_length
        self.local_area_width = local_area_width

    def smooth_image(self):
        t1 = time.time()
        self.gimg = convolve(self.gimg,self.smoothing_kernel)
        t2 = time.time()
        print("Smoothing time=>", int(t2 - t1))
        plt.imshow(self.gimg,cmap="gray")
        plt.show()

    def calculate_gradients(self):
        t1 = time.time()
        self.x_grad = convolve(self.gimg, sobel_x)
        t2 = time.time()
        print("x_grad time->", int(t2 - t1))
        plt.imshow(self.x_grad, cmap="gray")
        plt.show()


        t1 = time.time()
        self.y_grad = convolve(self.gimg, sobel_y)
        t2 = time.time()
        print("y_grad time->", int(t2 - t1))
        plt.imshow(self.y_grad, cmap="gray")
        plt.show()

        self.grad_mag = np.zeros(self.image_shape)
        self.grad_angle = np.zeros(self.image_shape)

        for i in range(self.image_shape[0]):
            for j in range(self.image_shape[1]):
                self.grad_mag[i][j] = np.sqrt(self.x_grad[i][j]**2 + self.y_grad[i][j]**2)
                if self.x_grad[i][j] != 0:
                    self.grad_angle[i][j] = np.arctan(self.y_grad[i][j]/self.x_grad[i][j])

        for i in range(self.image_shape[0]):
            for j in range(self.image_shape[1]):
                if self.grad_mag[i][j] == 0:
                    continue
                if self.grad_mag[i][j] < 10:
                    self.grad_mag[i][j] = 0
                    continue
                if self.grad_mag[i][j] > 25:
                    self.grad_mag[i][j] = 255
                    continue

                m = self.grad_angle[i][j]
                c = (i+1)/(m*(j+1))
                h_exist = False

                for di in range(i-self.local_area_length, i + self.local_area_length):
                    if 0 < di < self.image_shape[0]:
                        for dj in range(j-self.local_area_width, j+self.local_area_width):
                            if 0 < dj < self.image_shape[1]:
                                if np.isclose(di+1, m*(dj+1) + c, 5):
                                    if self.grad_mag[di][dj] > 25:
                                        h_exist = True
                                        self.grad_mag[i][j] = 255
                                        break
                        if h_exist:
                            break
                if not h_exist:
                    self.grad_mag[i][j] = 0

                for dj in range(0, self.local_area_length):
                    x = j+dj+1
                    di_l = int(np.floor(m*x + c))
                    di_u = int(np.ceil(m*x + c))
                    not_max = False
                    for a in range(-self.local_area_width, self.local_area_width):
                        if 0 < x < self.image_shape[1]:

                            if 0 < di_l - 1 + a < self.image_shape[0]:
                                if self.grad_mag[i][j] < self.grad_mag[di_l - 1 + a][x]:
                                    not_max = True
                                    self.grad_mag[i][j] = 0
                                    break
                            if 0 < di_u - 1 + a < self.image_shape[0]:
                                if self.grad_mag[i][j] < self.grad_mag[di_u -1 + a ][x]:
                                    not_max = True
                                    self.grad_mag[i][j] = 0
                                    break

                    if self.grad_mag[i][j] != 0:
                        x = j - dj + 1
                        di_l = int(np.floor(m * x + c))
                        di_u = int(np.ceil(m * x + c))
                        for a in range(-self.local_area_width, self.local_area_width):
                            if 0 < x < self.image_shape[1]:
                                if 0 < di_l - 1 + a < self.image_shape[0]:
                                    if self.grad_mag[i][j] < self.grad_mag[di_l - 1 + a][x]:
                                        not_max = True
                                        self.grad_mag[i][j] = 0
                                        break

                                if 0 < di_u - 1 + a < self.image_shape[0]:
                                    if self.grad_mag[i][j] < self.grad_mag[di_u - 1 + a][x]:
                                        not_max = True
                                        self.grad_mag[i][j] = 0
                                        break
                        if not_max:
                            self.grad_mag[i][j] = 0
                        else:
                            self.grad_mag[i][j] = 255

        # print(self.grad_mag)
        plt.imshow(self.grad_mag, cmap='gray')
        plt.show()

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

        with cProfile.Profile() as pr:
            image_raw = cv2.imread(path)
            self.image_shape = image_raw.shape[0:2]
            plt.imshow(image_raw)
            plt.show()
            self.gimg = convert_to_grayscale(image_raw)
            plt.imshow(self.gimg, cmap="gray")
            plt.show()
            self.smooth_image()
            self.calculate_gradients()
            pr.print_stats()


if __name__ == '__main__':
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]

    images = os.listdir(input_dir)
    smoothing_kernel = generate_smoothing_kernel(5, 2)
    lane_detector = LaneDetection(smoothing_kernel)

    for img in images:
        lane_detector.run_detection(os.path.join(input_dir, img))




