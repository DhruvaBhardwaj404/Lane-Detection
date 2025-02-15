import sys
import os
import time
import matplotlib.pyplot as plt
import cv2
import numpy as np
import cProfile

sobel_x = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])/8
sobel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])/8

sharpen_kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])


def generate_smoothing_kernel(n, std_dev):

    kernel = np.zeros((n, n))
    c = n // 2
    for x in range(0, n):
        for y in range(0, n):
            dist = np.sqrt((x - c)**2 + (y - c)**2)
            kernel[x][y] = np.exp(-(dist**2) / (2 * std_dev**2))

    kernel /= np.sum(kernel)
    return kernel


def convolve(img:np.ndarray, kernel: np.ndarray):
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


def log_img(img, n, std_dev):
    t1 = time.time()
    k = convolve(generate_smoothing_kernel(n, std_dev), sharpen_kernel)
    img = convolve(img, k)
    t2 = time.time()
    print("LOG time=>", int(t2 - t1))
    # plt.imshow(self.gimg,cmap="gray")
    # plt.show()
    return img


class LaneDetection:
    def __init__(self, local_area_size=2, local_area_width=2, local_area_length=10,
                 low_threshold=10, high_threshold=20, bin_max_percent = 0.9,
                 thetha_interval=1, poly_curve_interval=3, poly_curve_lower=0.5,
                 poly_curve_upper = 3):

        self.org_image = None
        self.gimg = None
        self.image_shape = None
        self.x_grad = None
        self.y_grad = None
        self.grad_mag = None
        self.grad_angle = None
        self.local_area_size = local_area_size
        self.local_area_width = local_area_width
        self.local_area_length = local_area_length
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold
        self.bin_max_percent = bin_max_percent
        self.thetha_interval = thetha_interval
        self.poly_curve_interval = poly_curve_interval
        self.poly_curve_lower = poly_curve_lower
        self.poly_curve_upper =  poly_curve_upper

    def canny_edge_detection(self):
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
        t1 = time.time()
        for i in range(self.image_shape[0]):
            for j in range(self.image_shape[1]):
                self.grad_mag[i][j] = np.sqrt(self.x_grad[i][j]**2 + self.y_grad[i][j]**2)
                if self.x_grad[i][j] != 0:
                    self.grad_angle[i][j] = np.arctan2(self.y_grad[i][j],self.x_grad[i][j])
        t2 = time.time()

        print("grad mag and angle time->", int(t2 - t1))
        plt.imshow(self.grad_mag, cmap="gray")
        plt.show()
        input()
        # temp = np.zeros((self.image_shape))
        # for i in range(self.image_shape[0]):
        #     for j in range(self.image_shape[1]):
        #         if self.grad_mag[i][j] == 0:
        #             temp[i][j] = 255
        #
        # plt.imshow(temp, cmap="gray")
        # plt.show()

        # t1 = time.time()
        # for i in range(self.image_shape[0]):
        #     for j in range(self.image_shape[1]):
        #         if self.grad_mag[i][j] == 0:
        #             continue
        #         # 0 90 135 45
        #
        #         move_down = False
        #         move_left = False
        #
        #         check_up = [(315, 360), (0, 45), (135, 225)]
        #         check_right = [(45, 135), (225, 315)]
        #
        #         for c in check_up:
        #             if c[0] <= self.grad_angle[i][j] <= c[1]:
        #                 move_left = True
        #
        #         for c in check_right:
        #             if c[0] <= self.grad_angle[i][j] <= c[1]:
        #                 move_down = True
        #         max_edge = True
        #
        #         if move_down:
        #             for dy in range(i-1, i-self.local_area_size-1):
        #                 if 0 < dy < self.image_shape[0]:
        #                     for dx in range(j-self.local_area_size//2, j+self.local_area_size//2):
        #                         if 0 < dx < self.image_shape[1]:
        #                             if np.isclose(self.grad_angle[i][j], self.grad_angle[dy][dx], 10):
        #                                 if self.grad_mag[i][j] < self.grad_mag[dy][dx]:
        #                                     self.grad_mag[i][j] = 0
        #                                     max_edge = False
        #                                 else:
        #                                     self.grad_mag[dy][dx] = 0
        #                         if self.grad_mag[i][j]==0:
        #                             break
        #                 if self.grad_mag[i][j] == 0:
        #                     break
        #             if max_edge:
        #                 self.grad_mag[i][j] = 255
        #
        #         if move_left:
        #             for dy in range(i-self.local_area_size//2, i+self.local_area_size//2):
        #                 if 0 < dy < self.image_shape[0]:
        #                     for dx in range(j-1, j-self.local_area_size-1):
        #                         if 0 < dx < self.image_shape[1]:
        #                             if np.isclose(self.grad_angle[i][j], self.grad_angle[dy][dx], 10):
        #                                 if self.grad_mag[i][j] < self.grad_mag[dy][dx]:
        #                                     self.grad_mag[i][j] = 0
        #                                 else:
        #                                     self.grad_mag[dy][dx] = 0
        #                         if self.grad_mag[i][j] == 0:
        #                             break
        #                 if self.grad_mag[i][j] == 0:
        #                     break
        #
        # t2 = time.time()
        # print("Edge filtering->", int(t2 - t1))

        plt.imshow(self.grad_mag, cmap='gray')
        plt.show()

        l_threshold = self.low_threshold
        h_threshold = self.high_threshold

        # for i in range(self.image_shape[0]):
        #     for j in range(self.image_shape[1]):
        #         if self.grad_mag[i][j] < l_threshold:
        #             self.grad_mag[i][j] = 0
        #             continue
        #         if self.grad_mag[i][j] > h_threshold:
        #             self.grad_mag[i][j] = 255
        #             continue

        for i in range(self.image_shape[0]):
            for j in range(self.image_shape[1]):
                if self.grad_mag[i][j] == 0:
                    continue

                m = self.grad_angle[i][j]

                if m == 0:
                    m = 0.0001

                c = (i + 1) / (m * (j + 1))

                for dj in range(0, self.local_area_length):
                    x = j + dj + 1
                    # print(i,j,x,m,c)
                    di_l = int(np.floor(m * x + c))
                    di_u = int(np.ceil(m * x + c))
                    not_max = False
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
                                    else:
                                        self.grad_mag[di_u - 1 + a][x] = 0

                        if not_max:
                            self.grad_mag[i][j] = 0

        plt.imshow(self.grad_mag, cmap='gray')
        plt.show()

    def hough_line_transform(self):

        #detecting lines
        diag = np.array([[self.image_shape[0]-1, 0], [0,self.image_shape[1]-1]])
        thetha_interval = self.thetha_interval

        line_bins = dict()    

        mem_sin = dict()
        mem_cos = dict()

        for thetha in np.arange(0, 180, thetha_interval):
            mem_sin[thetha] = np.sin(thetha)
            mem_cos[thetha] = np.cos(thetha)

        for i in range(self.image_shape[0]-1, -1, -1):
            for j in range(0, self.image_shape[1]):
                thetha_min = np.round(self.grad_mag[i][j] - 3, 0)
                thetha_max = np.round(self.grad_mag[i][j] + 3, 0)

                if thetha_min < 0:
                    thetha_min = 0
                if thetha_max > 180:
                    thetha_max = 180

                for thetha in np.arange(thetha_min, thetha_max, thetha_interval):
                    rho = np.round(j*mem_cos[thetha] + i*mem_sin[thetha], 0)
                    if line_bins.get((rho, thetha)) is None:
                        line_bins[(rho, thetha)] = 1
                    else:
                        line_bins[(rho, thetha)] += 1

        line_bins = [[line_bins[k], *k] for k in line_bins]
        line_bins_list = sorted(line_bins, reverse=True)
        pix_threshold = line_bins_list[0][0] * self.bin_max_percent
        i = 1
        for l in line_bins_list:
            if l[0] > pix_threshold:
                i += 1
            else:
                break
        line_bins_list = line_bins_list[0:i]

        line_pixs = [[] for _ in line_bins_list]

        first = [False for _ in line_pixs]
        count = [0 for _ in line_pixs]
        dont = [False for _ in line_pixs]

        for y in range(self.image_shape[0] - 1, -1, -1):
            for x in range(0, self.image_shape[1]):
                if self.grad_mag[y][x] != 0:
                    for i, l in enumerate(line_bins_list):
                        if dont[i]:
                            continue

                        r = np.round(x*mem_cos[l[2]] + y*mem_sin[l[2]], 0)
                        if r == l[1]:
                            first[i] =True
                            line_pixs[i].append((x, y))
                        else:
                            count[i] += 1
                            if count == 5:
                                dont[i] = True

        temp = np.zeros((*self.image_shape, 3))
        for l in line_pixs:
            if len(l) > 0:
                temp = cv2.line(temp, l[0], l[-1],(0, 255, 0),3)
        plt.imshow(temp)
        plt.show()

    def polynomial_curve_fitting(self):
        # fitting curve
        pass

    def output_image(self):
        pass


    def generate_centroid_for_lanes(self):
        pass

    def run_detection(self, path):

        with cProfile.Profile() as pr:
            image_raw = cv2.imread(path)
            self.org_image = image_raw
            temp = convert_to_grayscale(self.org_image)
            temp = log_img(temp, 5, 1.4)
            self.org_image = self.org_image[::2, ::2]
            self.gimg = temp[::2, ::2]
            self.image_shape = self.org_image.shape[0:2]

            plt.imshow(self.gimg, cmap="gray")
            plt.show()
            self.canny_edge_detection()
            self.hough_line_transform()
            # pr.print_stats()


if __name__ == '__main__':
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]

    images = os.listdir(input_dir)
    lane_detector = LaneDetection(local_area_size=3, local_area_width=3,
                                  local_area_length=6, low_threshold=10, high_threshold=20)

    for img in images:
        lane_detector.run_detection(os.path.join(input_dir, img))




