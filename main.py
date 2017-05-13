import cv2
import cv2extensions as cv2e
import numpy as np
from matplotlib import pyplot as plt
import time


def parse_video(file_name):
    # type: (str) -> []
    vidcap = cv2.VideoCapture(file_name)
    # success, image = vidcap.read()
    img_arr = []
    success = True
    count = 0
    while success and count <= 50:
        count += 1
        success, image = vidcap.read()
        if success:
            img_arr.append(image)  # save frame as JPEG file
            cv2.imshow('cool', cv2e.resize(image, 720))
            cv2.waitKey(1)
    return img_arr


def main():
    img_arr = parse_video('video.mp4')
    base_img = cv2.imread('arrowsighn.jpg')
    if True:
        img = img_arr[1]
        img1 = base_img  # queryImage
        img2 = img  # trainImage

        # Initiate SIFT detector
        sift = cv2.xfeatures2d.SIFT_create()

        # find the keypoints and descriptors with SIFT
        kp1, des1 = sift.detectAndCompute(img1, None)
        kp2, des2 = sift.detectAndCompute(img2, None)

        # BFMatcher with default params
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1, des2, k=2)

        # Apply ratio test
        good = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good.append([m])

        # cv2.drawMatchesKnn expects list of lists as matches.
        img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good, flags=2, outImg=None)

        plt.imshow(img3), plt.show()
        time.sleep(1)
        plt.close()


def show_campare(img1, img2):
    # Initiate SIFT detector
    sift = cv2.xfeatures2d.SIFT_create()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # BFMatcher with default params
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    # Apply ratio test
    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append([m])

    # cv2.drawMatchesKnn expects list of lists as matches.
    img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good, flags=2, outImg=None)

    plt.imshow(img3), plt.show()
    time.sleep(1)
    plt.close()


def show_images(images):
    from matplotlib import animation

    fig = plt.figure()
    init_img = images[0]
    im = plt.imshow(init_img, cmap='gist_gray_r', vmin=0, vmax=1)

    def init():
        im.set_data(init_img)

    def animate(i):
        img = images[i]
        im.set_data(img)
        return im

    anim = animation.FuncAnimation(fig, animate, init_func=init, frames=len(images), interval=50)
    plt.show(anim)


def test():
    parse_video('video.mp4')


if __name__ == '__main__':
    images = parse_video('video.mp4')
    show_images(images)
