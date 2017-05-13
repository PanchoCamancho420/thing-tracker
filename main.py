import cv2
from matplotlib import pyplot as plt


def parse_video(file_name):
    # type: (str) -> []
    vidcap = cv2.VideoCapture(str(file_name))
    # success, image = vidcap.read()
    img_arr = []
    success = True
    while success:
        success, image = vidcap.read()
        if success:
            img_arr.append(image)  # save frame as JPEG file
            cv2.imshow('cool', image)
            cv2.waitKey(1)
    return img_arr


def main():

    img1 = cv2.imread('symbol.jpg', 0)  # queryImage
    img2 = cv2.imread('symbols.jpg', 0)  # trainImage

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


def test():
    parse_video('video.mp4')


if __name__ == '__main__':
    test()
