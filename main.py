import cv2
import cv2extensions as cv2e
from matplotlib import pyplot as plt
import time


def parse_video(file_name, size=-1):
    # type: (str) -> []
    vidcap = cv2.VideoCapture(file_name)
    # success, image = vidcap.read()
    img_arr = []
    success = True
    while success:
        success, image = vidcap.read()
        if success:
            if size >= 1:
                img_arr.append(cv2e.resize(image, size))
            else:
                img_arr.append(image)  # save frame as JPEG file
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


def compare_img(img1, img2):
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

    return img3


def img_compare_vid(img, vid):
    ret = []
    for img_itr in vid:
        new_img = compare_img(img, img_itr)
        ret.append(new_img)
    return ret


def show_vid(vid):
    for vid_img in vid:
        cv2.imshow('cool', vid_img)
        cv2.waitKey(0)


def track_vid(video):
    tracker = cv2.Tracker_create("MIL")

    current_frame = video[0]

    bbox = cv2.selectROI(current_frame, False)
    op1 = (int(bbox[0]), int(bbox[1]))
    op2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))

    still_go = tracker.init(current_frame, bbox)
    del still_go

    for img in video[1:]:
        still_go, bbox = tracker.update(img)
        if not still_go:
            break
        p1 = (int(bbox[0]), int(bbox[1]))
        p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
        cv2e.rect(img, p1, p2)
        cv2e.rect(img, op1, op2, color=[0, 0, 255])
        cv2.imshow('tracker', img)

        k = cv2.waitKey(1) & 0xff
        if k == 27:
            break


def std_main():
    print 'loading images'

    images = parse_video('video.mp4')

    print 'resizing images'

    new_images = [cv2e.resize(x, 720) for x in images]

    print 'loading refrence'

    sighn = cv2.imread("arrowsighn.jpg")

    print 'compareing'

    compares = img_compare_vid(sighn, new_images)

    print 'dispaling'

    show_vid(compares)


if __name__ == '__main__':
    images = parse_video("video.mp4", 720)
    track_vid(images)
