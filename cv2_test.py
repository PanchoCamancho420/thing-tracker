#! /Library/Frameworks/Python.framework/Versions/2.7/Resources/Python.app/Contents/MacOS/Python
import numpy as np
import cv2


img = np.random.rand(420, 420, 3)
cv2.imshow('test image', img)
cv2.waitKey(0)
