import cv2
import numpy as np


def thresholding(img):
    imgHsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lowerWhite = np.array([0, 47, 0])
    upperWhite = np.array([83, 255, 255])
    maskWhite = cv2.inRange(imgHsv, lowerWhite, upperWhite)
    return maskWhite


def warpImg(img, points, w, h, inv=False):
    pts1 = np.float32(points)
    pts2 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])

    if inv:
        matrix = cv2.getPerspectiveTransform(pts2, pts1)
    else:
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
    # wanna test this with removing the (w,h)
    imgWarp = cv2.warpPerspective(img, matrix, (w, h))
    return imgWarp


def nothing(a):
    pass


def initializeTrackbars(initialTracbarVals, wT=480, hT=240):
    cv2.namedWindow('Trackbars')
    cv2.resizeWindow('Trackbars', 360, 240)
    cv2.createTrackbar('Width Top', 'Trackbars',
                       initialTracbarVals[0], wT//2, nothing)
    cv2.createTrackbar('Height Top', 'Trackbars',
                       initialTracbarVals[1], hT, nothing)
    cv2.createTrackbar('Width Bottom', 'Trackbars',
                       initialTracbarVals[2], wT//2, nothing)
    cv2.createTrackbar('Height Bottom', 'Trackbars',
                       initialTracbarVals[3], hT, nothing)


def valTrackbars(wT=480, hT=240):
    widthTop = cv2.getTrackbarPos('Width Top', 'Trackbars')
    HeightTop = cv2.getTrackbarPos('Height Top', 'Trackbars')
    widthBottom = cv2.getTrackbarPos('Width Bottom', 'Trackbars')
    HeightBottom = cv2.getTrackbarPos('Height Bottom', 'Trackbars')
    points = np.float32([(widthTop, HeightTop), (wT-widthTop, HeightTop),
                        (widthBottom, HeightBottom), (wT-widthBottom, HeightBottom)])
    return points


def drawPoints(img, points):
    for x in range(4):
        cv2.circle(img, (int(points[x][0]), int(
            points[x][1])), 15, (0, 0, 255), cv2.FILLED)
    return img


def getHistogram(img, minPer=0.1, display=False, region=1):
    if region == 1:
        histValues = np.sum(img, axis=0)
    else:
        histValues = np.sum(img[img.shape[0]//region:, :], axis=0)
    maxValue = np.max(histValues)
    minValue = minPer*maxValue
    indexArray = np.where(histValues >= minValue)
    basePoint = int(np.average(indexArray))
    if display:
        imgHist = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)
        for x, intensity in enumerate(histValues):
            cv2.line(imgHist, (x, img.shape[0]),
                     (x, img.shape[0]-intensity//255//region), (255, 0, 255), 1)
            cv2.circle(
                imgHist, (basePoint, img.shape[0]), 20, (0, 255, 255), cv2.FILLED)
        return basePoint, imgHist
    return basePoint
