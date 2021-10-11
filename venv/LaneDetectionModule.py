import cv2
import numpy as np
import utils

curveList = []
avgVal = 10


def getLaneCurve(img):

    imgCopy = img.copy()

    imgThres = utils.thresholding(img)

    h, w, c = img.shape
    points = utils.valTrackbars()
    imgWarp = utils.warpImg(imgThres, points, w, h)
    imgWarpPoints = utils.drawPoints(img, points)

    middlePoint, imgHist = utils.getHistogram(
        imgWarp, display=True, minPer=0.5, region=4)
    curveAveragePoint, imgHist = utils.getHistogram(
        imgWarp, display=True, minPer=0.9)
    curveRaw = curveAveragePoint - middlePoint

    curveList.append(curveRaw)
    if len(curveList) > avgVal:
        curveList.pop(0)
    curve = int(sum(curveList)/len(curveList))

    cv2.imshow('BasePoint', curveAveragePoint)
    cv2.imshow('Histogram', imgHist)
    cv2.imshow('Warp Points', imgWarpPoints)
    cv2.imshow('Thres', imgThres)
    cv2.imshow('Warp', imgWarp)

    curve = curve/100
    if curve > 1:
        curve = 1
    if curve < -1:
        curve = -1

    return curve


if __name__ == '__main__':
    cap = cv2.VideoCapture('carvid.mp4')
    intialTrackBarVals = [134, 141, 73, 240]
    utils.initializeTrackbars(intialTrackBarVals)
    frameCounter = 0
    while True:
        frameCounter += 1
        if cap.get(cv2.CAP_PROP_FRAME_COUNT) == frameCounter:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            frameCounter = 0

        success, img = cap.read()
        img = cv2.resize(img, (480, 240))
        getLaneCurve(img)

        cv2.imshow('Video', img)
        cv2.waitKey(1)
