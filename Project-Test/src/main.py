import cv2 as cv
import imutils
import numpy as np
import sys
import glob
import os
import argparse
import math

# from utils import remove_barrel
from utils import remove_barrel
from transform import four_point_transform
import json

"""
PROGRAM SETTINGS
"""
# Per velocizzare l'algoritmo, processo i vari frame ad una risoluzione meno elevata
RESIZE_VIDEO = 0.5

RECOGNIZE_PAINTING = True
RECTIFY_PAINTING = True

"""
SIFT
Loading all the necessary files BEFORE executing the program
Then a function to recognize the paiting is defined
"""
DATASET_FOLDER = "Project-Test/WebScraping/dataset/"
if os.path.isfile(DATASET_FOLDER + "sift_500.json"):
    with open(DATASET_FOLDER + "sift_500.json", "r") as fin:
        data_sift = json.load(fin)
else:
    print("ERRORE. File sift_500.json not found in " + DATASET_FOLDER)
    print("Please fix the path by modifying the DATASET_FOLDER var")
    sys.exit()


def recognize_paiting(artwork):

    query_img = cv.cvtColor(artwork, cv.COLOR_BGR2GRAY)
    # query_img = cv.imread('14.jpg', cv.IMREAD_GRAYSCALE)
    # query_img = imutils.resize(query_img, width=1000)

    scale_percent = 60  # percent of original size
    width = int(query_img.shape[1] * scale_percent / 100)
    height = int(query_img.shape[0] * scale_percent / 100)
    dim = (width, height)
    # resize image if it is too big
    if query_img.shape[0] * query_img.shape[1] > 9e5:
        query_img = cv.resize(
            query_img, None, fx=0.7, fy=0.7, interpolation=cv.INTER_AREA
        )

    # cv.imshow('cropnew', query_img)
    # la pulisco rimuovendo il rumore
    # query_img_clean = cv.fastNlMeansDenoising(query_img)
    # cv.imshow('cropnew_clean', query_img_clean)
    MIN_MATCH_COUNT = 0

    e1 = cv.getTickCount()

    # Initiate SIFT detector
    sift = cv.xfeatures2d.SIFT_create()
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(query_img, None)
    # kp2, des2 = sift.detectAndCompute(img2,None)

    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv.FlannBasedMatcher(index_params, search_params)

    results = {}
    for key, value in data_sift.items():
        des = np.array(value, dtype="float32")
        # matches = flann.knnMatch(des1, des, k=2)
        matches = flann.knnMatch(
            des, des1, k=2
        )  # USA QUESTA! QUELLA DI PRIMA DA I NUMERI
        # store all the good matches as per Lowe's ratio test.
        good = []
        # Need to draw only good matches, so create a mask
        # matchesMask = [[0, 0] for i in range(len(matches))]
        for i, (m, n) in enumerate(matches):
            if m.distance < 0.55 * n.distance:
                good.append(m)
                # matchesMask[i] = [1, 0]

        if len(good) > MIN_MATCH_COUNT:
            # draw_params = dict(matchColor=(0, 255, 0),
            #                    singlePointColor=(255, 0, 0),
            #                    matchesMask=matchesMask,
            #                    flags=cv.DrawMatchesFlags_DEFAULT)
            # print(key+" seems to be the perfect match")
            # print(str(len(good))+"\n")
            results[key] = len(good)

    sorted_x = sorted(results.items(), key=lambda kv: kv[1], reverse=True)
    print(sorted_x[:10])

    with open(DATASET_FOLDER + "data.json", "r") as fin:
        data = json.load(fin)
        nome = data[sorted_x[0][0]].get("nome", None)
        autore = data[sorted_x[0][0]].get("autore", None)
        cronologia = data[sorted_x[0][0]].get("cronologia", None)
        materia = data[sorted_x[0][0]].get("materia", None)
        dimensioni = data[sorted_x[0][0]].get("dimensioni", None)
        url = data[sorted_x[0][0]].get("url", None)
        filename = data[sorted_x[0][0]].get("filename", None)
        print("Nome quadro: " + nome)
        print("Autore: " + autore)
        print("Cronologia: " + cronologia)
        print("Materia e tecnica: " + materia)
        print("Dimensioni: " + dimensioni)
        print("url: " + url)
        e2 = cv.getTickCount()
        t = (e2 - e1) / cv.getTickFrequency()
        print(t)

        quadro = cv.imread(DATASET_FOLDER + "img/" + filename)
        cv.imshow("quadro originale", quadro)

        # Create a black image
        text = np.zeros((512, 512, 3), np.uint8)

        # if cv.waitKey(0) & 0xFF == ord('c'):
        #     pass
    # cv.imshow('crop', artwork)


"""
mouse callback function
params = [(roi), (img)]
roi = sono i rettangoli dei quadri
img = immagine pulita

Lo scopo di questa funzione è riconoscere dove si è fatto il doppio click per fare 2 cose:
1) passare il CROP del quadro all'algoritmo SIFT
2) passare il CROP del quadro alla mia getPerspectiveTransform per rettificarlo
"""


def mouse_callback(event, x, y, flags, params):
    if event == cv.EVENT_LBUTTONDBLCLK:
        img = params[1]
        x = int(x * 1 / RESIZE_VIDEO)
        y = int(y * 1 / RESIZE_VIDEO)
        # devo sapere il mouse dentro quale rettangolo si trova
        for rect in params[0]:
            # rect = (x, y, w, h)
            x_q = int(rect[0] * 1 / RESIZE_VIDEO)
            y_q = int(rect[1] * 1 / RESIZE_VIDEO)
            w_q = int(rect[2] * 1 / RESIZE_VIDEO)
            h_q = int(rect[3] * 1 / RESIZE_VIDEO)
            if (x > x_q and x < (x_q + w_q)) and (y > y_q and y < (y_q + h_q)):
                imgcrop = img[y_q : y_q + h_q, x_q : x_q + w_q]
                # cv.imshow('crop', imgcrop)
                if RECOGNIZE_PAINTING:
                    print("Riconoscimento quadro...")
                    recognize_paiting(imgcrop)

                if RECTIFY_PAINTING:
                    print("Rettificazione...")
                    # getPerspectiveTransform2(imgcrop)
                    getPerspectiveTransform_auto(imgcrop)

                if cv.waitKey(0) & 0xFF == ord("c"):
                    pass

                break
        # cv.imshow('prova', img)
        # if cv.waitKey(0) & 0xFF == ord('c'):
        #     pass


"""
***ATTENZIONE: QUESTA FUNZIONE POTETE ANCHE ELIMINARLA, NON SERVE A NULLA
Era il vecchio metodo per fare una getPerspectiveTransform automatica,
ma non funziona ed è incompleta***
Lo scopo di questa funzione è quella di rettificare il quadro.
In input hai già il quadro a grandezza originale (dal video o immagine)
"""


def getPerspectiveTransform(img):
    # metodo classico con canny
    color_blurred = cv.medianBlur(img, 3)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    blurred = cv.GaussianBlur(gray, (3, 3), 0)

    # metodo HSV
    hsv = cv.cvtColor(color_blurred, cv.COLOR_BGR2HSV)
    lower = np.array([0, 0, 80])
    upper = np.array([179, 255, 255])
    mask = cv.inRange(hsv, lower, upper)
    mask = cv.bitwise_not(mask)
    kernel = np.ones((3, 3), np.uint8)
    erosion = cv.erode(mask, kernel, iterations=1)
    dilation = cv.dilate(erosion, kernel, iterations=5)
    cv.imshow("dilation", dilation)

    # edged = imutils.auto_canny(dilation)
    edged = cv.Canny(dilation, 50, 200)
    cv.imshow("edged", edged)

    # TODO
    """
    Da qui in poi quello che sto cercando di fare (senza riuscirci)
    è quello di ottenere i 4 punti (top-right, top-left, bottom-right, bottom-left) del quadro
    in modo che, se dati in pasto alla funzione four_point_transform, ti ribalta il quadro
    come facciamo? 
    """

    # find contours in the edge map, then initialize
    # the contour that corresponds to the document
    # cnts = cv.findContours(edged.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    im2, contours, hierarchy = cv.findContours(
        edged.copy(), cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE
    )
    # cnts = imutils.grab_contours(cnts)
    # cnts = sorted(cnts, key=cv.contourArea, reverse=True)[:5]

    docCnt = None

    cv.drawContours(img, contours, -1, (0, 0, 255), 2)

    # loop over the contours
    rectangles = []
    for c in contours:
        # approximate the contour
        peri = cv.arcLength(c, True)
        approx = cv.approxPolyDP(c, 0.2 * peri, True)

        # if our approximated contour has four points, then we
        # can assume that we have found our screen
        if len(approx) > 3:
            # x, y, w, h = cv.boundingRect(c)
            # cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 4)
            rect = cv.minAreaRect(c)
            box = cv.boxPoints(rect)
            box = np.int0(box)
            rectangles.append(box)
            cv.drawContours(img, [box], 0, (0, 255, 0), 2)

            screenCnt = approx
            # break

    # # devo trovare il rettangolo più grande
    # rect_list = []
    # for rect in rectangles:
    #     peri = 0
    #     dist1 = math.sqrt((rect[1][0] - rect[0][0]) ** 2 + (rect[1][1] - rect[0][1]) ** 2)
    #     dist2 = math.sqrt((rect[2][0] - rect[1][0]) ** 2 + (rect[2][1] - rect[1][1]) ** 2)
    #     dist3 = math.sqrt((rect[3][0] - rect[2][0]) ** 2 + (rect[3][1] - rect[2][1]) ** 2)
    #     dist4 = math.sqrt((rect[3][0] - rect[0][0]) ** 2 + (rect[3][1] - rect[0][1]) ** 2)
    #     peri = dist1+dist2+dist3+dist4
    #     peri = int(peri)
    #     rect_list.append((rect, peri))
    #
    # biggest_rect = sorted(rect_list, key=lambda rect : rect[1], reverse=True)
    #
    # print(biggest_rect)
    #
    # cv.drawContours(img, [biggest_rect[0]], 0, (100, 100, 100), 5)

    # TODO bisogna ordinare i punti top-left, top-right etc...

    cv.imshow("imgcont", img)

    # apply the four point transform to obtain a top-down
    # view of the original image
    # try:
    #     warped = four_point_transform(img.copy(), screenCnt.reshape(4, 2))
    #     # show the original and scanned images
    #     print("Apply perspective transform")
    #     # cv.imshow("Original", img)
    #     cv.imshow("Warped", warped)
    # except:
    #     print("Non ho potuto rettificare il quadro")


"""
Queste 3 funzioni sono quelle definitive
Le prime due get_4_points e getPerspectiveTransform_manual servono in caso si vogliano selezionare i punti manualmente
L'ultima, getPerspectiveTransform_auto, fa tutto automaticamente
"""
four_coord = []


def get_4_points(event, x, y, flags, params):
    global four_coord
    if event == cv.EVENT_LBUTTONDOWN:
        four_coord.append((x, y))
        # four_coord.append(x)
        # four_coord.append(y)


def getPerspectiveTransform_manual(img):
    global four_coord
    cv.namedWindow("get_4_points")
    cv.setMouseCallback("get_4_points", get_4_points)
    while 1:
        cv.imshow("get_4_points", img)
        k = cv.waitKey(20) & 0xFF
        if k == 27:
            break
        if len(four_coord) == 4:
            print(four_coord)
            # four_coord = []
            break

    # apply the four point transform to obtain a top-down
    # view of the original image
    try:
        # warped = four_point_transform(img.copy(), screenCnt.reshape(4, 2))
        warped = four_point_transform(img.copy(), np.asarray(four_coord))
        four_coord = []
        # show the original and scanned images
        print("Apply perspective transform")
        # cv.imshow("Original", img)
        cv.imshow("Warped", warped)
        # cv.waitKey(0)
        # cv.destroyAllWindows()
    except:
        print("Non ho potuto rettificare il quadro")


def getPerspectiveTransform_auto(img):
    # metodo classico con canny
    color_blurred = cv.medianBlur(img, 3)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    blurred = cv.GaussianBlur(gray, (3, 3), 0)

    # metodo HSV
    hsv = cv.cvtColor(color_blurred, cv.COLOR_BGR2HSV)
    lower = np.array([0, 0, 80])
    upper = np.array([179, 255, 255])
    mask = cv.inRange(hsv, lower, upper)
    mask = cv.bitwise_not(mask)
    kernel = np.ones((3, 3), np.uint8)
    erosion = cv.erode(mask, kernel, iterations=1)
    erosion = cv.medianBlur(erosion, 15)
    # dilation = cv.dilate(erosion, kernel, iterations=2)
    erosion = cv.medianBlur(erosion, 15)
    erosion[:1, :] = 0
    erosion[:, :1] = 0
    erosion[-1:, :] = 0
    erosion[:, -1:] = 0
    # cv.imshow('dilation', erosion)

    # edged = imutils.auto_canny(dilation)
    edged = cv.Canny(erosion, 50, 200)
    # cv.imshow('edged', edged)

    # find contours in the edge map, then initialize
    # the contour that corresponds to the document
    cnts = cv.findContours(edged.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    # cnts = sorted(cnts, key=cv.contourArea, reverse=True)[:5]

    c = max(cnts, key=cv.contourArea)
    # cv.drawContours(img, cnts, -1, (0, 0, 255), 2)

    docCnt = None

    # determine the most extreme points along the contour
    top_left = c[np.argmin(np.sum(c, axis=2))].squeeze(0)
    bot_right = c[np.argmax(np.sum(c, axis=2))].squeeze(0)
    top_right = c[
        np.argmax(np.sum(np.concatenate((c[:, :, 0], c[:, :, 1] * -1), axis=1), axis=1))
    ].squeeze(0)
    bot_left = c[
        np.argmin(np.sum(np.concatenate((c[:, :, 0], c[:, :, 1] * -1), axis=1), axis=1))
    ].squeeze(0)

    # pts = np.array([top_left, top_right, bot_right, bot_left], np.int32)
    # pts = pts.reshape((-1, 1, 2))
    # cv.polylines(img, [pts], True, (255, 0, 0), 3)

    width_A = np.sqrt(
        ((bot_right[0] - bot_left[0]) ** 2) + ((bot_right[1] - bot_left[1]) ** 2)
    )
    width_B = np.sqrt(
        ((top_right[0] - top_left[0]) ** 2) + ((top_right[1] - top_left[1]) ** 2)
    )
    mean_width = int(np.mean((width_A, width_B)))

    height_A = np.sqrt(
        ((top_right[0] - bot_right[0]) ** 2) + ((top_right[1] - bot_right[1]) ** 2)
    )
    height_B = np.sqrt(
        ((top_left[0] - bot_left[0]) ** 2) + ((top_left[1] - bot_left[1]) ** 2)
    )
    mean_height = int(np.mean((height_A, height_B)))

    alpha = (
        (max(height_A, height_B) / min(height_A, height_B) - 1) * 1.2 + 1
    ) * np.clip(704 / max(width_A, width_B) / 4, a_min=1, a_max=2)
    # print(alpha)

    dst = np.array(
        [
            [0, 0],
            [int(mean_width * alpha) - 1, 0],
            [int(mean_width * alpha) - 1, mean_height - 1],
            [0, mean_height - 1],
        ],
        dtype="float32",
    )

    M = cv.getPerspectiveTransform(
        np.array([top_left, top_right, bot_right, bot_left]).astype(np.float32), dst
    )
    warped = cv.warpPerspective(img, M, (int(mean_width * alpha), mean_height))

    cv.imshow("imgcont", img)
    cv.imshow("warped", warped)

    return warped


"""
process video
"""


def process_video(user_path):
    # cap = cv.VideoCapture('../video/marcello/20180206_113059.mp4')
    # cap = cv.VideoCapture('../video/marcello/20180206_113716.mp4')
    # cap = cv.VideoCapture('../video/marcello/20180206_113800.mp4')
    # cap = cv.VideoCapture('../video/marcello/20180206_114506.mp4')
    # cap = cv.VideoCapture('../video/marcello/20180206_114720.mp4')
    # cap = cv.VideoCapture('../video/vittoria/GOPR1927.MP4')
    # cap = cv.VideoCapture('../video/vittoria/GOPR1929.MP4')
    # cap = cv.VideoCapture('../video/vittoria/GOPR1936.MP4')
    cap = cv.VideoCapture(user_path)

    FRAME_NON_LETTO = 0

    while cap.isOpened():
        ret, frame = cap.read()

        # a volte non riesce a leggere correttamente alcuni frame
        if ret != 1:
            # print("ERRORE. FRAME NON LETTO")
            FRAME_NON_LETTO += 1
            if FRAME_NON_LETTO > 10:  # se vero, è finito il video
                print("TROPPI FRAME SALTATI. IL VIDEO NON ERA BUONO O E' TERMINATO")
                break
            continue

        # fps = cap.get(cv.CAP_PROP_FPS)

        resized = cv.resize(
            frame, None, fx=RESIZE_VIDEO, fy=RESIZE_VIDEO, interpolation=cv.INTER_LINEAR
        )

        # dst = remove_barrel(resized)
        dst = resized

        dst_clean = dst.copy()
        cv.namedWindow("final")

        # Convert BGR to HSV
        hsv = cv.cvtColor(dst, cv.COLOR_BGR2HSV)

        # define range of blue color in HSV
        lower_thresh = np.array([0, 0, 90])
        upper_thresh = np.array([179, 180, 255])

        # Threshold the HSV image
        mask = cv.inRange(hsv, lower_thresh, upper_thresh)
        # Bitwise-AND mask and original imageq
        mask = cv.bitwise_not(mask)
        # cv.imshow('mask',mask)

        # Apply a morphol transf to the image
        kernel = np.ones((5, 5), np.uint8)
        dilation = cv.dilate(mask, kernel, iterations=2)
        # cv.imshow('dilation', dilation)
        erosion = cv.erode(dilation, kernel, iterations=1)
        # cv.imshow('erosion', erosion)

        erosion_blurred = cv.medianBlur(erosion, 5)
        # cv.imshow('HSV_erosion_blurred', erosion_blurred)

        im2, contours, h = cv.findContours(
            erosion_blurred, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE
        )
        # cv.drawContours(resized, contours2, -1, (0, 0, 255), 2)

        rois = []
        for cont in contours:
            if cv.contourArea(cont) > 6000:
                # if cv.arcLength(cont, True) > 1000:
                arc_len = cv.arcLength(cont, True)
                approx = cv.approxPolyDP(cont, 0.06 * arc_len, True)

                # or try to rectangle it
                if len(approx) > 2:
                    x, y, w, h = cv.boundingRect(cont)
                    cv.rectangle(dst, (x, y), (x + w, y + h), (0, 255, 0), 3)
                    rois.append((x, y, w, h))
        cv.setMouseCallback("final", mouse_callback, [rois, frame])

        font = cv.FONT_HERSHEY_SIMPLEX
        position_text = (10, dst.shape[0] - 30)
        cv.putText(
            dst, str(len(rois)), position_text, font, 3, (0, 0, 255), 2, cv.LINE_AA
        )
        cv.imshow("final", dst)

        if cv.waitKey(25) & 0xFF == ord("q"):
            break

    cap.release()
    cv.destroyAllWindows()


"""
process image
"""


def process_image(user_path):
    frame = cv.imread(user_path)

    resized = cv.resize(
        frame, None, fx=RESIZE_VIDEO, fy=RESIZE_VIDEO, interpolation=cv.INTER_LINEAR
    )

    # dst = remove_barrel(resized)
    dst = resized

    dst_clean = dst.copy()
    cv.namedWindow("final")

    # Convert BGR to HSV
    hsv = cv.cvtColor(dst, cv.COLOR_BGR2HSV)

    # define range of blue color in HSV
    lower_thresh = np.array([0, 0, 90])
    upper_thresh = np.array([179, 180, 255])

    # Threshold the HSV image
    mask = cv.inRange(hsv, lower_thresh, upper_thresh)
    # Bitwise-AND mask and original imageq
    mask = cv.bitwise_not(mask)
    # cv.imshow('mask',mask)

    # Apply a morphol transf to the image
    kernel = np.ones((5, 5), np.uint8)
    dilation = cv.dilate(mask, kernel, iterations=2)
    # cv.imshow('dilation', dilation)
    erosion = cv.erode(dilation, kernel, iterations=1)
    # cv.imshow('erosion', erosion)

    erosion_blurred = cv.medianBlur(erosion, 5)
    # cv.imshow('HSV_erosion_blurred', erosion_blurred)

    im2, contours, h = cv.findContours(
        erosion_blurred, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE
    )
    # cv.drawContours(resized, contours2, -1, (0, 0, 255), 2)

    rois = []
    for cont in contours:
        if cv.contourArea(cont) > 6000:
            # if cv.arcLength(cont, True) > 1000:
            arc_len = cv.arcLength(cont, True)
            approx = cv.approxPolyDP(cont, 0.06 * arc_len, True)

            # or try to rectangle it
            if len(approx) > 2:
                x, y, w, h = cv.boundingRect(cont)
                cv.rectangle(dst, (x, y), (x + w, y + h), (0, 255, 0), 3)
                rois.append((x, y, w, h))
    cv.setMouseCallback("final", mouse_callback, [rois, frame])

    font = cv.FONT_HERSHEY_SIMPLEX
    position_text = (10, dst.shape[0] - 30)
    cv.putText(dst, str(len(rois)), position_text, font, 3, (0, 0, 255), 2, cv.LINE_AA)
    cv.imshow("final", dst)

    if cv.waitKey(0) & 0xFF == ord("q"):
        cv.destroyAllWindows()
        return


"""
Funzione che riassume le impostazioni del programma prima di elaborare il video
"""


def do_some_checks(args):
    print("###################### CHECKS ######################")

    print("Video Setting Resize: " + str(RESIZE_VIDEO))

    print("Database SIFT successfully loaded")

    if RECOGNIZE_PAINTING == True:
        print("Recognize painting: \tActive")
    else:
        print("Recognize painting: \tNOT Active")

    if RECTIFY_PAINTING == True:
        print("Rectifying painting: \tActive")
    else:
        print("Rectifying painting: \tNOT active")

    if args["image"] is None and args["video"] is None:
        print("No video or image as input. Please give me something to work with")
        sys.exit()
    if args["image"] is not None:
        if os.path.isfile(args["image"]):
            print("Looks like the input " + args["image"] + " is good")
        else:
            print("Bad input. Exiting program. Is the path to the image correct?")
            sys.exit()
    if args["video"] is not None:
        if os.path.isfile(args["video"]):
            print("Looks like the input " + args["video"] + " is good")
        else:
            print("Bad input. Exiting program. Is the path to the video correct?")
            sys.exit()

    print("#################### SO FAR SO GOOD! ####################")
    print("")


if __name__ == "__main__":
    # example usage : python3 main.py --image /path/to/image/file.jpg
    # example usage : python3 main.py --video /path/to/video/file.mp4
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=False, help="path to input image")
    ap.add_argument("-v", "--video", required=False, help="path to input video")
    args = vars(ap.parse_args())

    # do some preliminary checks before starting processing video or image
    do_some_checks(args)

    # START PROCESSING
    if args["video"] is not None:
        process_video(args["video"])

    if args["image"] is not None:
        process_image(args["image"])
    sys.exit()  # exit the program gently
