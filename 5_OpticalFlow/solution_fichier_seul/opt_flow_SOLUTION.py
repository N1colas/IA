#!/usr/bin/env python

'''
example to show optical flow

USAGE: opt_flow.py [<video_source>]

Keys:
 1 - toggle HSV flow visualization
 2 - toggle glitch

Keys:
    ESC    - exit
'''

# Python 2/3 compatibility
from __future__ import print_function

import numpy as np
import cv2 as cv
import video

sensibilite_mouvement = 2
sensibilite_duree = 5;
x_gauche=0;
x_droite=0;
x_neutre=0;
x_text="";
old_x_text="";

commandes=[]

def draw_flow(img, flow, step=16):
    global x_gauche,x_droite, x_neutre
    global sensibilite_duree,sensibilite_mouvement
    global x_text,old_x_text

    h, w = img.shape[:2]
    y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2,-1).astype(int)
    fx, fy = flow[y,x].T
    lines = np.vstack([x, y, x+fx, y+fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)
    vis = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
    cv.polylines(vis, lines, 0, (0, 255, 0))
    for (x1, y1), (_x2, _y2) in lines:
        cv.circle(vis, (x1, y1), 1, (0, 255, 0), -1)

    x_avg = np.average(fx)
    y_avg = np.average(fy)


    if(abs(x_avg)<sensibilite_mouvement):
       x_neutre+=1
    else:
        if(x_avg>sensibilite_mouvement):
            x_gauche+=1
        else:
            x_droite+=1

    if (x_gauche > sensibilite_duree):
        x_gauche = 0
        x_droite = 0
        x_neutre = 0
        x_text = "gauche";

    if (x_droite > sensibilite_duree):
        x_gauche = 0
        x_droite = 0
        x_neutre = 0
        x_text = "droite";

    if (x_neutre > sensibilite_duree):
        x_gauche = 0
        x_droite = 0
        x_neutre = 0
        x_text = "neutre";


    cv.putText(vis, x_text, (10, 100), cv.FONT_HERSHEY_SIMPLEX, 4, (0, 255, 0), 2, cv.LINE_AA)

    cv.putText(vis, ', '.join(commandes), (10, 200), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2, cv.LINE_AA)

    if old_x_text!= x_text:
        commandes.append(x_text);
        old_x_text = x_text

    return vis


def draw_hsv(flow):
    h, w = flow.shape[:2]
    fx, fy = flow[:,:,0], flow[:,:,1]
    ang = np.arctan2(fy, fx) + np.pi
    v = np.sqrt(fx*fx+fy*fy)
    hsv = np.zeros((h, w, 3), np.uint8)
    hsv[...,0] = ang*(180/np.pi/2)
    hsv[...,1] = 255
    hsv[...,2] = np.minimum(v*4, 255)
    bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
    return bgr


def warp_flow(img, flow):
    h, w = flow.shape[:2]
    flow = -flow
    flow[:,:,0] += np.arange(w)
    flow[:,:,1] += np.arange(h)[:,np.newaxis]
    res = cv.remap(img, flow, None, cv.INTER_LINEAR)
    return res

if __name__ == '__main__':
    import sys
    print(__doc__)
    try:
        fn = sys.argv[1]
    except IndexError:
        fn = 0

    #cam = video.create_capture(fn)
    cam = cv.VideoCapture(fn)

    ret, prev = cam.read()
    prevgray = cv.cvtColor(prev, cv.COLOR_BGR2GRAY)
    show_hsv = False
    show_glitch = False
    cur_glitch = prev.copy()

    while True:
        ret, img = cam.read()
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        flow = cv.calcOpticalFlowFarneback(prevgray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        prevgray = gray

        cv.imshow('flow', draw_flow(gray, flow))
        if show_hsv:
            cv.imshow('flow HSV', draw_hsv(flow))
        if show_glitch:
            cur_glitch = warp_flow(cur_glitch, flow)
            cv.imshow('glitch', cur_glitch)



        ch = cv.waitKey(5)
        if ch == 27:
            break
        if ch == ord('1'):
            show_hsv = not show_hsv
            print('HSV flow visualization is', ['off', 'on'][show_hsv])
        if ch == ord('2'):
            show_glitch = not show_glitch
            if show_glitch:
                cur_glitch = img.copy()
            print('glitch is', ['off', 'on'][show_glitch])
    cv.destroyAllWindows()
