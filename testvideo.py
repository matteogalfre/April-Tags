#!/usr/bin/env python
# -*- coding: utf-8 -*-
import copy
import time
import argparse
import cv2 as cv
from pupil_apriltags import Detector
import matplotlib.pyplot as plt

def get_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--device", type=int, default=0)
    # tag36h11
    parser.add_argument("--families", type=str, default='tag36h11')
    parser.add_argument("--nthreads", type=int, default=1)
    parser.add_argument("--quad_decimate", type=float, default=1.0)
    parser.add_argument("--quad_sigma", type=float, default=1.0)
    parser.add_argument("--refine_edges", type=int, default=1)
    parser.add_argument("--decode_sharpening", type=float, default=1)
    parser.add_argument("--debug", type=int, default=0)
    
    # parser.add_argument("--device", type=int, default=0)
    # tag16h5
    # parser.add_argument("--families", type=str, default='tag16h5')
    # parser.add_argument("--nthreads", type=int, default=1)
    # #La détection des quads peut être effectuée sur une image de résolution inférieure, améliorant la vitesse au détriment de la précision de la pose et d'une légère diminution du taux de détection.
    # #Le décodage de la charge utile binaire est toujours effectué à pleine résolution.  Réglez-le sur 1,0 pour utiliser la pleine résolution.
    # parser.add_argument("--quad_decimate", type=float, default=0.1)
    # #Quel flou gaussien doit être appliqué à l'image segmentée.  Le paramètre est l'écart type en pixels.  Les images très bruitées bénéficient de valeurs non nulles (par exemple 0,8)
    # parser.add_argument("--quad_sigma", type=float, default=1.0)
    # parser.add_argument("--refine_edges", type=int, default=1)
    # #Quel degré de netteté faut-il apporter aux images décodées ?
    # parser.add_argument("--decode_sharpening", type=float, default=0)
    # parser.add_argument("--debug", type=int, default=0)

    args = parser.parse_args()
    return args

def main():
    
    menu = {}
    menu['[1]']="Video tag16h5" 
    menu['[2]']="Video tag36h11"
    menu['[3]']="WebCam tag16h5"
    menu['[4]']="WebCam tag36h11"

    while True: 
        options=menu.keys()
        for entry in options: 
            print(entry, menu[entry])

        selection = input("Select option :") 
        
        if selection =='1':
            cap = cv.VideoCapture('video16h5.mp4')
            families = 'tag16h5'
            break
        elif selection == '2':
            cap = cv.VideoCapture('video36h11.mp4')
            families = 'tag36h11'
            break
        elif selection == '3':
            cap = cv.VideoCapture(0)
            families = 'tag16h5'
            break
        elif selection == '4':
            cap = cv.VideoCapture(0)
            families = 'tag36h11'
            break
        else:
            print("Unknown Option Selected !")
            
    args = get_args()

    cap_device = args.device

    nthreads = args.nthreads
    quad_decimate = args.quad_decimate
    quad_sigma = args.quad_sigma
    refine_edges = args.refine_edges
    decode_sharpening = args.decode_sharpening
    debug = args.debug

    # Detector
    at_detector = Detector(
        families=families,
        nthreads=nthreads,
        quad_decimate=quad_decimate,
        quad_sigma=quad_sigma,
        refine_edges=refine_edges,
        decode_sharpening=decode_sharpening,
        debug=debug
    )
    
    # Read until video is completed
    while(cap.isOpened()):
        cap.set(cv.CAP_PROP_FRAME_WIDTH, 3)
        cap.set(cv.CAP_PROP_FRAME_HEIGHT, 4)
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret == True:
        
            # Display the resulting frame
            image = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            tags = at_detector.detect(
                image,
                estimate_tag_pose=False,
                camera_params=None,
                tag_size=None,
            )

            debug_image = draw_tags(frame, tags, 10)
            
    

        cv.imshow('AprilTag Detect Demo', debug_image)
        if cv.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv.destroyAllWindows()

    
    
def draw_tags(
    image,
    tags,
    elapsed_time,
):
    for tag in tags:
        tag_family = tag.tag_family
        tag_id = tag.tag_id
        center = tag.center
        corners = tag.corners
        print(str(tag_id))
        center = (int(center[0]), int(center[1]))
        corner_01 = (int(corners[0][0]), int(corners[0][1]))
        corner_02 = (int(corners[1][0]), int(corners[1][1]))
        corner_03 = (int(corners[2][0]), int(corners[2][1]))
        corner_04 = (int(corners[3][0]), int(corners[3][1]))

        cv.circle(image, (center[0], center[1]), 5, (0, 0, 255), 2)

        cv.line(image, (corner_01[0], corner_01[1]),
                (corner_02[0], corner_02[1]), (255, 0, 0), 2)
        cv.line(image, (corner_02[0], corner_02[1]),
                (corner_03[0], corner_03[1]), (255, 0, 0), 2)
        cv.line(image, (corner_03[0], corner_03[1]),
                (corner_04[0], corner_04[1]), (0, 255, 0), 2)
        cv.line(image, (corner_04[0], corner_04[1]),
                (corner_01[0], corner_01[1]), (0, 255, 0), 2)

        cv.putText(image, str(tag_id), (center[0] - 10, center[1] - 10),
                   cv.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2, cv.LINE_AA)
    return image


if __name__ == '__main__':
    main()
    
