#!/usr/bin/env python
import os
import sys
import getopt
import time 
import cv2
from PIL import Image, ImageDraw
import numpy as np
### Add load path
base = os.path.dirname(__file__)
if '' == base:
    base = '.'
sys.path.append('%s/../'%base)

from cascade import *
from utils   import *

def nms_detections(dets, score, overlap=0.3):
  """
  Non-maximum suppression: Greedily select high-scoring detections and
  skip detections that are significantly covered by a previously
  selected detection.

  This version is translated from Matlab code by Tomasz Malisiewicz,
  who sped up Pedro Felzenszwalb's code.

  Parameters
  ----------
  dets: ndarray
    each row is ['xmin', 'ymin', 'xmax', 'ymax', 'score']
  overlap: float
    minimum overlap ratio (0.3 default)

  Output
  ------
  dets: ndarray
    remaining after suppression.
  """
  x1 = dets[:, 0]
  y1 = dets[:, 1]
  x2 = dets[:, 2] + x1
  y2 = dets[:, 3] + y1
  ind = np.argsort(score)

  w = x2 - x1
  h = y2 - y1
  area = (w * h).astype(float)

  pick = []
  while len(ind) > 0:
    i = ind[-1]
    pick.append(i)
    ind = ind[:-1]

    xx1 = np.maximum(x1[i], x1[ind])
    yy1 = np.maximum(y1[i], y1[ind])
    xx2 = np.minimum(x2[i], x2[ind])
    yy2 = np.minimum(y2[i], y2[ind])

    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)

    wh = w * h
    o = wh / (area[i] + area[ind] - wh)

    ind = ind[np.nonzero(o <= overlap)[0]]

  return dets[pick, :]

def usage():
    print("-----------------------------------------------")
    print('[[Usage]]::')
    print('\t%s [Paras] train.model test.jpg'%(sys.argv[0]))
    print("[[Paras]]::")
    print("\thelp|h : Print the help information ")
    print("-----------------------------------------------")
    return 

def detect_jpg(detector, jpg_path):
    if not os.path.exists(jpg_path):
        raise Exception("Image not exist:"%(jpg_path))
    
    src_img = Image.open(jpg_path)
    if 'L' != src_img.mode:
        img = src_img.convert("L")
    else:
        img = src_img
        
    maxSide = 200.0
    w, h = img.size
    scale = max(1, max(w/maxSide, h/maxSide))
    ws = int(w/scale + 0.5)
    hs = int(h/scale + 0.5)
    img = img.resize((ws,hs), Image.NEAREST)
    imgArr = np.asarray(img).astype(np.uint8)
    
    time_b = time.time()
    rects, confs = detector.detect(imgArr, 1.1, 2)
    t = getTimeByStamp(time_b, time.time(), 'sec')    

    ### TODO: Use NMS to merge the candidate rects and show the landmark, Now merge the rects with opencv,   
    #res = cv2.groupRectangles(rects, 3, 0.2)
    rects = nms_detections(np.asarray(rects), confs)
    res = rects[0]

    ### Show Result
    draw = ImageDraw.Draw(img)
    draw.rectangle((res[0], 
        res[1],
        res[0]+res[2], 
        res[1]+res[3]), 
        outline = "red")

    print("Detect time : %f s"%t)        
    img.show()
    #########################################
    
    if len(rects) > 0:
        num, _ = rects.shape
        for i in xrange(num):
            rects[i][0] = int(rects[i][0]*scale +0.5)
            rects[i][1] = int(rects[i][1]*scale +0.5)
            rects[i][2] = int(rects[i][2]*scale +0.5)
            rects[i][3] = int(rects[i][3]*scale +0.5)     
    return

    
def main(argv):
    try:
        options, args = getopt.getopt(argv, 
                                      "h", 
                                      ["help"])
    except getopt.GetoptError:  
        usage()
        return
    
    if len(argv) < 2:
        usage()
        return

    for opt , arg in options:
        if opt in ('-h', '--help'):
            usage()
            return
    
    detector = JointCascador()  
    detector = detector.loadModel(args[0])    
    detect_jpg(detector, args[1])    
            
if __name__ == '__main__' :
    main(sys.argv[1:])
