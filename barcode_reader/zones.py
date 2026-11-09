import cv2
import numpy as np

def normalize_polygon(points,width,height):
    out=[]
    for x,y in points:
        if 0<=x<=1 and 0<=y<=1:out.append((int(x*width),int(y*height)))
        else:out.append((int(x),int(y)))
    return out

def point_in_polygon(point,polygon):
    if len(polygon)<3:return True
    return cv2.pointPolygonTest(np.asarray(polygon,dtype=np.int32),
                               (float(point[0]),float(point[1])),False)>=0

def scale_polygon(polygon,scale):
    return [(int(x/scale),int(y/scale)) for x,y in polygon]
