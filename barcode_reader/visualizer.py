import cv2
import numpy as np

def draw_reading_zone(frame,polygon):
    if not polygon:return
    pts=np.asarray(polygon,dtype=np.int32)
    overlay=frame.copy()
    cv2.fillPoly(overlay,[pts],(255,180,0))
    cv2.addWeighted(overlay,.08,frame,.92,0,frame)
    cv2.polylines(frame,[pts],True,(255,200,0),2,cv2.LINE_AA)
    x,y=pts[0]
    cv2.putText(frame,"READING ZONE",(x+4,max(18,y-7)),cv2.FONT_HERSHEY_SIMPLEX,
                .48,(255,255,255),2,cv2.LINE_AA)

def draw_barcode(frame,result):
    x,y,w,h=result.rect
    color=(0,220,0) if result.in_zone else (0,0,255)
    if result.polygon:
        cv2.polylines(frame,[np.asarray(result.polygon,dtype=np.int32)],True,color,2,cv2.LINE_AA)
    else:
        cv2.rectangle(frame,(x,y),(x+w,y+h),color,2,cv2.LINE_AA)
    label=f"{result.data} [{result.barcode_type}]"
    font=cv2.FONT_HERSHEY_SIMPLEX; scale=.55; thick=2
    (tw,th),base=cv2.getTextSize(label,font,scale,thick)
    yy=max(th+8,y-6); xx=max(0,min(x,frame.shape[1]-tw-12))
    cv2.rectangle(frame,(xx,yy-th-8),(xx+tw+10,yy+base),(0,0,0),-1)
    cv2.rectangle(frame,(xx,yy-th-8),(xx+tw+10,yy+base),color,1)
    cv2.putText(frame,label,(xx+5,yy),font,scale,(255,255,255),thick,cv2.LINE_AA)

def draw_hud(frame,count,fps,frame_no,total):
    text=f"BARCODE READER | FOUND: {count} | FPS: {fps:.1f} | FRAME: {frame_no}/{total or '?'}"
    cv2.rectangle(frame,(0,0),(frame.shape[1],28),(15,15,15),-1)
    cv2.putText(frame,text,(8,19),cv2.FONT_HERSHEY_SIMPLEX,.45,(0,255,120),1,cv2.LINE_AA)
