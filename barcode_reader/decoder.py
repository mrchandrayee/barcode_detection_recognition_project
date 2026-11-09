from __future__ import annotations
from typing import List
import cv2
import numpy as np
from .models import BarcodeResult

def validate_barcode_checksum(data: str, barcode_type: str) -> bool:
    """Validate EAN-13, EAN-8 and UPC-A check digits."""
    if barcode_type=="EAN13" and len(data)==13 and data.isdigit():
        digits=[int(c) for c in data]
        total=sum(digits[i]*(1 if i%2==0 else 3) for i in range(12))
        return (10-total%10)%10==digits[12]
    if barcode_type=="EAN8" and len(data)==8 and data.isdigit():
        digits=[int(c) for c in data]
        total=sum(digits[i]*(3 if i%2==0 else 1) for i in range(7))
        return (10-total%10)%10==digits[7]
    if barcode_type=="UPCA" and len(data)==12 and data.isdigit():
        digits=[int(c) for c in data]
        total=sum(digits[i]*(3 if i%2==0 else 1) for i in range(11))
        return (10-total%10)%10==digits[11]
    return True

def preprocess_grayscale(image: np.ndarray) -> np.ndarray:
    if len(image.shape)==3:
        return cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    return image.copy()

def preprocess_clahe(gray: np.ndarray, clip_limit=2.0, grid_size=8) -> np.ndarray:
    clahe=cv2.createCLAHE(clipLimit=float(clip_limit),tileGridSize=(int(grid_size),int(grid_size)))
    return clahe.apply(gray)

def preprocess_threshold(gray: np.ndarray, blur_kernel=3, block_size=31, c=7) -> np.ndarray:
    k=max(1,int(blur_kernel))
    if k%2==0:k+=1
    blurred=cv2.GaussianBlur(gray,(k,k),0)
    block=max(3,int(block_size))
    if block%2==0:block+=1
    return cv2.adaptiveThreshold(blurred,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY,block,int(c))

def preprocess_sharpen(gray: np.ndarray) -> np.ndarray:
    kernel=np.array([[-1,-1,-1],[-1,9,-1],[-1,-1,-1]],dtype=np.float32)
    return cv2.filter2D(gray,-1,kernel)

def decode_pyzbar(image: np.ndarray):
    try:
        from pyzbar.pyzbar import decode
        return decode(image)
    except Exception:
        return []

def _type_name(raw_type):
    mapping={"EAN13":"EAN13","EAN-13":"EAN13","EAN8":"EAN8","EAN-8":"EAN8",
             "UPCA":"UPCA","UPC-A":"UPCA","QRCODE":"QRCODE","CODE128":"CODE128",
             "CODE39":"CODE39","ITF":"I25","I25":"I25","PDF417":"PDF417",
             "DATABAR":"DATABAR"}
    return mapping.get(str(raw_type).upper(),str(raw_type).upper())

def _scale_polygon(poly, scale):
    if scale==1.0:return poly
    return [(int(x/scale),int(y/scale)) for x,y in poly]

def _scale_rect(rect, scale):
    if scale==1.0:return rect
    x,y,w,h=rect
    return (int(x/scale),int(y/scale),int(w/scale),int(h/scale))

def try_decode(image: np.ndarray, scales=(1.0,1.5,2.0), clip_limit=2.0,
               grid_size=8, blur_kernel=3, block_size=31, adaptive_c=7,
               validate_checksums=False) -> List[BarcodeResult]:
    results=[]; found_data=set()
    h0,w0=image.shape[:2]
    for scale in scales:
        if scale<=0:continue
        if scale!=1.0:
            nw,nh=int(w0*scale),int(h0*scale)
            if nw<10 or nh<10:continue
            work=cv2.resize(image,(nw,nh),interpolation=cv2.INTER_CUBIC if scale>1 else cv2.INTER_AREA)
        else:work=image
        gray=preprocess_grayscale(work)
        variants=[
            ("original",gray),
            ("clahe",preprocess_clahe(gray,clip_limit,grid_size)),
            ("threshold",preprocess_threshold(gray,blur_kernel,block_size,adaptive_c)),
            ("sharpen",preprocess_sharpen(gray)),
        ]
        for _,processed in variants:
            for barcode in decode_pyzbar(processed):
                data=barcode.data.decode("utf-8",errors="replace").strip()
                btype=_type_name(barcode.type)
                if not data or data in found_data:continue
                if validate_checksums and not validate_barcode_checksum(data,btype):continue
                found_data.add(data)
                poly=[]
                if getattr(barcode,"polygon",None):
                    poly=[(int(pt.x),int(pt.y)) for pt in barcode.polygon]
                rect=barcode.rect
                rect=(int(rect.left),int(rect.top),int(rect.width),int(rect.height))
                results.append(BarcodeResult(data,btype,_scale_polygon(poly,scale),
                                             _scale_rect(rect,scale),scale,True))
        if len(results)>=12:break
    # OpenCV's built-in detector is a fallback when zbar is not installed / returns none.
    if not results and hasattr(cv2,"barcode_BarcodeDetector"):
        try:
            detector=cv2.barcode_BarcodeDetector()
            ok, decoded, types, corners=cv2.barcode_BarcodeDetector().detectAndDecode(image)
            if ok and decoded:
                for i,data in enumerate(decoded):
                    if not data or data in found_data:continue
                    found_data.add(data)
                    pts=[]
                    if corners is not None and i<len(corners):
                        pts=[(int(q[0]),int(q[1])) for q in corners[i]]
                    if pts:
                        xs=[p[0] for p in pts];ys=[p[1] for p in pts]
                        rect=(min(xs),min(ys),max(xs)-min(xs),max(ys)-min(ys))
                    else:rect=(0,0,0,0)
                    typ=_type_name(types[i] if types is not None and i<len(types) else "UNKNOWN")
                    if not validate_checksums or validate_barcode_checksum(data,typ):
                        results.append(BarcodeResult(data,typ,pts,rect,1.0,True))
        except Exception:
            pass
    return results
