from .decoder import try_decode
from .zones import point_in_polygon

class BarcodeReader:
    def __init__(self,config):
        self.cfg=config

    def read(self,frame,reading_zone=None):
        h,w=frame.shape[:2]
        if reading_zone and len(reading_zone)>=3:
            # Limit processing to the reading-zone bounding rectangle for speed.
            import cv2
            import numpy as np
            mask=np.zeros((h,w),dtype=np.uint8)
            cv2.fillPoly(mask,[np.asarray(reading_zone,dtype=np.int32)],255)
            x,y,bw,bh=cv2.boundingRect(np.asarray(reading_zone,dtype=np.int32))
            crop=frame[y:y+bh,x:x+bw]
            decoded=try_decode(crop,
                scales=self.cfg["scales"],clip_limit=self.cfg["clahe_clip_limit"],
                grid_size=self.cfg["clahe_grid_size"],blur_kernel=self.cfg["blur_kernel"],
                block_size=self.cfg["adaptive_block_size"],adaptive_c=self.cfg["adaptive_c"],
                validate_checksums=self.cfg["validate_checksums"])
            results=[]
            for item in decoded:
                px,py,rw,rh=item.rect
                item.rect=(px+x,py+y,rw,rh)
                item.polygon=[(qx+x,qy+y) for qx,qy in item.polygon]
                item.in_zone=point_in_polygon(item.center,reading_zone)
                if item.in_zone:results.append(item)
            return results
        return try_decode(frame,scales=self.cfg["scales"],
            clip_limit=self.cfg["clahe_clip_limit"],grid_size=self.cfg["clahe_grid_size"],
            blur_kernel=self.cfg["blur_kernel"],block_size=self.cfg["adaptive_block_size"],
            adaptive_c=self.cfg["adaptive_c"],validate_checksums=self.cfg["validate_checksums"])
