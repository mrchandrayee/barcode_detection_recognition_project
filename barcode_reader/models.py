from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class BarcodeResult:
    data: str
    barcode_type: str
    polygon: List[Tuple[int,int]]
    rect: Tuple[int,int,int,int]  # x, y, width, height
    scale_found: float = 1.0
    in_zone: bool = True

    def __str__(self):
        zone_str = "[IN ZONE]" if self.in_zone else "[OUT OF ZONE]"
        return f"{self.barcode_type} {self.data} {zone_str}"

    @property
    def center(self):
        if self.polygon and len(self.polygon)>3:
            cx=sum(p[0] for p in self.polygon)//len(self.polygon)
            cy=sum(p[1] for p in self.polygon)//len(self.polygon)
            return cx,cy
        x,y,w,h=self.rect
        return x+w//2,y+h//2
