import json,time
from pathlib import Path
import cv2
from .reader import BarcodeReader
from .zones import normalize_polygon
from .visualizer import draw_reading_zone,draw_barcode,draw_hud

class BarcodePipeline:
    def __init__(self,args):
        self.args=args
        self.cfg=json.loads(Path(args.config).read_text(encoding="utf-8"))
        self.reader=BarcodeReader(self.cfg)
        self.seen={}
        self.records=[]

    def run(self):
        source=int(self.args.source) if str(self.args.source).isdigit() else self.args.source
        cap=cv2.VideoCapture(source)
        if not cap.isOpened():raise RuntimeError(f"Cannot open source: {self.args.source}")
        width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
        height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
        fps_in=cap.get(cv2.CAP_PROP_FPS);fps_in=fps_in if fps_in and fps_in>1 else 25.0
        total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        output=Path(self.args.output);output.parent.mkdir(parents=True,exist_ok=True)
        writer=cv2.VideoWriter(str(output),cv2.VideoWriter_fourcc(*"mp4v"),fps_in,(width,height))
        if not writer.isOpened():raise RuntimeError(f"Cannot create output video: {output}")
        zone=None
        if self.cfg.get("reading_zone") and not self.args.no_zone:
            zone=normalize_polygon(self.cfg["reading_zone"],width,height)
        start=time.time();frame_no=0;fps_smooth=0.0
        try:
            while True:
                ok,frame=cap.read()
                if not ok:break
                frame_no+=1;t0=time.time()
                results=self.reader.read(frame,zone)
                draw_reading_zone(frame,zone)
                for result in results:
                    draw_barcode(frame,result)
                    key=(result.barcode_type,result.data)
                    if key not in self.seen:
                        self.seen[key]={"barcode":result.data,"type":result.barcode_type,
                            "first_frame":frame_no,"first_time_sec":round(frame_no/fps_in,3),
                            "rect":list(result.rect)}
                        print(f"[BARCODE] {result.data} [{result.barcode_type}]")
                elapsed=max(time.time()-t0,1e-6)
                fps_now=1/elapsed
                fps_smooth=fps_now if fps_smooth==0 else .9*fps_smooth+.1*fps_now
                draw_hud(frame,len(results),fps_smooth,frame_no,total)
                writer.write(frame)
                if self.args.display:
                    cv2.imshow("Barcode Detection and Recognition",frame)
                    if cv2.waitKey(1)&0xFF==ord("q"):break
        finally:
            cap.release();writer.release()
            if self.args.display:cv2.destroyAllWindows()
        report={"source":str(self.args.source),"frames_processed":frame_no,
            "runtime_sec":round(time.time()-start,2),"unique_barcodes":len(self.seen),
            "barcodes":list(self.seen.values())}
        report_path=output.with_name(output.stem+"_report.json")
        report_path.write_text(json.dumps(report,indent=2),encoding="utf-8")
        print(f"[DONE] Annotated video: {output}")
        print(f"[DONE] Barcode report: {report_path}")
