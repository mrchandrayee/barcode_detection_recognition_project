#!/usr/bin/env python3
import argparse
from barcode_reader.pipeline import BarcodePipeline

def parse_args():
    p=argparse.ArgumentParser(description="Barcode Detection and Recognition")
    p.add_argument("--source",default="0",help="Camera index or video path")
    p.add_argument("--config",default="config/settings.json")
    p.add_argument("--output",default="output/barcode_output.mp4")
    p.add_argument("--display",action="store_true")
    p.add_argument("--no-zone",action="store_true",help="Scan the full frame")
    return p.parse_args()

if __name__=="__main__":
    args=parse_args()
    BarcodePipeline(args).run()
