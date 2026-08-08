import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

from ultralytics import solutions
from ultralytics import YOLO



if __name__ == '__main__':
    # Pass a model as an argument
    # solutions.inference(model="assets/pave-yolo8n-seg.pt")
    results = YOLO("assets/pave-yolo8n-seg.pt").predict(source=0)


    ### Make sure to run the file using command `streamlit run <file-name.py>`