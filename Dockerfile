FROM nvcr.io/nvidia/pytorch:22.04-py3

WORKDIR /yolov9
RUN rm -rf /opt/conda/lib/python3.8/site-packages/cv2/
COPY requirements.txt requirements.txt
RUN pip3 install --upgrade --no-cache-dir pip && \ 
    pip3 install --no-cache-dir -r requirements.txt && \
    pip3 install wandb
