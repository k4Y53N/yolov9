#!/bin/bash
NAME="model-name"
DATA="path/to/dataset.yml"
CFG="models/detect/gelan-c.yaml"

BATCH=8
WORKERS=4
SIZE=640
EPOCHS=1000

WEIGHTS_PATH="runs/train/$NAME/weights/last.pt"
COMMON_PARAMS="
--workers $WORKERS
--device 0
--batch $BATCH
--name $NAME
--data $DATA
--img $SIZE
--cfg $CFG
--epochs $EPOCHS
--hyp data/hyps/hyp.scratch-high.yaml
--min-items 0
--close-mosaic 15
"

while true; do
    if [ -f "$WEIGHTS_PATH" ]; then
        python3 train.py $COMMON_PARAMS --resume "$WEIGHTS_PATH"
    else
        python3 train.py $COMMON_PARAMS
    fi

    if [ $? -eq 143 ]; then
        echo "Received SIGTERM. Exiting..."
        break
    fi

    if [ $? -eq 0 ]; then
        echo "Training completed successfully"
        break
    fi

    echo "Training failed. Retrying in 10 seconds..."
    sleep 10
done
