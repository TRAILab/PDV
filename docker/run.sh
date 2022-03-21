#!/bin/bash

# PARAMETERS
KITTI_TRAIN=$(readlink -f ../data/kitti/training)
KITTI_TEST=$(readlink -f ../data/kitti/testing)
WAYMO_RAW=$(readlink -f ../data/waymo/raw_data)
WAYMO_PROCESSED=$(readlink -f ../data/waymo/waymo_processed_data)

# Setup volume linking
CUR_DIR=$(pwd)
PROJ_DIR=$(dirname $CUR_DIR)
KITTI_TRAIN=$KITTI_TRAIN:/PDV/data/kitti/training
KITTI_TEST=$KITTI_TEST:/PDV/data/kitti/testing
WAYMO_RAW=$WAYMO_RAW:/PDV/data/waymo/raw_data
WAYMO_PROCESSED=$WAYMO_PROCESSED:/PDV/data/waymo/waymo_processed_data

PCDET_VOLUMES=""
for entry in $PROJ_DIR/pcdet/*
do
    name=$(basename $entry)

    if [ "$name" != "version.py" ] && [ "$name" != "ops" ]
    then
        PCDET_VOLUMES+="--volume $entry:/PDV/pcdet/$name "
    fi
done

docker run -it \
        --runtime=nvidia \
        --net=host \
        --privileged=true \
        --ipc=host \
        --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
        --volume="$XAUTHORITY:/root/.Xauthority:rw" \
        --env="DISPLAY" \
        --env="QT_X11_NO_MITSHM=1" \
        --hostname="inside-DOCKER" \
        --name="PDV" \
        --volume $KITTI_TRAIN \
        --volume $KITTI_TEST \
        --volume $WAYMO_RAW \
        --volume $WAYMO_PROCESSED \
        --volume $PROJ_DIR/data:/PDV/data \
        --volume $PROJ_DIR/output:/PDV/output \
        --volume $PROJ_DIR/tools:/PDV/tools \
        $PCDET_VOLUMES \
        --rm \
        pdv bash
