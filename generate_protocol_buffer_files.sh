#!/bin/sh

echo "Generating objects detection files..."
cd src/objects_detection/
protoc --cpp_out=./ detector_model.proto detections.proto
protoc --python_out=../../tools/objects_detection/ detector_model.proto detections.proto

# FIXME add the ground plane and calibration protocol buffer models
echo "(Ground plane and video input files not yet handled by this script)"
cd ../..
cd src/video_input/calibration
protoc --cpp_out=./ calibration.proto

cd ../../..
cd src/helpers/data
protoc --cpp_out=./ DataSequenceHeader.proto

echo "End of game. Have a nice day!"
