#!/bin/bash

BUILD_DIRECTORY="./build"
BINARY_DIRECTORY="./bin"

if [ ! -d "$BUILD_DIRECTORY" ]; then
    mkdir "$BUILD_DIRECTORY"
else
    rm -r "$BUILD_DIRECTORY"
    mkdir "$BUILD_DIRECTORY"
fi

cd "$BUILD_DIRECTORY"
cmake ..
make
cd ..

if [ -f "$BINARY_DIRECTORY/NeuraCraft" ]; then
    ./"$BINARY_DIRECTORY/NeuraCraft"
else
    echo "Executable not found: $BINARY_DIRECTORY/GameCpp"
fi