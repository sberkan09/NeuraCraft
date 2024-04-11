#!/bin/bash

BUILD_DIRECTORY="./build"
BINARY_DIRECTORY="./bin"
SOURCE_DIRECTORY="./src"
BINARY_PATH="$BINARY_DIRECTORY/NeuraCraft"

function check_modified() {
    if [ ! -f "$BINARY_PATH" ]; then
        return 0
    fi

    local build_time=$(stat -c "%Y" "$BINARY_PATH")

    for file in $(find $SOURCE_DIRECTORY -name '*.cpp' -or -name '*.h'); do
        local file_time=$(stat -c "%Y" "$file")
        if [ "$file_time" -gt "$build_time" ]; then
            return 0
        fi
    done

    return 1
}

if check_modified; then
    echo "Source files have been modified, building..."

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
else
    echo "No source files modified, skipping build..."
fi

if [ -f "$BINARY_PATH" ]; then
    ./"$BINARY_PATH"
else
    echo "Executable not found: $BINARY_PATH"
fi
