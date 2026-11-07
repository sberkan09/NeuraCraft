#!/usr/bin/env bash
# A few motions from the CMU Graphics Lab Motion Capture Database (mocap.cs.cmu.edu; "free for use in
# research and commercial projects worldwide"), as Bruce Hahne's BVH conversion shares them, for the
# animate step: walk, run, jump, dance and cartwheel (120 frames a second, +Y up, facing +Z).
#   scripts/download_cmu_motions.sh [DIR]      (default data/motions)
# CMU asks for this acknowledgement: "The data used in this project was obtained from mocap.cs.cmu.edu.
# The database was created with funding from NSF EIA-0196217."
set -euo pipefail
DIR=${1:-data/motions}
BASE=https://raw.githubusercontent.com/Shriinivas/cmubvh/main
mkdir -p "$DIR"
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
fetch() {  # name, path in the repository, file in the zip
    [ -f "$DIR/$1.bvh" ] && return
    curl -sfL -o "$TMP/m.zip" "$BASE/$2" && unzip -o -q "$TMP/m.zip" "$3" -d "$TMP" && mv "$TMP/$3" "$DIR/$1.bvh"
    echo "$DIR/$1.bvh"
}
fetch walk Sequence-001-009/02/Data/02_01.zip 02_01.bvh
fetch run Sequence-001-009/09/Data/09_01.zip 09_01.bvh
fetch jump Sequence-015-019/16/Data/16_01.zip 16_01.bvh
fetch dance Sequence-001-009/05/Data/05_02.zip 05_02.bvh
fetch cartwheel Sequence-046-056/49/Data/49_06.zip 49_06.bvh
