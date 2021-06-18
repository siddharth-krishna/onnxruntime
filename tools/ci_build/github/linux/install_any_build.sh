#!/bin/bash -x
set -e

readonly REGION="westus2"
readonly CLUSTER_NAME="onnxr"

readonly DOWNLOAD_URI=https://anybuild${CLUSTER_NAME}${REGION}.blob.core.windows.net/clientreleases
readonly ANYBUILD_HOME="$HOME/.local/share/Microsoft/AnyBuild"

if [[ ! -f "$ANYBUILD_HOME/AnyBuild.sh" ]]; then
    echo
    echo "=== Installing AnyBuild client ==="
    echo
    wget -O bootstrapper.sh ${DOWNLOAD_URI}/bootstrapper.sh
    bash bootstrapper.sh ${DOWNLOAD_URI} Dogfood
    rm bootstrapper.sh
fi