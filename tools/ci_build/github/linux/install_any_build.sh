#!/bin/bash -x
set -e

readonly REGION="westus2"
readonly CLUSTER_NAME="onnxr"

readonly containerUrl=https://anybuild${CLUSTER_NAME}${REGION}.blob.core.windows.net/clientreleases
readonly ANYBUILD_HOME="$HOME/.local/share/Microsoft/AnyBuild"
readonly channel="Dogfood"

if [[ ! -f "$ANYBUILD_HOME/AnyBuild.sh" ]]; then
    echo
    echo "=== Installing AnyBuild client ==="
    echo
    	

	# .Net Core follows this guidance as well: https://specifications.freedesktop.org/basedir-spec/basedir-spec-latest.html
	localAppData=${XDG_DATA_HOME:-$HOME/.local/share}

	anyBuildClientBaseDir=$localAppData/Microsoft/AnyBuild
	mkdir -p $anyBuildClientBaseDir

	echo "Downloading and running AnyBuildUpdater from $channel channel from $containerUrl"
	wget $containerUrl/ReleasesLinux.json -O $anyBuildClientBaseDir/ReleasesLinux.json

	currentRelease=$(cat $anyBuildClientBaseDir/ReleasesLinux.json | python -c "import sys, json; print(json.load(sys.stdin)['${channel}Channel']['Release'])")

	updaterDir=$anyBuildClientBaseDir/BootstrapUpdater_$currentRelease
	if [ -d $updaterDir ]; then
	  echo "Deleting $updaterDir"
	  rm -r $updaterDir
	fi

	updaterArchive="$updaterDir/AnyBuildUpdater.tar.gz"
	mkdir -p $updaterDir
	wget $containerUrl/$currentRelease/Linux/AnyBuildUpdater.tar.gz -O $updaterArchive
	tar xzf $updaterArchive --directory $updaterDir

	updaterBinary=$updaterDir/AnyBuildUpdater

	# Ensure the permission to execute is set because if AnyBuild validation/release workflow creates the resulting archive on Windows machine which doesn't preserve this flag.
	chmod +x $updaterBinary

	echo "Executing: $updaterBinary --Channel $channel --ReleaseContainerUrl $containerUrl"
	$updaterBinary --Channel $channel --ReleaseContainerUrl $containerUrl
fi