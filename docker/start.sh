#! /bin/bash

# This script drops you into a habitat container. The container image must
# exist. If it does not exist, build it first using the setup.sh script.
#
# If no containers exist (stopped or running), this script will run a new container.
# You can optionally pass in another directory to mount as an argument to this script.
# If no arguments are provided to the script, your home directory will be mounted
# inside the container at ~/home.
#
# If a running container exists, this script will just start a new bash shell inside
# the container.
#
# If a stopped container exists, this script will restart it.

SCRIPT_LOCATION=$(cd $(dirname $0) && pwd -P)
source $SCRIPT_LOCATION/vars.sh

UNAME=$(id -un)
CONTAINER_NAME=$IMAGE_NAME-$UNAME

if [ -z $1 ]; then
  MOUNT_VOL=$(cd ~ && pwd):/home/$UNAME/home
else
  ABS_MOUNT_DIR=$(cd $1 && pwd)
  MOUNT_DIR=$(basename $ABS_MOUNT_DIR)
  MOUNT_VOL=$ABS_MOUNT_DIR:/home/$UNAME/$MOUNT_DIR
fi

RUNNING=$(docker ps --filter "status=running" --filter "name=$CONTAINER_NAME" --format "{{.ID}}")
EXITED=$(docker ps --filter "status=exited" --filter "name=$CONTAINER_NAME" --format "{{.ID}}")

if [ -z "$RUNNING" ] && [ -z "$EXITED" ]; then
  # Container was never started

  # NOTE: For stable all reduce measurements, it's important to ensure that the
  #       shared memory limits are increased using
  #
  #         --shm-size=1g --ulimit memlock=-1
  #
  docker run -ti \
    -e "CONTAINER_UID=$(id -u)" \
    -e "CONTAINER_UNAME=$(id -un)" \
    --name $CONTAINER_NAME \
    --volume $MOUNT_VOL \
    --runtime=nvidia \
    --workdir=/home/$UNAME \
    --shm-size=1g \
    --ulimit memlock=-1 \
    $IMAGE_NAME
elif [ -z "$RUNNING" ]; then
  # Container exited but was not removed. We can restart it.
  docker start -ai $EXITED
else
  # Already running, so just attach
  docker exec -it $CONTAINER_NAME \
    /usr/local/bin/gosu \
    $UNAME \
    /bin/bash
fi
