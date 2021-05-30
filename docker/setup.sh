#! /bin/bash
SCRIPT_LOCATION=$(cd $(dirname $0) && pwd -P)
source $SCRIPT_LOCATION/vars.sh

echo "This script will build a habitat container image."
echo ""
read -p "Do you want to continue? (y/n) " -r
if [[ ! $REPLY =~ ^[Yy]$ ]]
then
  exit 1
fi

docker build -t $IMAGE_NAME .
