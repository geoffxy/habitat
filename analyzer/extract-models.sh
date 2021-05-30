#! /bin/bash

set -e

if [ -z $1 ]; then
  >&2 echo "Usage: $0 path/to/habitat-models.tar.gz"
  >&2 echo ""
  >&2 echo "This script extracts and installs Habitat's pre-trained models."
  exit 1
fi

archive_loc=$(pwd)/$1

script_loc=$(cd $(dirname $0) && pwd -P)
cd $script_loc

tar xzf $archive_loc -C habitat/data/
cd habitat/data/

./verify.sh
