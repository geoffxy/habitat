#! /bin/bash

function usage() {
  echo "Usage: $0 <output file name> path/to/libtorch.so"
  exit 1
}

if [ -z "$1" ] || [ -z "$2" ]; then
  usage "$@"
fi

DATABASE_NAME=$1
LIBTORCH_PATH=$2

declare -a SHARED_LIBS=(
  $(ldd $LIBTORCH_PATH | grep -E -o "/\S+")
  "$LIBTORCH_PATH"
)

for shared_lib in ${SHARED_LIBS[@]}; do
  echo "Processing $shared_lib"
  cuobjdump -res-usage $shared_lib 2> /dev/null | \
    python3 process-cuobjdump-output.py --database $DATABASE_NAME
done

echo "Done!"
