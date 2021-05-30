#! /bin/bash

CHECKSUM_FILE="checksums"
declare -a FILES=(
  "bmm/model.pth"
  "conv2d/model.pth"
  "kernels.sqlite"
  "linear/model.pth"
  "lstm/model.pth"
)

function generate() {
  rm -f $CHECKSUM_FILE
  for file in "${FILES[@]}"; do
    shasum $file >> $CHECKSUM_FILE
  done
  echo "Done! Checksum file has been generated."
}

function validate() {
  shasum -c $CHECKSUM_FILE
}

function usage() {
  echo "Usage: $0 [-g | --generate]"
  echo ""
  echo "This utility checks that Habitat's data files have the correct contents."
  echo ""
  echo "Use the -g or --generate options to generate the checksum file."
  exit 1
}

if [ -z "$1" ]; then
  validate
elif [ "$1" = "-g" ] || [ "$1" = "--generate" ]; then
  generate
else
  usage "$@"
fi
