#! /bin/bash

SO_NAME="habitat_cuda.cpython-36m-x86_64-linux-gnu.so"
PACKAGE_NAME="habitat-predict"

# Operate out of the script directory
SCRIPT_PATH=$(cd $(dirname $0) && pwd -P)
cd $SCRIPT_PATH

# Abort if an error occurs
set -e

function pushd() {
  command pushd "$@" > /dev/null
}

function popd() {
  command popd "$@" > /dev/null
}

function compile_habitat_cuda() {
  echo "Compiling the Habitat C++ extension..."
  pushd ../cpp
  mkdir -p build
  pushd build

  cmake -DCMAKE_BUILD_TYPE=Release ..
  make -j 8 habitat_cuda

  if [ ! -f $SO_NAME ]; then
    echo "ERROR: Could not find $SO_NAME after compilation. Please double "
    echo "check that compilation completed successfully."
    exit 1
  fi

  popd
  popd
  echo ""
}

function symlink_habitat_cuda() {
  echo "Adding a symbolic link to the Habitat C++ extension..."
  if [ ! -h habitat/$SO_NAME ]; then
    ln -s ../../cpp/build/$SO_NAME habitat
  fi
  echo ""
}

function install_habitat() {
  echo "Install an editable version of the Habitat package..."
  pip3 install --editable .
  echo ""
}

function uninstall_habitat() {
  pip3 uninstall $PACKAGE_NAME
}

function check_prereqs() {
  if [ -z $(which cmake) ]; then
    echo "Please ensure cmake 3.17+ is installed."
    exit 1
  fi
  if [ -z $(which make) ]; then
    echo "Please ensure make is installed."
  fi
  if [ -z $(which pip3) ]; then
    echo "Please ensure pip3 is installed."
    exit 1
  fi
}

function main() {
  if [ "$1" = "--uninstall" ]; then
    uninstall_habitat
  else
    check_prereqs
    compile_habitat_cuda
    symlink_habitat_cuda
    install_habitat
  fi
}

main $@
