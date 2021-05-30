#!/bin/bash

# This script is executed each time the container is started to create a user

if [ -z $CONTAINER_UID ] || [ -z $CONTAINER_UNAME ]; then
  echo "Please set the \"CONTAINER_UID\" and \"CONTAINER_UNAME\" environment variables."
  exit 1
fi

# Create the user if they do not exist
if ! id -u ${CONTAINER_UNAME} &> /dev/null; then
  # NOTE: The home directory is automatically created because we mount stuff inside it
  useradd --shell /bin/bash -u ${CONTAINER_UID} ${CONTAINER_UNAME} && \
    adduser ${CONTAINER_UNAME} sudo && \
    echo "${CONTAINER_UNAME}:${CONTAINER_UNAME}" | chpasswd

  export HOME=/home/${CONTAINER_UNAME}
  echo "cd /home/${CONTAINER_UNAME}" >> /home/${CONTAINER_UNAME}/.bashrc
  echo "alias ls=\"ls --color\"" >> /home/${CONTAINER_UNAME}/.bashrc
  chown ${CONTAINER_UNAME}:${CONTAINER_UNAME} /home/${CONTAINER_UNAME}
  chown ${CONTAINER_UNAME}:${CONTAINER_UNAME} /home/${CONTAINER_UNAME}/.bashrc
fi

exec /usr/local/bin/gosu ${CONTAINER_UNAME} "$@"
