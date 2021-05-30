Habitat Docker Image
=====================
The Dockerfile in this directory specifies a Docker image that is used as our
development and test environment. Run `setup.sh` to build the Docker image.

To start a container, run the `start.sh` script.  The container is set up so
that your current account is duplicated inside the container (with the same
user ID and username). This prevents permission issues when accessing files in
mounted volumes inside and outside the container. The user inside the container
will have `sudo` permissions; the account's password will be set to your
username.

Note that the `start.sh` script will restart any containers that are stopped
but have not been removed. If you make any changes to the Docker image and/or
want to start a new container, you need to remove the existing container with
`docker rm` before running `start.sh` again.
