# Habitat: A Runtime-Based Computational Performance Predictor for Deep Neural Network Training

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4885489.svg)](https://doi.org/10.5281/zenodo.4885489)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4876277.svg)](https://doi.org/10.5281/zenodo.4876277)

Habitat is a tool that predicts a deep neural network's training iteration
execution time on a given GPU. It currently supports PyTorch. To learn more
about how Habitat works, please see our [research
paper](https://arxiv.org/abs/2102.00527).


## Running From Source

Currently, the only way to run Habitat is to build it from source. You should
use the Docker image provided in this repository to make sure that you can
compile the code.

1. Download the [Habitat pre-trained
   models](https://doi.org/10.5281/zenodo.4876277).
2. Run `extract-models.sh` under `analyzer` to extract and install the
   pre-trained models.
3. Run `setup.sh` under `docker/` to build the Habitat container image.
4. Run `start.sh` to start a new container. By default, your home directory
   will be mounted inside the container under `~/home`.
5. Once inside the container, run `install-dev.sh` under `analyzer/` to build
   and install the Habitat package.
6. In your scripts, `import habitat` to get access to Habitat. See
   `experiments/run_experiment.py` for an example showing how to use Habitat.

**Note:** Habitat needs access to your GPU's performance counters, which
requires special permissions if you are running with a recent driver (418.43 or
later). If you encounter a `CUPTI_ERROR_INSUFFICIENT_PRIVILEGES` error when
running Habitat, please follow the instructions
[here](https://developer.nvidia.com/ERR_NVGPUCTRPERM)
and in [issue #5](https://github.com/geoffxy/habitat/issues/5).


## License

The code in this repository is licensed under the Apache 2.0 license (see
`LICENSE` and `NOTICE`), with the exception of the files mentioned below.

This software contains source code provided by NVIDIA Corporation. These files
are:

- The code under `cpp/external/cupti_profilerhost_util/` (CUPTI sample code)
- `cpp/src/cuda/cuda_occupancy.h`

The code mentioned above is licensed under the [NVIDIA Software Development
Kit End User License Agreement](https://docs.nvidia.com/cuda/eula/index.html).

We include the implementations of several deep neural networks under
`experiments/` for our evaluation. These implementations are copyrighted by
their original authors and carry their original licenses. Please see the
corresponding `README` files and license files inside the subdirectories for
more information.


## Research Paper

Habitat began as a research project in the [EcoSystem
Group](https://www.cs.toronto.edu/ecosystem) at the [University of
Toronto](https://cs.toronto.edu). The accompanying research paper will appear
in the proceedings of [USENIX
ATC'21](https://www.usenix.org/conference/atc21/presentation/yu). If you are
interested, you can read a preprint of the paper
[here](https://arxiv.org/abs/2102.00527).

If you use Habitat in your research, please consider citing our paper:

```bibtex
@inproceedings{habitat-yu21,
  author = {Yu, Geoffrey X. and Gao, Yubo and Golikov, Pavel and Pekhimenko,
    Gennady},
  title = {{Habitat: A Runtime-Based Computational Performance Predictor for
    Deep Neural Network Training}},
  booktitle = {{Proceedings of the 2021 USENIX Annual Technical Conference
    (USENIX ATC'21)}},
  year = {2021},
}
```
