#pragma once

#include <pybind11/pybind11.h>

namespace habitat {
namespace frontend {

void bindModels(pybind11::module& m);

}
}
