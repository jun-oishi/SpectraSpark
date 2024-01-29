#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>

#include "PlanarRmc.hpp"

namespace py = pybind11;

PYBIND11_MODULE(rmc, m) {
  m.doc() = "pybind11 rmc plugin";  // optional module docstring

  py::class_<PlanarRmc::TrianglePlanarSimulator>(m, "TrianglePlanarSimulator")
      .def(py::init<>())
      .def("init", &PlanarRmc::TrianglePlanarSimulator::init)
      .def("load_exp_data", &PlanarRmc::TrianglePlanarSimulator::load_exp_data)
      .def("run", &PlanarRmc::TrianglePlanarSimulator::run)
      .def("save", &PlanarRmc::TrianglePlanarSimulator::save);
}