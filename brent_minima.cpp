#include <pybind11/pybind11.h>
#include "brent.hpp"


namespace py = pybind11;

double brent_find_minima(py::object py_func, double min, double max, int bits = std::numeric_limits<double>::digits) {
    auto cpp_func = [&py_func](double x) -> double {
        return py_func(x).cast<double>();
    };

    auto result = boost::math::tools::brent_find_minima(cpp_func, min, max, bits);
    return result.first; 
}

PYBIND11_MODULE(brent_minima, m) {
    m.doc() = "Wrapper for boost::math::tools::brent_find_minima";
    m.def("brent_find_minima", &brent_find_minima,
          py::arg("func"), py::arg("min"), py::arg("max"), py::arg("bits") = std::numeric_limits<double>::digits,
          "Find minimum using Brent method.");
}

