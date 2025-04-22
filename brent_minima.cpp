// #include <pybind11/pybind11.h>
// #include <pybind11/numpy.h>
// #include <cmath>
// #include "brent.hpp"

// namespace py = pybind11;

// py::array_t<double> solve_rho(py::function F_func,
//                               py::array_t<double> mbar2,
//                               py::array_t<double> sbar2,
//                               py::array_t<double> nbar2,
//                               py::array_t<double> rho_bar) {
//     auto m_arr = mbar2.unchecked<1>();
//     auto s_arr = sbar2.unchecked<1>();
//     auto n_arr = nbar2.unchecked<1>();
//     auto rho_bar_arr = rho_bar.unchecked<1>();

//     size_t N = m_arr.shape(0);
//     auto result = py::array_t<double>(N);
//     auto result_mut = result.mutable_unchecked<1>();

//     for (size_t i = 0; i < N; ++i) {
//         double mbar2 = m_arr(i);
//         double sbar2 = s_arr(i);
//         double nbar2 = n_arr(i);
//         double rhobar = rho_bar_arr(i);

//         // Capture all variables into the lambda
//         auto F_fixed_idx = [&](long double rho) -> long double {
//             py::object val = F_func(rho, rhobar, mbar2, nbar2, sbar2, i);
//             return val.cast<long double>();
//         };

//         auto res = boost::math::tools::brent_find_minima(F_fixed_idx, 1e-15L, 2.0L, std::numeric_limits<long double>::digits);
//         result_mut(i) = static_cast<double>(res.first);
//     }

//     return result;
// }

// PYBIND11_MODULE(brent_minima, m) {
//     m.doc() = "Batch Brent solver for rho with Python-defined F";
//     m.def("solve_rho", &solve_rho,
//           py::arg("F_func"),
//           py::arg("mbar2"),
//           py::arg("sbar2"),
//           py::arg("nbar2"),
//           py::arg("rho_bar"),
//           "Solve rho for each index using Brent's method, calling Python-defined F.");
// }

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include "brent.hpp"

namespace py = pybind11;

py::array_t<double> solve_rho(py::function F_func,
                              py::array_t<double> mbar2,
                              py::array_t<double> sbar2,
                              py::array_t<double> nbar2,
                              py::array_t<double> rho_bar,
                              long double lower_bound,
                              long double upper_bound) {
    auto m_arr = mbar2.unchecked<1>();
    auto s_arr = sbar2.unchecked<1>();
    auto n_arr = nbar2.unchecked<1>();
    auto rho_bar_arr = rho_bar.unchecked<1>();

    size_t N = m_arr.shape(0);
    auto result = py::array_t<double>(N);
    auto result_mut = result.mutable_unchecked<1>();

    for (size_t i = 0; i < N; ++i) {
        double mbar2 = m_arr(i);
        double sbar2 = s_arr(i);
        double nbar2 = n_arr(i);
        double rhobar = rho_bar_arr(i);

        auto F_fixed_idx = [&](long double rho) -> long double {
            py::object val = F_func(rho, rhobar, mbar2, nbar2, sbar2, i);
            return val.cast<long double>();
        };

        auto res = boost::math::tools::brent_find_minima(
            F_fixed_idx, 
            lower_bound, 
            upper_bound, 
            std::numeric_limits<long double>::digits
        );
        result_mut(i) = static_cast<double>(res.first);
    }

    return result;
}

PYBIND11_MODULE(brent_minima, m) {
    m.doc() = "Batch Brent solver for rho with Python-defined F";
    m.def("solve_rho", &solve_rho,
          py::arg("F_func"),
          py::arg("mbar2"),
          py::arg("sbar2"),
          py::arg("nbar2"),
          py::arg("rho_bar"),
          py::arg("lower_bound"),
          py::arg("upper_bound"),
          "Solve rho for each index using Brent's method, calling Python-defined F. Customizable bounds.");
}
