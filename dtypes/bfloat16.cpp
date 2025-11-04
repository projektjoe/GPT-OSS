#include <pybind11/pybind11.h>
#include <pybind11/operators.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "bfloat16.hpp"
#include <sstream>
#include <iomanip>
#include <vector>
#include <stdexcept>

namespace py = pybind11;
using bf = bfloat16;

// Small helper: pack/unpack array conversions (operate on raw bf16 bits stored in uint16 arrays)
static py::array_t<uint16_t> from_float32_array(py::array_t<float, py::array::c_style | py::array::forcecast> arr) {
    py::buffer_info info = arr.request();
    auto result = py::array_t<uint16_t>(info.size);
    py::buffer_info rinfo = result.request();
    float *src = static_cast<float*>(info.ptr);
    uint16_t *dst = static_cast<uint16_t*>(rinfo.ptr);
    for (ssize_t i = 0; i < info.size; ++i) {
        dst[i] = bf::pack(src[i]);
    }
    result.resize(info.shape);
    return result;
}

static py::array_t<float> to_float32_array(py::array_t<uint16_t, py::array::c_style | py::array::forcecast> arr) {
    py::buffer_info info = arr.request();
    auto result = py::array_t<float>(info.size);
    py::buffer_info rinfo = result.request();
    uint16_t *src = static_cast<uint16_t*>(info.ptr);
    float *dst = static_cast<float*>(rinfo.ptr);
    for (ssize_t i = 0; i < info.size; ++i) {
        dst[i] = bf::unpack(src[i]);
    }
    result.resize(info.shape);
    return result;
}

// Matrix container owning a contiguous row-major vector of bf16 values
struct Array {
    std::vector<bf> data;
    std::vector<size_t> shape; 

    size_t rows = 0;
    size_t cols = 0;

    Array() = default;
    Array(const std::vector<size_t>& s) : shape(s) {
        size_t total = 1;
        for (auto d : s) total *= d;
        data.resize(total);
    }
    Array(const std::vector<std::vector<float>>& values) {
        size_t rows = values.size();
        size_t cols = (rows ? values[0].size() : 0);
        if (rows == 0) {
            // Interpret [] (empty outer list) as a 1-D empty array, shape = {0}
            shape = {0};
            data.clear();
        } else {
            shape = {rows, cols};
            data.reserve(rows * cols);
            for (const auto& row : values) {
                if (row.size() != cols)
                    throw std::runtime_error("ragged list provided to Array constructor");
                for (float v : row) data.emplace_back(bf(v));
            }
        }
    }


    size_t ndim() const { return shape.size(); }
    size_t size() const {
        size_t total = 1;
        for (auto d : shape) total *= d;
        return total;
    }
    size_t offset(const std::vector<size_t>& idx) const {
        if (idx.size() != shape.size())
            throw std::runtime_error("Index dimensionality mismatch");
        size_t off = 0, stride = 1;
        for (ssize_t i = shape.size() - 1; i >= 0; --i) {
            if (idx[i] >= shape[i]) throw std::out_of_range("Index out of bounds");
            off += idx[i] * stride;
            stride *= shape[i];
        }
        return off;
    }


    // index
    bf& operator()(size_t i, size_t j) { return data[i * cols + j]; }
    const bf& operator()(size_t i, size_t j) const { return data[i * cols + j]; }
    bf& at(const std::vector<size_t>& idx) { return data[offset(idx)]; }
    const bf& at(const std::vector<size_t>& idx) const { return data[offset(idx)]; }

    // transpose (returns a new Array)
    Array transpose(const std::vector<size_t>& axes = {}) const {
        if (shape.empty()) return *this;
    
        // default: reverse all axes
        std::vector<size_t> perm = axes.empty() ? 
            std::vector<size_t>(shape.size()) : axes;
        if (axes.empty()) {
            std::iota(perm.begin(), perm.end(), 0);
            std::reverse(perm.begin(), perm.end());
        }
    
        if (perm.size() != shape.size())
            throw std::runtime_error("transpose: axis count mismatch");
    
        // new shape
        std::vector<size_t> new_shape(shape.size());
        for (size_t i = 0; i < shape.size(); ++i)
            new_shape[i] = shape[perm[i]];
    
        // allocate
        Array result(new_shape);
    
        // compute strides for old and new
        std::vector<size_t> old_strides(shape.size());
        std::vector<size_t> new_strides(new_shape.size());
        {
            size_t stride = 1;
            for (ssize_t i = shape.size()-1; i >= 0; --i) {
                old_strides[i] = stride;
                stride *= shape[i];
            }
            stride = 1;
            for (ssize_t i = new_shape.size()-1; i >= 0; --i) {
                new_strides[i] = stride;
                stride *= new_shape[i];
            }
        }
    
        // permute
        for (size_t idx = 0; idx < data.size(); ++idx) {
            // decode old index
            size_t tmp = idx;
            std::vector<size_t> old_index(shape.size());
            for (size_t i = 0; i < shape.size(); ++i) {
                old_index[i] = tmp / old_strides[i];
                tmp %= old_strides[i];
            }
            // apply permutation
            std::vector<size_t> new_index(new_shape.size());
            for (size_t i = 0; i < shape.size(); ++i)
                new_index[i] = old_index[perm[i]];
            // compute new offset
            size_t new_off = 0;
            for (size_t i = 0; i < new_shape.size(); ++i)
                new_off += new_index[i] * new_strides[i];
    
            result.data[new_off] = data[idx];
        }
        return result;
    }

    // return a numpy float32 array (row-major)
    py::array_t<float> to_numpy() const {
        std::vector<ssize_t> np_shape(shape.begin(), shape.end());
        py::array_t<float> arr(np_shape);
        py::buffer_info info = arr.request();
        float* dst = static_cast<float*>(info.ptr);
        for (size_t i = 0; i < data.size(); ++i) {
            dst[i] = static_cast<float>(data[i]);
        }
        return arr;
    }

    py::object to_list() const {
        std::function<py::object(size_t, size_t, const std::vector<size_t>&)> rec;
        rec = [&](size_t dim, size_t offset, const std::vector<size_t>& shape) -> py::object {
            if (dim == shape.size() - 1) {
                py::list lst;
                for (size_t i = 0; i < shape[dim]; ++i)
                    lst.append(static_cast<float>(data[offset + i]));
                return lst;
            } else {
                py::list outer;
                size_t stride = 1;
                for (size_t j = dim+1; j < shape.size(); ++j) stride *= shape[j];
                for (size_t i = 0; i < shape[dim]; ++i)
                    outer.append(rec(dim+1, offset + i*stride, shape));
                return outer;
            }
        };
        if (shape.empty()) return py::list();
        return rec(0, 0, shape);
    }

    Array add(const Array& other) const {
        if (shape != other.shape)
            throw std::runtime_error("Array shapes must match for addition");
    
        Array result(shape);
        for (size_t i = 0; i < data.size(); ++i) {
            float a = static_cast<float>(data[i]);
            float b = static_cast<float>(other.data[i]);
            result.data[i] = bf(a + b);
        }
        return result;
    }
    
    Array matmul(const Array& other) const {
        // ---------- Vector × Vector ----------
        if (ndim() == 1 && other.ndim() == 1) {
            size_t n = shape[0];
            if (other.shape[0] != n) 
                throw std::runtime_error("vector dimensions not aligned for dot product");
            float acc = 0.0f;
            for (size_t i = 0; i < n; ++i) {
                float a = static_cast<float>(bf(static_cast<float>(data[i])));
                float b = static_cast<float>(bf(static_cast<float>(other.data[i])));
                acc = static_cast<float>(acc + a * b);
            }
            Array result(std::vector<size_t>{}); 
            result.data.push_back(bf(acc)); // downcast at the end
            return result;
        }

        // ---------- Matrix × Vector ----------
        if (ndim() == 2 && other.ndim() == 1) {
            size_t m = shape[0], n = shape[1];
            if (other.shape[0] != n) 
                throw std::runtime_error("matrix × vector dimension mismatch");
            Array result(std::vector<size_t>{m});
            for (size_t i = 0; i < m; ++i) {
                float acc = 0.0f;
                for (size_t j = 0; j < n; ++j) {
                    float a = static_cast<float>(bf(static_cast<float>(data[i * n + j])));
                    float b = static_cast<float>(bf(static_cast<float>(other.data[j])));
                    acc = static_cast<float>(acc + a * b);
                }
                result.data[i] = bf(acc);
            }
            return result;
        }

        // ---------- Vector × Matrix ----------
        if (ndim() == 1 && other.ndim() == 2) {
            size_t n = shape[0];
            size_t k = other.shape[0], p = other.shape[1];
            if (n != k)
                throw std::runtime_error("vector × matrix dimension mismatch");
            Array result(std::vector<size_t>{p});
            for (size_t j = 0; j < p; ++j) {
                float acc = 0.0f;
                for (size_t i = 0; i < n; ++i) {
                    float a = static_cast<float>(bf(static_cast<float>(data[i])));
                    float b = static_cast<float>(bf(static_cast<float>(other.data[i * p + j])));
                    acc = static_cast<float>(acc + a * b);
                }
                result.data[j] = bf(acc);
            }
            return result;
        }

        // ---------- Matrix × Matrix ----------
        if (ndim() == 2 && other.ndim() == 2) {
            size_t m = shape[0], k1 = shape[1];
            size_t k2 = other.shape[0], n = other.shape[1];
            if (k1 != k2) throw std::runtime_error("matrix dimensions not aligned for matmul");

            Array result(std::vector<size_t>{m, n}); 
            for (size_t i = 0; i < m; ++i) {
                for (size_t j = 0; j < n; ++j) {
                    float acc = 0.0f;
                    for (size_t k = 0; k < k1; ++k) {
                        float a = static_cast<float>(bf(static_cast<float>(data[i * k1 + k])));
                        float b = static_cast<float>(bf(static_cast<float>(other.data[k * n + j])));
                        acc = static_cast<float>(acc + a * b);
                    }
                    result.data[i * n + j] = bf(acc);
                }
            }
            return result;
        }

        throw std::runtime_error("matmul only supports 1D or 2D operands");
    }
    Array multiply_scalar(float scalar) const {
        Array result(shape);
        for (size_t i = 0; i < data.size(); ++i) {
            float a = static_cast<float>(data[i]);
            result.data[i] = bf(a * scalar);
        }
        return result;
    }
    Array linear(const Array& other, const Array& bias) const {
        // ---------- Matrix × Vector + Bias ----------
        // FUSED KERNEL FOR MATMUL AND ADD, TO MIMIC PYTORCH'S BEHAVIOR
        if (ndim() == 2 && other.ndim() == 1 && bias.ndim() == 1) {
            size_t m = shape[0], n = shape[1];
            if (other.shape[0] != n) 
                throw std::runtime_error("linear: matrix × vector dimension mismatch");
            if (bias.shape[0] != m)
                throw std::runtime_error("linear: bias length mismatch");

            Array result(std::vector<size_t>{m});
            for (size_t i = 0; i < m; ++i) {
                float acc = 0.0f;
                for (size_t j = 0; j < n; ++j) {
                    float a = static_cast<float>(data[i * n + j]);
                    float b = static_cast<float>(other.data[j]);
                    acc += static_cast<float>(a * b);
                }
                // add bias in float32
                acc += static_cast<float>(bias.data[i]);
                result.data[i] = bf(acc); // downcast at the end
            }
            return result;
        }
        throw std::runtime_error("linear only supports (2D matrix × 1D vector + 1D bias)");
    }
    Array concat(const Array& other, size_t axis = 0) const {
        // disallow concatenating scalars (0-d arrays)
        if (shape.empty() || other.shape.empty())
            throw std::runtime_error("cannot concatenate 0-d (scalar) arrays");

        if (shape.size() != other.shape.size())
            throw std::runtime_error("arrays must have the same number of dimensions");

        if (axis >= shape.size())
            throw std::runtime_error("concat: axis out of range");

        // check all dimensions except along concat axis
        for (size_t i = 0; i < shape.size(); ++i) {
            if (i == axis) continue;
            if (shape[i] != other.shape[i])
                throw std::runtime_error("shapes not aligned for concatenation");
        }

        // new shape
        std::vector<size_t> new_shape = shape;
        new_shape[axis] = shape[axis] + other.shape[axis];

        Array result(new_shape);

        // If both are empty just return the result (zero-length along axis)
        if (data.empty() && other.data.empty())
            return result;

        // compute number of elements in left part and right part
        size_t left_count = data.size();
        size_t right_count = other.data.size();

        // elementwise copy (safe for non-trivial types)
        if (left_count)
            std::copy_n(data.data(), left_count, result.data.data());
        if (right_count)
            std::copy_n(other.data.data(), right_count, result.data.data() + left_count);

        return result;
    }


};

std::string preview_recursive(const Array& M, 
                              const std::vector<size_t>& shape,
                              size_t depth,
                              size_t offset,
                              const std::vector<size_t>& strides,
                              size_t limit = 6) {
    if (depth == shape.size() - 1) {
        // Base case: last dimension → print up to "limit" elements
        std::ostringstream oss;
        oss << "[";
        size_t n = shape[depth];
        for (size_t i = 0; i < std::min(n, limit); ++i) {
            if (i > 0) oss << ", ";
            oss << std::setprecision(8)
                << static_cast<float>(M.data[offset + i * strides[depth]]);

        }
        if (n > limit) oss << ", ...";
        oss << "]";
        return oss.str();
    } else {
        // Recursive case
        std::ostringstream oss;
        oss << "[";
        size_t n = shape[depth];
        for (size_t i = 0; i < std::min(n, limit); ++i) {
            if (i > 0) oss << ", ";
            oss << preview_recursive(M, shape, depth+1, 
                                     offset + i * strides[depth], strides, limit);
        }
        if (n > limit) oss << ", ...";
        oss << "]";
        return oss.str();
    }
}

PYBIND11_MODULE(bfloat16, m) {
    
    m.doc() = "bfloat16 small module: scalar bfloat16 + Array with debug repr and matmul";

    // expose pack/unpack helpers for bulk conversion
    m.def("from_float32_array", &from_float32_array, py::arg("arr"));
    m.def("to_float32_array", &to_float32_array, py::arg("arr"));

    // scalar bfloat16 (small surface)
    py::class_<bf>(m, "bfloat16")
        .def(py::init<>())
        .def(py::init<float>())
        .def(py::init<double>())
        .def("__float__", [](const bf &x){ return static_cast<float>(x); })
        .def("__repr__", [](const bf &x){ char buf[64]; std::snprintf(buf, sizeof(buf), "float32(%.15g)", static_cast<float>(x)); return std::string(buf); })
        ;

    // Array bindings
    py::class_<Array>(m, "Array")
        .def(py::init<>())
        .def(py::init<const std::vector<std::vector<float>>&>(), py::arg("values"))
        .def(py::init([](py::array_t<float, py::array::c_style | py::array::forcecast> arr) {
            py::buffer_info info = arr.request();
            std::vector<size_t> shape(info.shape.begin(), info.shape.end());
            Array M(shape);
            float* src = static_cast<float*>(info.ptr);
            for (ssize_t i = 0; i < info.size; ++i) {
                M.data[i] = bf(src[i]);
            }
            return M;
        }))
        .def(py::init([](py::array_t<uint16_t, py::array::c_style | py::array::forcecast> arr) {
            py::buffer_info info = arr.request();
            std::vector<size_t> shape(info.shape.begin(), info.shape.end());
            Array M(shape);
            uint16_t* src = static_cast<uint16_t*>(info.ptr);
            for (ssize_t i = 0; i < info.size; ++i) {
                M.data[i] = bf::from_bits(src[i]);
            }
            return M;
        }))
        
        .def_property_readonly("rows", [](const Array &M){ return M.rows; })
        .def_property_readonly("cols", [](const Array &M){ return M.cols; })
        .def("__add__", &Array::add)
        .def("__radd__", &Array::add)
        .def("matmul", &Array::matmul, py::arg("other"))
        .def("__mul__", [](const Array& self, float scalar) {
            return self.multiply_scalar(scalar);
        }, py::arg("scalar"))
        .def("__rmul__", [](const Array& self, float scalar) {
            return self.multiply_scalar(scalar);
        }, py::arg("scalar"))
        .def("transpose", &Array::transpose, py::arg("axes") = std::vector<size_t>())
        .def_property_readonly("T", [](const Array &M){ return M.transpose(); })
        .def("to_list", &Array::to_list)
        .def("to_numpy", &Array::to_numpy)
        .def("concat", &Array::concat, py::arg("other"), py::arg("axis") = 0)
        .def("__repr__", [](const Array &M){
            std::ostringstream oss;
            oss << "Array(shape=[";
            for (size_t i = 0; i < M.shape.size(); ++i) {
                oss << M.shape[i];
                if (i + 1 < M.shape.size()) oss << ",";
            }
            oss << "], ndim=" << M.ndim() << ")\n";
        
            if (M.shape.empty()) return oss.str();
        
            // compute strides
            std::vector<size_t> strides(M.shape.size());
            size_t stride = 1;
            for (ssize_t i = M.shape.size()-1; i >= 0; --i) {
                strides[i] = stride;
                stride *= M.shape[i];
            }
        
            oss << preview_recursive(M, M.shape, 0, 0, strides, 6);
            return oss.str();
        })
        .def("__str__", [](const Array &M){
            return py::repr(M.to_list());
        })        
        .def("__matmul__", &Array::matmul, py::arg("other"))
        .def("__getitem__", [](const Array &M, py::object key) -> py::object {
            using namespace pybind11;
            size_t ndim = M.shape.size();
            if (ndim == 0) throw index_error("empty tensor");
        
            // strides (row-major)
            std::vector<size_t> strides(ndim);
            {
                size_t stride = 1;
                for (ssize_t i = (ssize_t)ndim - 1; i >= 0; --i) {
                    strides[(size_t)i] = stride;
                    stride *= M.shape[(size_t)i];
                }
            }
        
            auto normalize_idx = [&](ssize_t idx, size_t dim)->size_t {
                if (idx < 0) idx += static_cast<ssize_t>(M.shape[dim]);
                if (idx < 0 || static_cast<size_t>(idx) >= M.shape[dim])
                    throw index_error("index out of range");
                return static_cast<size_t>(idx);
            };
        
            // ---------- Handle slices ----------
            if (isinstance<slice>(key)) {
                // single slice on axis 0
                slice s = key.cast<slice>();
                size_t start, stop, step, slice_len;
                if (!s.compute(M.shape[0], &start, &stop, &step, &slice_len))
                    throw index_error("invalid slice");
        
                std::vector<size_t> new_shape = M.shape;
                new_shape[0] = slice_len;
                Array out(new_shape);
        
                for (size_t i = 0; i < slice_len; ++i) {
                    size_t src_i = start + i * step;
                    std::memcpy(out.data.data() + i * strides[0],
                                M.data.data() + src_i * strides[0],
                                strides[0] * sizeof(bf));
                }
                return cast(out);
            }
        
            // ---------- Handle tuple of slices/ints ----------
            if (isinstance<tuple>(key)) {
                tuple tup = key.cast<tuple>();
                if ((size_t)tup.size() > ndim)
                    throw index_error("too many indices");
        
                // prepare per-axis slice ranges
                std::vector<size_t> starts(ndim), stops(ndim), steps(ndim), slice_lens(ndim);
                std::vector<bool> is_slice(ndim, false);
        
                for (size_t d = 0; d < tup.size(); ++d) {
                    if (isinstance<slice>(tup[d])) {
                        slice s = tup[d].cast<slice>();
                        size_t start, stop, step, slen;
                        if (!s.compute(M.shape[d], &start, &stop, &step, &slen))
                            throw index_error("invalid slice");
                        starts[d] = start; stops[d] = stop; steps[d] = step; slice_lens[d] = slen;
                        is_slice[d] = true;
                    } else {
                        // assume int index
                        ssize_t id = tup[d].cast<ssize_t>();
                        size_t i = normalize_idx(id, d);
                        starts[d] = i; stops[d] = i+1; steps[d] = 1; slice_lens[d] = 1;
                        is_slice[d] = false;
                    }
                }
                // fill remaining axes as ":" (full slice)
                for (size_t d = tup.size(); d < ndim; ++d) {
                    starts[d] = 0; stops[d] = M.shape[d]; steps[d] = 1; slice_lens[d] = M.shape[d];
                    is_slice[d] = true;
                }
        
                // new shape excludes dimensions reduced to scalars
                std::vector<size_t> new_shape;
                for (size_t d = 0; d < ndim; ++d) {
                    if (is_slice[d]) new_shape.push_back(slice_lens[d]);
                }
        
                // scalar result?
                if (new_shape.empty()) {
                    size_t off = 0;
                    for (size_t d = 0; d < ndim; ++d)
                        off += starts[d] * strides[d];
                    return cast(static_cast<float>(M.data[off]));
                }
        
                Array out(new_shape);
        
                // brute-force nested loops
                std::vector<size_t> idx(ndim, 0);
                size_t total = out.size();
                for (size_t linear = 0; linear < total; ++linear) {
                    // decode linear idx into out index
                    size_t tmp = linear;
                    std::vector<size_t> out_index(new_shape.size());
                    for (ssize_t d = new_shape.size()-1; d >= 0; --d) {
                        out_index[d] = tmp % new_shape[d];
                        tmp /= new_shape[d];
                    }
        
                    // map back to source index
                    size_t src_off = 0;
                    size_t out_axis = 0;
                    for (size_t d = 0; d < ndim; ++d) {
                        size_t coord = is_slice[d] ? (starts[d] + out_index[out_axis++] * steps[d]) : starts[d];
                        src_off += coord * strides[d];
                    }
                    out.data[linear] = M.data[src_off];
                }
        
                return cast(out);
            }
        
            // fall back to int indexing
            try {
                ssize_t idx = key.cast<ssize_t>();
                size_t i = normalize_idx(idx, 0);
                if (ndim == 1) {
                    return cast(static_cast<float>(M.data[i]));
                }
                std::vector<size_t> new_shape(M.shape.begin()+1, M.shape.end());
                Array out(new_shape);
                std::memcpy(out.data.data(), M.data.data() + i * strides[0], strides[0] * sizeof(bf));
                return cast(out);
            } catch (const cast_error &) {
                throw type_error("Invalid index type. Supported: int, slice, or tuple of them");
            }
        })
        .def("__len__", [](const Array &M) {
            if (M.shape.empty()) 
                throw std::runtime_error("len() of empty tensor");
            return static_cast<py::ssize_t>(M.shape[0]);
        })
        .def("__setitem__", [](Array &M, py::object key, py::object value) {
            using namespace pybind11;
            size_t ndim = M.shape.size();
            if (ndim == 0) throw index_error("cannot assign to empty tensor");

            // strides (row-major)
            std::vector<size_t> strides(ndim);
            {
                size_t stride = 1;
                for (ssize_t i = (ssize_t)ndim - 1; i >= 0; --i) {
                    strides[(size_t)i] = stride;
                    stride *= M.shape[(size_t)i];
                }
            }

            auto normalize_idx = [&](ssize_t idx, size_t dim)->size_t {
                if (idx < 0) idx += static_cast<ssize_t>(M.shape[dim]);
                if (idx < 0 || static_cast<size_t>(idx) >= M.shape[dim])
                    throw index_error("index out of range");
                return static_cast<size_t>(idx);
            };

            // ---------- Handle simple int indexing ----------
            if (py::isinstance<py::int_>(key)) {
                ssize_t id = key.cast<ssize_t>();
                size_t i = normalize_idx(id, 0);
                if (ndim == 1) {
                    M.data[i] = bf(value.cast<float>());
                } else {
                    Array rhs = value.cast<Array>();
                    if (rhs.shape != std::vector<size_t>(M.shape.begin()+1, M.shape.end()))
                        throw std::runtime_error("shape mismatch in assignment");
                    std::memcpy(M.data.data() + i * strides[0],
                                rhs.data.data(),
                                strides[0] * sizeof(bf));
                }
                return;
            }

            // ---------- Handle slice on first axis ----------
            if (py::isinstance<py::slice>(key)) {
                slice s = key.cast<slice>();
                size_t start, stop, step, slice_len;
                if (!s.compute(M.shape[0], &start, &stop, &step, &slice_len))
                    throw index_error("invalid slice");

                Array rhs = value.cast<Array>();
                if (rhs.shape.size() != ndim || rhs.shape[0] != slice_len)
                    throw std::runtime_error("shape mismatch in slice assignment");

                for (size_t i = 0; i < slice_len; ++i) {
                    size_t dst_i = start + i * step;
                    std::memcpy(M.data.data() + dst_i * strides[0],
                                rhs.data.data() + i * strides[0],
                                strides[0] * sizeof(bf));
                }
                return;
            }

            // ---------- Handle tuple indexing ----------
            if (py::isinstance<py::tuple>(key)) {
                tuple tup = key.cast<tuple>();
                if ((size_t)tup.size() > ndim)
                    throw index_error("too many indices");

                std::vector<size_t> starts(ndim), steps(ndim), slice_lens(ndim);
                std::vector<bool> is_slice(ndim, false);

                for (size_t d = 0; d < tup.size(); ++d) {
                    if (isinstance<slice>(tup[d])) {
                        size_t start, stop, step, slen;
                        slice s = tup[d].cast<slice>();
                        if (!s.compute(M.shape[d], &start, &stop, &step, &slen))
                            throw index_error("invalid slice");
                        starts[d] = start; steps[d] = step; slice_lens[d] = slen;
                        is_slice[d] = true;
                    } else {
                        ssize_t id = tup[d].cast<ssize_t>();
                        size_t i = normalize_idx(id, d);
                        starts[d] = i; steps[d] = 1; slice_lens[d] = 1;
                        is_slice[d] = false;
                    }
                }
                for (size_t d = tup.size(); d < ndim; ++d) {
                    starts[d] = 0; steps[d] = 1; slice_lens[d] = M.shape[d];
                    is_slice[d] = true;
                }

                // scalar assignment
                bool all_scalar = true;
                for (bool s : is_slice) if (s) { all_scalar = false; break; }
                if (all_scalar) {
                    size_t off = 0;
                    for (size_t d = 0; d < ndim; ++d) off += starts[d] * strides[d];
                    M.data[off] = bf(value.cast<float>());
                    return;
                }

                // subarray assignment
                Array rhs = value.cast<Array>();
                std::vector<size_t> expected_shape;
                for (size_t d = 0; d < ndim; ++d)
                    if (is_slice[d]) expected_shape.push_back(slice_lens[d]);
                if (rhs.shape != expected_shape)
                    throw std::runtime_error("shape mismatch in slice assignment");

                size_t total = rhs.size();
                for (size_t linear = 0; linear < total; ++linear) {
                    size_t tmp = linear;
                    std::vector<size_t> rhs_index(rhs.shape.size());
                    for (ssize_t d = rhs.shape.size()-1; d >= 0; --d) {
                        rhs_index[d] = tmp % rhs.shape[d];
                        tmp /= rhs.shape[d];
                    }
                    size_t src_off = 0;
                    size_t rhs_axis = 0;
                    for (size_t d = 0; d < ndim; ++d) {
                        size_t coord = is_slice[d] ?
                            (starts[d] + rhs_index[rhs_axis++] * steps[d]) : starts[d];
                        src_off += coord * strides[d];
                    }
                    M.data[src_off] = rhs.data[linear];
                }
                return;
            }

            throw type_error("Invalid index type for assignment");
        })
        .def_property_readonly("shape", [](const Array &M){ return M.shape; })
        .def_property_readonly("ndim", [](const Array &M){ return M.ndim(); })
        .def_property_readonly("size", [](const Array &M){ return M.size(); })

        
        ;

        // module level function
        m.def("linear", [](const Array &x, const Array &W, const Array &bias) {
            return W.linear(x, bias);
        }, py::arg("x"), py::arg("W"), py::arg("bias"));

}
