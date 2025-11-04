#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <dnnl.hpp>
#include <vector>
#include <stdexcept>
#include <cstring>
#include <sstream>

namespace py = pybind11;
using namespace dnnl;

#ifdef _OPENMP
#include <omp.h>
#endif

inline uint16_t float_to_bf16(float value) {
    uint32_t input;
    std::memcpy(&input, &value, sizeof(float));
    uint32_t rounding_bias = 0x7fff + ((input >> 16) & 1);
    input += rounding_bias;
    return static_cast<uint16_t>(input >> 16);
}

inline float bf16_to_float(uint16_t bf16_value) {
    uint32_t result = static_cast<uint32_t>(bf16_value) << 16;
    float output;
    std::memcpy(&output, &result, sizeof(float));
    return output;
}

py::array_t<float> oneDNN_linear_f32(py::array_t<float> x_f32, 
    py::array_t<float> W_f32, 
    py::array_t<float> b_f32) {
engine eng(engine::kind::cpu, 0);
stream s(eng);

// Get shapes and data
auto x_info = x_f32.request();
auto W_info = W_f32.request();
auto b_info = b_f32.request();

long N = x_info.shape[0];
long IC = x_info.shape[1];
long OC = W_info.shape[0];


float* x_data = static_cast<float*>(x_info.ptr);
float* W_data = static_cast<float*>(W_info.ptr);
float* b_data = static_cast<float*>(b_info.ptr);




py::array_t<float> result({N, OC});
auto res_info = result.request();
float* y_data = static_cast<float*>(res_info.ptr);

// Use F32 throughout
auto src_md = memory::desc({N, IC}, memory::data_type::f32, memory::format_tag::nc);
auto weights_md = memory::desc({OC, IC}, memory::data_type::f32, memory::format_tag::oi);
auto bias_md = memory::desc({OC}, memory::data_type::f32, memory::format_tag::x);
auto dst_md = memory::desc({N, OC}, memory::data_type::f32, memory::format_tag::nc);

// Create primitive descriptor directly (oneDNN 3.x API)
try {
    #if DNNL_VERSION_MAJOR >= 3
    auto ip_pd = inner_product_forward::primitive_desc(
        eng,
        prop_kind::forward_inference,
        src_md,
        weights_md,
        bias_md,
        dst_md);
    #else
    auto ip_desc = inner_product_forward::desc(prop_kind::forward_inference,
         src_md, weights_md, bias_md, dst_md);
    auto ip_pd = inner_product_forward::primitive_desc(ip_desc, eng);
    #endif

// Create memory objects
auto src_mem = memory(src_md, eng, x_data);
auto weights_mem = memory(weights_md, eng, W_data);
auto bias_mem = memory(bias_md, eng, b_data);
auto dst_mem = memory(dst_md, eng, y_data);

// Execute
auto ip_prim = inner_product_forward(ip_pd);
ip_prim.execute(s, {
{DNNL_ARG_SRC, src_mem},
{DNNL_ARG_WEIGHTS, weights_mem},
{DNNL_ARG_BIAS, bias_mem},
{DNNL_ARG_DST, dst_mem}
});

s.wait();
return result;

} catch (const std::exception& e) {
throw std::runtime_error("Failed to create primitive: " + std::string(e.what()));
}


}

PYBIND11_MODULE(linear_layer, m) {
    m.def("linear", &oneDNN_linear_f32, "Linear using oneDNN (f32), x 1D, W 2D, b 1D");
}

//;batching
// #include <pybind11/pybind11.h>
// #include <pybind11/numpy.h>
// #include <dnnl.hpp>
// #include <vector>
// #include <stdexcept>
// #include <cstring>

// namespace py = pybind11;
// using namespace dnnl;

// py::array_t<uint16_t> oneDNN_linear(py::array_t<uint16_t> x_bf16,
//                                     py::array_t<uint16_t> W_bf16,
//                                     py::array_t<uint16_t> b_bf16) {
//     auto x_info = x_bf16.request();
//     auto W_info = W_bf16.request();
//     auto b_info = b_bf16.request();

//     if (x_info.ndim != 2 || W_info.ndim != 2 || b_info.ndim != 1)
//         throw std::runtime_error("Unexpected input shapes.");

//     int batch      = static_cast<int>(x_info.shape[0]);
//     int input_dim  = static_cast<int>(x_info.shape[1]);
//     int output_dim = static_cast<int>(W_info.shape[0]);

//     engine eng(engine::kind::cpu, 0);
//     stream s(eng);

//     memory::dims x_dims = {batch, input_dim};
//     memory::dims W_dims = {output_dim, input_dim};
//     memory::dims b_dims = {output_dim};
//     memory::dims y_dims = {batch, output_dim};

//     // User memory in bf16
//     auto x_user_md = memory::desc(x_dims, memory::data_type::bf16, memory::format_tag::ab);
//     auto W_user_md = memory::desc(W_dims, memory::data_type::bf16, memory::format_tag::ab);
//     auto b_user_md = memory::desc(b_dims, memory::data_type::bf16, memory::format_tag::a);

//     memory x_user_mem(x_user_md, eng, x_info.ptr);
//     memory W_user_mem(W_user_md, eng, W_info.ptr);
//     memory b_user_mem(b_user_md, eng, b_info.ptr);

//     std::vector<uint16_t> out_buf(batch * output_dim);

//     // Helper to copy from a dnnl memory to contiguous bf16 output
//     auto copy_to_out = [&](memory &src_mem) {
//         memory out_user_mem(
//             {y_dims, memory::data_type::bf16, memory::format_tag::ab},
//             eng, out_buf.data());
//         reorder(src_mem, out_user_mem).execute(s, src_mem, out_user_mem);
//         s.wait();
//     };

//     try {
//         // bf16 primitive descriptors
//         auto x_any_md = memory::desc(x_dims, memory::data_type::bf16, memory::format_tag::any);
//         auto W_any_md = memory::desc(W_dims, memory::data_type::bf16, memory::format_tag::any);
//         auto b_any_md = memory::desc(b_dims, memory::data_type::bf16, memory::format_tag::any);
//         auto y_any_md = memory::desc(y_dims, memory::data_type::bf16, memory::format_tag::any);

//         auto ip_desc = inner_product_forward::desc(
//             prop_kind::forward_inference, x_any_md, W_any_md, b_any_md, y_any_md);
//         auto ip_pd = inner_product_forward::primitive_desc(ip_desc, eng);

//         memory src_mem = x_user_mem;
//         if (ip_pd.src_desc() != x_user_mem.get_desc()) {
//             src_mem = memory(ip_pd.src_desc(), eng);
//             reorder(x_user_mem, src_mem).execute(s, x_user_mem, src_mem);
//         }

//         memory weights_mem = W_user_mem;
//         if (ip_pd.weights_desc() != W_user_mem.get_desc()) {
//             weights_mem = memory(ip_pd.weights_desc(), eng);
//             reorder(W_user_mem, weights_mem).execute(s, W_user_mem, weights_mem);
//         }

//         memory bias_mem = b_user_mem;
//         if (ip_pd.bias_desc() != b_user_mem.get_desc()) {
//             bias_mem = memory(ip_pd.bias_desc(), eng);
//             reorder(b_user_mem, bias_mem).execute(s, b_user_mem, bias_mem);
//         }

//         memory dst_mem(ip_pd.dst_desc(), eng);

//         auto ip_prim = inner_product_forward(ip_pd);
//         ip_prim.execute(s, {
//             {DNNL_ARG_SRC, src_mem},
//             {DNNL_ARG_WEIGHTS, weights_mem},
//             {DNNL_ARG_BIAS, bias_mem},
//             {DNNL_ARG_DST, dst_mem}
//         });
//         s.wait();

//         copy_to_out(dst_mem);
//     }
//     catch (dnnl::error &) {
//         // fallback: run in f32 if bf16 inner_product not supported
//         auto x_f32_user_md = memory::desc(x_dims, memory::data_type::f32, memory::format_tag::ab);
//         auto W_f32_user_md = memory::desc(W_dims, memory::data_type::f32, memory::format_tag::ab);
//         auto b_f32_user_md = memory::desc(b_dims, memory::data_type::f32, memory::format_tag::a);

//         memory x_f32_user_mem(x_f32_user_md, eng);
//         memory W_f32_user_mem(W_f32_user_md, eng);
//         memory b_f32_user_mem(b_f32_user_md, eng);

//         reorder(x_user_mem, x_f32_user_mem).execute(s, x_user_mem, x_f32_user_mem);
//         reorder(W_user_mem, W_f32_user_mem).execute(s, W_user_mem, W_f32_user_mem);
//         reorder(b_user_mem, b_f32_user_mem).execute(s, b_user_mem, b_f32_user_mem);
//         s.wait();

//         auto x_any_md_f32 = memory::desc(x_dims, memory::data_type::f32, memory::format_tag::any);
//         auto W_any_md_f32 = memory::desc(W_dims, memory::data_type::f32, memory::format_tag::any);
//         auto b_any_md_f32 = memory::desc(b_dims, memory::data_type::f32, memory::format_tag::any);
//         auto y_any_md_f32 = memory::desc(y_dims, memory::data_type::f32, memory::format_tag::any);

//         auto ip_desc_f32 = inner_product_forward::desc(
//             prop_kind::forward_inference, x_any_md_f32, W_any_md_f32, b_any_md_f32, y_any_md_f32);
//         auto ip_pd_f32 = inner_product_forward::primitive_desc(ip_desc_f32, eng);

//         memory src_f32_mem = x_f32_user_mem;
//         if (ip_pd_f32.src_desc() != x_f32_user_mem.get_desc()) {
//             src_f32_mem = memory(ip_pd_f32.src_desc(), eng);
//             reorder(x_f32_user_mem, src_f32_mem).execute(s, x_f32_user_mem, src_f32_mem);
//         }

//         memory weights_f32_mem = W_f32_user_mem;
//         if (ip_pd_f32.weights_desc() != W_f32_user_mem.get_desc()) {
//             weights_f32_mem = memory(ip_pd_f32.weights_desc(), eng);
//             reorder(W_f32_user_mem, weights_f32_mem).execute(s, W_f32_user_mem, weights_f32_mem);
//         }

//         memory bias_f32_mem = b_f32_user_mem;
//         if (ip_pd_f32.bias_desc() != b_f32_user_mem.get_desc()) {
//             bias_f32_mem = memory(ip_pd_f32.bias_desc(), eng);
//             reorder(b_f32_user_mem, bias_f32_mem).execute(s, b_f32_user_mem, bias_f32_mem);
//         }

//         memory dst_f32_mem(ip_pd_f32.dst_desc(), eng);

//         auto ip_prim_f32 = inner_product_forward(ip_pd_f32);
//         ip_prim_f32.execute(s, {
//             {DNNL_ARG_SRC, src_f32_mem},
//             {DNNL_ARG_WEIGHTS, weights_f32_mem},
//             {DNNL_ARG_BIAS, bias_f32_mem},
//             {DNNL_ARG_DST, dst_f32_mem}
//         });
//         s.wait();

//         // convert f32 output to bf16
//         memory dst_bf16_user(
//             {y_dims, memory::data_type::bf16, memory::format_tag::ab},
//             eng, out_buf.data());
//         reorder(dst_f32_mem, dst_bf16_user).execute(s, dst_f32_mem, dst_bf16_user);
//         s.wait();
//     }

//     // return NumPy array
//     py::array_t<uint16_t> result({batch, output_dim});
//     auto res_info = result.request();
//     std::memcpy(res_info.ptr, out_buf.data(), out_buf.size() * sizeof(uint16_t));
//     return result;
// }

// PYBIND11_MODULE(linear_layer, m) {
//     m.def("linear", &oneDNN_linear, "Linear using oneDNN bf16 inner_product");
// }
