#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <torch/torch.h>
#include <cstring>

namespace py = pybind11;

py::array_t<float> oneDNN_linear_f32_torch(py::array_t<float> x_f32,
    py::array_t<float> W_f32,
    py::array_t<float> b_f32) {
auto x_info = x_f32.request();
auto W_info = W_f32.request();
auto b_info = b_f32.request();

if (x_info.ndim != 1 || W_info.ndim != 2 || b_info.ndim != 1) {
throw std::runtime_error("Shapes must be: x [in], W [out,in], b [out]");
}

int input_dim = static_cast<int>(x_info.shape[0]);
int output_dim = static_cast<int>(W_info.shape[0]);

// Create torch tensors from numpy arrays (no copy)
auto x_tensor = torch::from_blob(x_info.ptr, {input_dim}, 
 torch::TensorOptions().dtype(torch::kFloat32));
auto W_tensor = torch::from_blob(W_info.ptr, {output_dim, input_dim}, 
 torch::TensorOptions().dtype(torch::kFloat32));
auto b_tensor = torch::from_blob(b_info.ptr, {output_dim}, 
 torch::TensorOptions().dtype(torch::kFloat32));

// Convert to bf16
auto x_bf16 = x_tensor.to(torch::kBFloat16);
auto W_bf16 = W_tensor.to(torch::kBFloat16);
auto b_bf16 = b_tensor.to(torch::kBFloat16);

// Perform linear operation: y = x @ W^T + b
auto y_bf16 = torch::linear(x_bf16, W_bf16, b_bf16);

// Convert back to float32
auto y_f32 = y_bf16.to(torch::kFloat32);

// Copy result to numpy array
py::array_t<float> result({output_dim});
auto res_info = result.request();
std::memcpy(res_info.ptr, y_f32.data_ptr<float>(), output_dim * sizeof(float));

return result;
}

PYBIND11_MODULE(linear_layer_torch, m) {
    m.def("linear_torch", &oneDNN_linear_f32_torch, "Linear using PyTorch with bf16 conversion, x 1D, W 2D, b 1D");
}











