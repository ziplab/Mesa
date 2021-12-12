// Copyright (c) 2021-present, Zhuang AI Group.
// All rights reserved.

#include <torch/extension.h>
#include <ATen/NativeFunctions.h>
#include <ATen/Config.h>

#include <array>

std::tuple<at::Tensor, at::Tensor> conv2d_backward(
    const at::Tensor& input,
    const at::Tensor& grad_output,
    const at::Tensor& weight,
    c10::ArrayRef<long int> padding,
    c10::ArrayRef<long int> stride,
    c10::ArrayRef<long int> dilation,
    int64_t groups,
    bool benchmark,
    bool deterministic,
#if TORCH_VERSION_MAJOR >= 1 && TORCH_VERSION_MINOR >= 7
    bool allow_tf32,
#endif
    std::array<bool, 2> output_mask
    ) {

    return at::cudnn_convolution_backward(
        input,
        grad_output,
        weight,
        padding,
        stride,
        dilation,
        groups,
        benchmark,
        deterministic,
#if TORCH_VERSION_MAJOR >= 1 && TORCH_VERSION_MINOR >= 7
        allow_tf32,
#endif
        output_mask);
}

// output, save_mean, save_var, reserve
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> batch_norm_forward(
    const at::Tensor& input,
    const at::Tensor& weight,
    const at::Tensor& bias,
    const at::Tensor& running_mean,
    const at::Tensor& running_var,
    bool training,
    double average_factor,
    double epsilon) {

    return at::cudnn_batch_norm(
        input,
        weight,
        bias,
        running_mean,
        running_var,
        training,
        average_factor,
        epsilon
        );
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> batch_norm_backward(
    const at::Tensor& input,
    const at::Tensor& grad_output,
    const at::Tensor& weight,
    const at::Tensor& running_mean,
    const at::Tensor& running_var,
    const at::Tensor& save_mean,
    const at::Tensor& save_var,
    double epsilon,
    const at::Tensor& reserveSpace) {

    return at::cudnn_batch_norm_backward(
        input,
        grad_output,
        weight,
        running_mean,
        running_var,
        save_mean,
        save_var,
        epsilon,
        reserveSpace);

}

at::Tensor gelu_backward_cpu(
    const at::Tensor& grad_output,
    const at::Tensor& input) {
#if TORCH_VERSION_MAJOR >= 1 && TORCH_VERSION_MINOR >= 10
    return at::gelu_backward(grad_output, input);
#else
    return at::native::gelu_backward_cpu(grad_output, input);
#endif
}


at::Tensor gelu_backward_cuda(
    const at::Tensor& grad_output,
    const at::Tensor& input) {
#if TORCH_VERSION_MAJOR >= 1 && TORCH_VERSION_MINOR >= 10
    return at::gelu_backward(grad_output, input);
#else
    return at::native::gelu_backward_cuda(grad_output, input);
#endif
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> layer_norm_forward_cpu(
    const at::Tensor & input,
#if TORCH_VERSION_MAJOR >= 1 && TORCH_VERSION_MINOR >= 8
    at::IntArrayRef normalized_shape,
#endif
    const at::Tensor & weight,
    const at::Tensor & bias,
#if TORCH_VERSION_MAJOR >= 1 && TORCH_VERSION_MINOR >= 8
#else
    int64_t M, int64_t N,
#endif
    double eps) {

#if TORCH_VERSION_MAJOR >= 1 && TORCH_VERSION_MINOR >= 8
    return at::native::layer_norm_cpu(input, normalized_shape, weight, bias, eps);
#else
    return at::native::layer_norm_cpu(input, weight, bias, M, N, eps);
#endif
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> layer_norm_backward_cpu(
    const at::Tensor & grad_out,
    const at::Tensor & input,
#if TORCH_VERSION_MAJOR >= 1 && TORCH_VERSION_MINOR >= 8
    at::IntArrayRef normalized_shape,
#endif
    const at::Tensor & mean,
    const at::Tensor & rstd,
    const at::Tensor & weight,
#if TORCH_VERSION_MAJOR >= 1 && TORCH_VERSION_MINOR >= 8
    const at::Tensor & bias,
#else
    int64_t M, int64_t N,
#endif
    std::array<bool,3> output_mask) {

#if TORCH_VERSION_MAJOR >= 1 && TORCH_VERSION_MINOR >= 8
    return at::native::layer_norm_backward_cpu(grad_out, input, normalized_shape, mean, rstd, weight, bias, output_mask);
#else
    return at::native::layer_norm_backward_cpu(grad_out, input, mean, rstd, weight, M, N, output_mask);
#endif
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> layer_norm_forward_cuda(
    const at::Tensor & input,
#if TORCH_VERSION_MAJOR >= 1 && TORCH_VERSION_MINOR >= 8
    at::IntArrayRef normalized_shape,
#endif
    const at::Tensor & weight,
    const at::Tensor & bias,
#if TORCH_VERSION_MAJOR >= 1 && TORCH_VERSION_MINOR >= 8
#else
    int64_t M, int64_t N,
#endif
    double eps) {
#if TORCH_VERSION_MAJOR >= 1 && TORCH_VERSION_MINOR >= 8
    return at::native::layer_norm_cuda(input, normalized_shape, weight, bias, eps);
#else
    return at::native::layer_norm_cuda(input, weight, bias, M, N, eps);
#endif
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> layer_norm_backward_cuda(
    const at::Tensor & grad_out,
    const at::Tensor & input,
#if TORCH_VERSION_MAJOR >= 1 && TORCH_VERSION_MINOR >= 8
    at::IntArrayRef normalized_shape,
#endif
    const at::Tensor & mean,
    const at::Tensor & rstd,
    const at::Tensor & weight,
#if TORCH_VERSION_MAJOR >= 1 && TORCH_VERSION_MINOR >= 8
    const at::Tensor & bias,
#else
    int64_t M, int64_t N,
#endif
    std::array<bool,3> output_mask) {
#if TORCH_VERSION_MAJOR >= 1 && TORCH_VERSION_MINOR >= 8
    return at::native::layer_norm_backward_cuda(grad_out, input, normalized_shape, mean, rstd, weight, bias, output_mask);
#else
    return at::native::layer_norm_backward_cuda(grad_out, input, mean, rstd, weight, M, N, output_mask);
#endif
}

at::Tensor softmax_backward_cpu(
    const at::Tensor& grad_output,
    const at::Tensor& output,
    int64_t dim,
    const at::Tensor& self) {
#if TORCH_VERSION_MAJOR >= 1 && TORCH_VERSION_MINOR >= 10
    return at::_softmax_backward_data(grad_output, output, dim, self);
#else
    return at::native::softmax_backward_cpu(grad_output, output, dim, self);
#endif
}

at::Tensor softmax_backward_cuda(
    const at::Tensor& grad_output,
    const at::Tensor& output,
    int64_t dim,
    const at::Tensor& self) {
#if TORCH_VERSION_MAJOR >= 1 && TORCH_VERSION_MINOR >= 10
    return at::_softmax_backward_data(grad_output, output, dim, self);
#else
    return at::native::softmax_backward_cuda(grad_output, output, dim, self);
#endif
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("conv2d_backward", &conv2d_backward, "2d convolution backward");
    m.def("batch_norm_forward",  &batch_norm_forward,  "batch norm forward");
    m.def("batch_norm_backward", &batch_norm_backward, "batch norm backward");
    m.def("gelu_backward_cpu",  &gelu_backward_cpu,  "gelu backward (cpu version)");
    m.def("gelu_backward_cuda", &gelu_backward_cuda, "gelu backward (cuda version)");
    m.def("layer_norm_forward_cpu",  &layer_norm_forward_cpu,  "layer norm forward (cpu version)");
    m.def("layer_norm_backward_cpu", &layer_norm_backward_cpu, "layer norm backward (cpu version)");
    m.def("layer_norm_forward_cuda",  &layer_norm_forward_cuda,  "layer norm forward (cuda version)");
    m.def("layer_norm_backward_cuda", &layer_norm_backward_cuda, "layer norm backward (cuda version)");
    m.def("softmax_backward_cpu",  &softmax_backward_cpu,  "softmax backward (cpu version)");
    m.def("softmax_backward_cuda", &softmax_backward_cuda, "softmax backward (cuda version)");
}

