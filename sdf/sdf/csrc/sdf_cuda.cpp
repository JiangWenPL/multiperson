#include <torch/torch.h>

#define CHECK_CUDA(x) AT_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// CUDA forward declarations

at::Tensor sdf_cuda(
        at::Tensor phi,
        at::Tensor faces,
        at::Tensor vertices);

at::Tensor sdf(
        at::Tensor phi,
        at::Tensor faces,
        at::Tensor vertices) {

	CHECK_INPUT(phi);
	CHECK_INPUT(faces);
	CHECK_INPUT(vertices);

	return sdf_cuda(phi, faces, vertices);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("sdf", &sdf, "SDF (CUDA)");
}
