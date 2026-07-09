/*
 * Kipp CUDA backend.
 *
 * CUDA kernels and their host-side launch code stay isolated here so CUDA
 * changes cannot silently alter the Metal or CPU reference paths.
 */

extern "C" const char *kipp_cuda_scaffold_status(void) {
    return "Kipp CUDA backend is not implemented";
}
