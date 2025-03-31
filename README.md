# DeepStream and VPI with Python on NVIDIA Jetson

Exploring the integration of DeepStream and VPI with Python on an NVIDIA Jetson device. Successfully converted the GStreamer buffer into a VPI Image in CUDA but encountered a `cudaErrorIllegalAddress` error after the pipeline ran for some time with a longer video.

Raised the question on [NVIDIA forum](https://forums.developer.nvidia.com/t/deepstream-and-vpi-with-python-on-jetson/328796).

## Setup Details

- Hardware: NVIDIA Orin NX
- Jetpack Ver: 5.1.2 [L4T 35.4.1]
- DeepStream Ver: 6.3

## Docker

```bash
docker build . -t deepstream-vpi
docker run -it --rm --runtime nvidia -v ${PWD}:/app/ -w /app/ deepstream-vpi bash
```

## Run Command

```bash
python3 app.py -i file:///opt/nvidia/deepstream/deepstream/samples/streams/sample_1080p_h265.mp4
```

## CUDA Illegal Address Error

```log
libnvosd (1357):(ERROR) : cuGraphicsEGLRegisterImage failed : 700 
0:00:32.958105834   289      0x9fd4980 WARN                 nvinfer gstnvinfer.cpp:2397:gst_nvinfer_output_loop:<primary-inference> error: Internal data stream error.
0:00:32.958127467   289      0x9fd4980 WARN                 nvinfer gstnvinfer.cpp:2397:gst_nvinfer_output_loop:<primary-inference> error: streaming stopped, reason error (-5)
ERROR: Failed to synchronize on cuda copy-coplete-event, cuda err_no:700, err_str:cudaErrorIllegalAddress
0:00:32.958187564   289      0x9fd4980 WARN                 nvinfer gstnvinfer.cpp:2435:gst_nvinfer_output_loop:<primary-inference> error: Failed to dequeue output from inferencing. NvDsInferContext error: NVDSINFER_CUDA_ERROR
0:00:32.958211181   289      0x9fd4980 WARN                 nvinfer gstnvinfer.cpp:679:gst_nvinfer_logger:<primary-inference> NvDsInferContext[UID 1]: Warning from NvDsInferContextImpl::releaseBatchOutput() <nvdsinfer_context_impl.cpp:1884> [UID = 1]: Tried to release an outputBatchID which is already with the context
Error: gst-resource-error-quark: Unable to draw shapes onto video frame by GPU (1): /dvs/git/dirty/git-master_linux/deepstream/sdk/src/gst-plugins/gst-nvdsosd/gstnvdsosd.c(643): gst_nvds_osd_transform_ip (): /GstPipeline:pipeline0/GstNvDsOsd:onscreendisplay
Exiting app

ERROR: Failed to make stream wait on event, cuda err_no:700, err_str:cudaErrorIllegalAddress
ERROR: Preprocessor transform input data failed., nvinfer error:NVDSINFER_CUDA_ERROR
0:00:32.958920963   289      0x9fd49e0 WARN                 nvinfer gstnvinfer.cpp:1404:gst_nvinfer_input_queue_loop:<primary-inference> error: Failed to queue input batch for inferencing
CUDA Runtime error cudaFreeHost(host_) # an illegal memory access was encountered, code = cudaErrorIllegalAddress [ 700 ] in file /dvs/git/dirty/git-master_linux/deepstream/sdk/src/utils/nvll_osd/memory.hpp:67
CUDA Runtime error cudaFree(device_) # an illegal memory access was encountered, code = cudaErrorIllegalAddress [ 700 ] in file /dvs/git/dirty/git-master_linux/deepstream/sdk/src/utils/nvll_osd/memory.hpp:68
CUDA Runtime error cudaFreeHost(host_) # an illegal memory access was encountered, code = cudaErrorIllegalAddress [ 700 ] in file /dvs/git/dirty/git-master_linux/deepstream/sdk/src/utils/nvll_osd/memory.hpp:67
CUDA Runtime error cudaFree(device_) # an illegal memory access was encountered, code = cudaErrorIllegalAddress [ 700 ] in file /dvs/git/dirty/git-master_linux/deepstream/sdk/src/utils/nvll_osd/memory.hpp:68
CUDA Runtime error cudaFreeHost(host_) # an illegal memory access was encountered, code = cudaErrorIllegalAddress [ 700 ] in file /dvs/git/dirty/git-master_linux/deepstream/sdk/src/utils/nvll_osd/memory.hpp:67
CUDA Runtime error cudaFree(device_) # an illegal memory access was encountered, code = cudaErrorIllegalAddress [ 700 ] in file /dvs/git/dirty/git-master_linux/deepstream/sdk/src/utils/nvll_osd/memory.hpp:68
CUDA Runtime error cudaFreeHost(host_) # an illegal memory access was encountered, code = cudaErrorIllegalAddress [ 700 ] in file /dvs/git/dirty/git-master_linux/deepstream/sdk/src/utils/nvll_osd/memory.hpp:67
CUDA Runtime error cudaFree(device_) # an illegal memory access was encountered, code = cudaErrorIllegalAddress [ 700 ] in file /dvs/git/dirty/git-master_linux/deepstream/sdk/src/utils/nvll_osd/memory.hpp:68
CUDA Runtime error cudaFreeHost(host_) # an illegal memory access was encountered, code = cudaErrorIllegalAddress [ 700 ] in file /dvs/git/dirty/git-master_linux/deepstream/sdk/src/utils/nvll_osd/memory.hpp:67
CUDA Runtime error cudaFree(device_) # an illegal memory access was encountered, code = cudaErrorIllegalAddress [ 700 ] in file /dvs/git/dirty/git-master_linux/deepstream/sdk/src/utils/nvll_osd/memory.hpp:68
CUDA Runtime error cudaFreeHost(host_) # an illegal memory access was encountered, code = cudaErrorIllegalAddress [ 700 ] in file /dvs/git/dirty/git-master_linux/deepstream/sdk/src/utils/nvll_osd/memory.hpp:67
CUDA Runtime error cudaFree(device_) # an illegal memory access was encountered, code = cudaErrorIllegalAddress [ 700 ] in file /dvs/git/dirty/git-master_linux/deepstream/sdk/src/utils/nvll_osd/memory.hpp:68
Traceback (most recent call last):
  File "app.py", line 205, in tiler_sink_pad_buffer_probe
    graphics_resources = check_cudart_err(cudart.cudaGraphicsEGLRegisterImage(int(egl_img), 1))
  File "app.py", line 100, in check_cudart_err
    raise RuntimeError(format_cudart_err(err))
RuntimeError: cudaErrorIllegalAddress(700): an illegal memory access was encountered

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "app.py", line 233, in tiler_sink_pad_buffer_probe
    cudart.cudaGraphicsUnregisterResource(graphics_resources)
UnboundLocalError: local variable 'graphics_resources' referenced before assignment
ERROR: Failed to synchronize on cuda copy-coplete-event, cuda err_no:700, err_str:cudaErrorIllegalAddress
0:00:32.968563757   289      0x9fd4980 WARN                 nvinfer gstnvinfer.cpp:2435:gst_nvinfer_output_loop:<primary-inference> error: Failed to dequeue output from inferencing. NvDsInferContext error: NVDSINFER_CUDA_ERROR
0:00:32.968617551   289      0x9fd4980 WARN                 nvinfer gstnvinfer.cpp:679:gst_nvinfer_logger:<primary-inference> NvDsInferContext[UID 1]: Warning from NvDsInferContextImpl::releaseBatchOutput() <nvdsinfer_context_impl.cpp:1884> [UID = 1]: Tried to release an outputBatchID which is already with the context
0:00:33.037163663   289 0xfffa9800b760 WARN                 nvinfer gstnvinfer.cpp:1429:convert_batch_and_push_to_input_thread:<primary-inference> error: Failed to set cuda device 0
0:00:33.037204017   289 0xfffa9800b760 WARN                 nvinfer gstnvinfer.cpp:1429:convert_batch_and_push_to_input_thread:<primary-inference> error: cudaSetDevice failed with error cudaErrorCudartUnloading
Unable to set device in gst_nvstreammux_src_collect_buffers
Traceback (most recent call last):
  File "app.py", line 205, in tiler_sink_pad_buffer_probe
    graphics_resources = check_cudart_err(cudart.cudaGraphicsEGLRegisterImage(int(egl_img), 1))
  File "app.py", line 100, in check_cudart_err
    raise RuntimeError(format_cudart_err(err))
RuntimeError: cudaErrorIllegalAddress(700): an illegal memory access was encountered

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "app.py", line 233, in tiler_sink_pad_buffer_probe
    cudart.cudaGraphicsUnregisterResource(graphics_resources)
UnboundLocalError: local variable 'graphics_resources' referenced before assignment
ERROR: dequeue buffer failed to set cuda device((null)), cuda err_no:4, err_str:cudaErrorCudartUnloading
0:00:33.038389045   289      0x9fd4980 WARN                 nvinfer gstnvinfer.cpp:2435:gst_nvinfer_output_loop:<primary-inference> error: Failed to dequeue output from inferencing. NvDsInferContext error: NVDSINFER_CUDA_ERROR
0:00:33.038434518   289      0x9fd4980 WARN                 nvinfer gstnvinfer.cpp:679:gst_nvinfer_logger:<primary-inference> NvDsInferContext[UID 1]: Warning from NvDsInferContextImpl::releaseBatchOutput() <nvdsinfer_context_impl.cpp:1884> [UID = 1]: Tried to release an outputBatchID which is already with the context

 *** Unable to set device in gst_nvvideoconvert_transform Line 3326
0:00:33.038502009   289      0x9fd4980 ERROR         nvvideoconvert gstnvvideoconvert.c:4120:gst_nvvideoconvert_transform: Set Device failed
0:00:33.038555738   289      0x9fd4980 WARN                 nvinfer gstnvinfer.cpp:2397:gst_nvinfer_output_loop:<primary-inference> error: Internal data stream error.
0:00:33.038566811   289      0x9fd4980 WARN                 nvinfer gstnvinfer.cpp:2397:gst_nvinfer_output_loop:<primary-inference> error: streaming stopped, reason error (-5)

 *** Unable to set device in gst_nvvideoconvert_transform Line 3326
0:00:33.038612092   289      0x9fd4980 ERROR         nvvideoconvert gstnvvideoconvert.c:4120:gst_nvvideoconvert_transform: Set Device failed
Unable to set device in gst_nvstreammux_src_collect_buffers
Unable to set device in gst_nvstreammux_src_collect_buffers
[WARN ] 2025-03-31 09:31:35 (cudaErrorIllegalAddress)
[ERROR] 2025-03-31 09:31:35 Error destroying cuda device: �n�(
[WARN ] 2025-03-31 09:31:35 (cudaErrorIllegalAddress)
[WARN ] 2025-03-31 09:31:35 (cudaErrorIllegalAddress)
[WARN ] 2025-03-31 09:31:35 (cudaErrorIllegalAddress)
[WARN ] 2025-03-31 09:31:35 (cudaErrorIllegalAddress)
[WARN ] 2025-03-31 09:31:35 (cudaErrorIllegalAddress)
[WARN ] 2025-03-31 09:31:35 (cudaErrorIllegalAddress)
[WARN ] 2025-03-31 09:31:35 (cudaErrorIllegalAddress)
[WARN ] 2025-03-31 09:31:35 (cudaErrorIllegalAddress)
[WARN ] 2025-03-31 09:31:35 (cudaErrorIllegalAddress)
[WARN ] 2025-03-31 09:31:35 (cudaErrorIllegalAddress)
[WARN ] 2025-03-31 09:31:35 (cudaErrorIllegalAddress)
[WARN ] 2025-03-31 09:31:35 (cudaErrorIllegalAddress)
[WARN ] 2025-03-31 09:31:35 (cudaErrorIllegalAddress)
[WARN ] 2025-03-31 09:31:35 (cudaErrorIllegalAddress)
[WARN ] 2025-03-31 09:31:35 (cudaErrorIllegalAddress)
[WARN ] 2025-03-31 09:31:35 (cudaErrorIllegalAddress)
[WARN ] 2025-03-31 09:31:35 (cudaErrorIllegalAddress)
[WARN ] 2025-03-31 09:31:35 (cudaErrorIllegalAddress)
[WARN ] 2025-03-31 09:31:35 (cudaErrorIllegalAddress)
[WARN ] 2025-03-31 09:31:35 (cudaErrorIllegalAddress)
[WARN ] 2025-03-31 09:31:35 (cudaErrorIllegalAddress)
[WARN ] 2025-03-31 09:31:35 (cudaErrorIllegalAddress)
[WARN ] 2025-03-31 09:31:35 (cudaErrorIllegalAddress)
[WARN ] 2025-03-31 09:31:35 (cudaErrorIllegalAddress)
[WARN ] 2025-03-31 09:31:35 (cudaErrorIllegalAddress)
[WARN ] 2025-03-31 09:31:35 (cudaErrorIllegalAddress)
[WARN ] 2025-03-31 09:31:35 (cudaErrorIllegalAddress)
[WARN ] 2025-03-31 09:31:35 (cudaErrorIllegalAddress)
[WARN ] 2025-03-31 09:31:35 (cudaErrorIllegalAddress)
[WARN ] 2025-03-31 09:31:35 (cudaErrorIllegalAddress)
[WARN ] 2025-03-31 09:31:35 (cudaErrorIllegalAddress)
[WARN ] 2025-03-31 09:31:35 (cudaErrorIllegalAddress)
[WARN ] 2025-03-31 09:31:35 (cudaErrorIllegalAddress)
[WARN ] 2025-03-31 09:31:35 (cudaErrorIllegalAddress)
[WARN ] 2025-03-31 09:31:35 (cudaErrorIllegalAddress)
[WARN ] 2025-03-31 09:31:35 (cudaErrorIllegalAddress)
[WARN ] 2025-03-31 09:31:35 (cudaErrorIllegalAddress)
[WARN ] 2025-03-31 09:31:35 (cudaErrorIllegalAddress)
[WARN ] 2025-03-31 09:31:35 (cudaErrorIllegalAddress)
[WARN ] 2025-03-31 09:31:35 (cudaErrorIllegalAddress)
[WARN ] 2025-03-31 09:31:35 (cudaErrorIllegalAddress)
[WARN ] 2025-03-31 09:31:35 (cudaErrorIllegalAddress)
[WARN ] 2025-03-31 09:31:35 (cudaErrorIllegalAddress)
[WARN ] 2025-03-31 09:31:35 (cudaErrorIllegalAddress)
[WARN ] 2025-03-31 09:31:35 (cudaErrorIllegalAddress)
[WARN ] 2025-03-31 09:31:35 (cudaErrorIllegalAddress)
[WARN ] 2025-03-31 09:31:35 (cudaErrorIllegalAddress)
[WARN ] 2025-03-31 09:31:35 (cudaErrorIllegalAddress)
[WARN ] 2025-03-31 09:31:35 (cudaErrorIllegalAddress)
[WARN ] 2025-03-31 09:31:35 (cudaErrorIllegalAddress)
[WARN ] 2025-03-31 09:31:35 (cudaErrorIllegalAddress)
[WARN ] 2025-03-31 09:31:35 (cudaErrorIllegalAddress)
[WARN ] 2025-03-31 09:31:35 (cudaErrorIllegalAddress)
[WARN ] 2025-03-31 09:31:35 (cudaErrorIllegalAddress)
[WARN ] 2025-03-31 09:31:35 (cudaErrorIllegalAddress)
[WARN ] 2025-03-31 09:31:35 (cudaErrorIllegalAddress)
[WARN ] 2025-03-31 09:31:35 (cudaErrorIllegalAddress)
[WARN ] 2025-03-31 09:31:35 (cudaErrorIllegalAddress)
[WARN ] 2025-03-31 09:31:35 (cudaErrorIllegalAddress)
[WARN ] 2025-03-31 09:31:35 (cudaErrorIllegalAddress)
[WARN ] 2025-03-31 09:31:35 (cudaErrorIllegalAddress)
[WARN ] 2025-03-31 09:31:35 (cudaErrorIllegalAddress)
[WARN ] 2025-03-31 09:31:35 (cudaErrorIllegalAddress)
[WARN ] 2025-03-31 09:31:35 (cudaErrorIllegalAddress)
[ERROR] 2025-03-31 09:31:35 Exiting the Stream worker thread failed with exception: VPI_ERROR_INTERNAL: (cudaErrorIllegalAddress)
[WARN ] 2025-03-31 09:31:35 (cudaErrorIllegalAddress)
[ERROR] 2025-03-31 09:31:35 Error destroying cuda device: о�(
[WARN ] 2025-03-31 09:31:35 (cudaErrorIllegalAddress)
[WARN ] 2025-03-31 09:31:35 (cudaErrorIllegalAddress)
[WARN ] 2025-03-31 09:31:35 (cudaErrorIllegalAddress)
[WARN ] 2025-03-31 09:31:35 (cudaErrorIllegalAddress)
[WARN ] 2025-03-31 09:31:35 (cudaErrorIllegalAddress)
[WARN ] 2025-03-31 09:31:35 (cudaErrorIllegalAddress)
[WARN ] 2025-03-31 09:31:35 (cudaErrorIllegalAddress)
[WARN ] 2025-03-31 09:31:35 (cudaErrorIllegalAddress)
[WARN ] 2025-03-31 09:31:35 (cudaErrorIllegalAddress)
[WARN ] 2025-03-31 09:31:35 (cudaErrorIllegalAddress)
[WARN ] 2025-03-31 09:31:35 (cudaErrorIllegalAddress)
[WARN ] 2025-03-31 09:31:35 (cudaErrorIllegalAddress)
[WARN ] 2025-03-31 09:31:35 (cudaErrorIllegalAddress)
[WARN ] 2025-03-31 09:31:35 (cudaErrorIllegalAddress)
[WARN ] 2025-03-31 09:31:35 (cudaErrorIllegalAddress)
[WARN ] 2025-03-31 09:31:35 (cudaErrorIllegalAddress)
[WARN ] 2025-03-31 09:31:35 (cudaErrorIllegalAddress)
[WARN ] 2025-03-31 09:31:35 (cudaErrorIllegalAddress)
[WARN ] 2025-03-31 09:31:35 (cudaErrorIllegalAddress)
[WARN ] 2025-03-31 09:31:35 (cudaErrorIllegalAddress)
[WARN ] 2025-03-31 09:31:35 (cudaErrorIllegalAddress)
[WARN ] 2025-03-31 09:31:35 (cudaErrorIllegalAddress)
[WARN ] 2025-03-31 09:31:35 (cudaErrorIllegalAddress)
[WARN ] 2025-03-31 09:31:35 (cudaErrorIllegalAddress)
[WARN ] 2025-03-31 09:31:35 (cudaErrorIllegalAddress)
[WARN ] 2025-03-31 09:31:35 (cudaErrorIllegalAddress)
[WARN ] 2025-03-31 09:31:35 (cudaErrorIllegalAddress)
[WARN ] 2025-03-31 09:31:35 (cudaErrorIllegalAddress)
[WARN ] 2025-03-31 09:31:35 (cudaErrorIllegalAddress)
[WARN ] 2025-03-31 09:31:35 (cudaErrorIllegalAddress)
[WARN ] 2025-03-31 09:31:35 (cudaErrorIllegalAddress)
[WARN ] 2025-03-31 09:31:35 (cudaErrorIllegalAddress)
[WARN ] 2025-03-31 09:31:35 (cudaErrorIllegalAddress)
[WARN ] 2025-03-31 09:31:35 (cudaErrorIllegalAddress)
[WARN ] 2025-03-31 09:31:35 (cudaErrorIllegalAddress)
[WARN ] 2025-03-31 09:31:35 (cudaErrorIllegalAddress)
[WARN ] 2025-03-31 09:31:35 (cudaErrorIllegalAddress)
[WARN ] 2025-03-31 09:31:35 (cudaErrorIllegalAddress)
[WARN ] 2025-03-31 09:31:35 (cudaErrorIllegalAddress)
[WARN ] 2025-03-31 09:31:35 (cudaErrorIllegalAddress)
[WARN ] 2025-03-31 09:31:35 (cudaErrorIllegalAddress)
[WARN ] 2025-03-31 09:31:35 (cudaErrorIllegalAddress)
[WARN ] 2025-03-31 09:31:35 (cudaErrorIllegalAddress)
[WARN ] 2025-03-31 09:31:35 (cudaErrorIllegalAddress)
[WARN ] 2025-03-31 09:31:35 (cudaErrorIllegalAddress)
[WARN ] 2025-03-31 09:31:35 (cudaErrorIllegalAddress)
[WARN ] 2025-03-31 09:31:35 (cudaErrorIllegalAddress)
[WARN ] 2025-03-31 09:31:35 (cudaErrorIllegalAddress)
[WARN ] 2025-03-31 09:31:35 (cudaErrorIllegalAddress)
[WARN ] 2025-03-31 09:31:35 (cudaErrorIllegalAddress)
[WARN ] 2025-03-31 09:31:35 (cudaErrorIllegalAddress)
[WARN ] 2025-03-31 09:31:35 (cudaErrorIllegalAddress)
[WARN ] 2025-03-31 09:31:35 (cudaErrorIllegalAddress)
[WARN ] 2025-03-31 09:31:35 (cudaErrorIllegalAddress)
[WARN ] 2025-03-31 09:31:35 (cudaErrorIllegalAddress)
[WARN ] 2025-03-31 09:31:35 (cudaErrorIllegalAddress)
[WARN ] 2025-03-31 09:31:35 (cudaErrorIllegalAddress)
[WARN ] 2025-03-31 09:31:35 (cudaErrorIllegalAddress)
[WARN ] 2025-03-31 09:31:35 (cudaErrorIllegalAddress)
[WARN ] 2025-03-31 09:31:35 (cudaErrorIllegalAddress)
[WARN ] 2025-03-31 09:31:35 (cudaErrorIllegalAddress)
[WARN ] 2025-03-31 09:31:35 (cudaErrorIllegalAddress)
[WARN ] 2025-03-31 09:31:35 (cudaErrorIllegalAddress)
[WARN ] 2025-03-31 09:31:35 (cudaErrorIllegalAddress)
[WARN ] 2025-03-31 09:31:35 (cudaErrorIllegalAddress)
[WARN ] 2025-03-31 09:31:35 (cudaErrorIllegalAddress)
[WARN ] 2025-03-31 09:31:35 (cudaErrorIllegalAddress)
[WARN ] 2025-03-31 09:31:35 (cudaErrorIllegalAddress)
[WARN ] 2025-03-31 09:31:35 (cudaErrorIllegalAddress)
[WARN ] 2025-03-31 09:31:35 (cudaErrorIllegalAddress)
[WARN ] 2025-03-31 09:31:35 (cudaErrorIllegalAddress)
[WARN ] 2025-03-31 09:31:35 (cudaErrorIllegalAddress)

Segmentation fault (core dumped)
```
