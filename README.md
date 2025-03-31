# DeepStream and VPI with Python on NVIDIA Jetson

Exploring the integration of DeepStream and VPI with Python on an NVIDIA Jetson device. Successfully converted the GStreamer buffer into a VPI Image in CUDA but encountered a `cudaErrorIllegalAddress` error after the pipeline ran for some time with a longer video.

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
