FROM nvcr.io/nvidia/deepstream-l4t:6.3-samples

RUN apt update && apt install software-properties-common \
    python3.8 python3.8-venv python3-venv python3.8-dev -y

ARG ENVDIR=/opt/venv/

RUN mkdir -p ${ENVDIR} && python3.8 -m venv ${ENVDIR}
ENV PATH=${ENVDIR}/bin:$PATH
RUN chmod -R 777 ${ENVDIR}/

RUN apt install python3.8-vpi2 -y
RUN pip3 install --upgrade pip setuptools && \
    pip3 install cupy-cuda11x Cython

RUN git clone https://github.com/NVIDIA/cuda-python.git && cd cuda-python && \
    git checkout v11.6.1 && \
    python3 setup.py install

RUN ln -s /opt/nvidia/vpi2/lib64/python/vpi.cpython-38-aarch64-linux-gnu.so ${ENVDIR}/lib/python3.8/site-packages/

RUN apt-get update && apt-get install -y libgirepository1.0-dev libcairo2-dev gstreamer-1.0
RUN apt install --reinstall -y gstreamer1.0-plugins-good gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly libavutil56 libavcodec58 libavformat58 libavfilter7 libx264-155 libde265-dev libde265-0 libx265-179 libvpx6 libmpeg2encpp-2.1-0 libmpeg2-4 libmpg123-0; exit 0

RUN pip3 install \ 
    opencv-python==4.5.3.56 \
    pygobject==3.44.1 \
    pycairo==1.24.0

RUN pip3 install https://github.com/NVIDIA-AI-IOT/deepstream_python_apps/releases/download/v1.1.8/pyds-1.1.8-py3-none-linux_aarch64.whl
