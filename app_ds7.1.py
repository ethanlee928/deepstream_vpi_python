#!/usr/bin/env python3

################################################################################
# SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
################################################################################

import sys
from ctypes import (
    CDLL,
    POINTER,
    Structure,
    addressof,
    byref,
    c_bool,
    c_int,
    c_uint,
    c_ulong,
    c_void_p,
    memmove,
    sizeof,
)

sys.path.append('../')
import gi

gi.require_version('Gst', '1.0')
from gi.repository import GLib, Gst
import math
from common.bus_call import bus_call
from common.FPS import PERF_DATA
import pyds
import argparse

import numpy as np
import cupy as cp
import vpi
from cuda import cudart


nvbufsurface = CDLL("libnvbufsurface.so")

max_planes = 4
structure_padding = 4


perf_data = None

MAX_DISPLAY_LEN = 64
PGIE_CLASS_ID_VEHICLE = 0
PGIE_CLASS_ID_BICYCLE = 1
PGIE_CLASS_ID_PERSON = 2
PGIE_CLASS_ID_ROADSIGN = 3
MUXER_OUTPUT_WIDTH = 1920
MUXER_OUTPUT_HEIGHT = 1080
MUXER_BATCH_TIMEOUT_USEC = 33000
TILED_OUTPUT_WIDTH = 1920
TILED_OUTPUT_HEIGHT = 1080
GST_CAPS_FEATURES_NVMM = "memory:NVMM"
pgie_classes_str= ["Vehicle", "TwoWheeler", "Person","RoadSign"]


##### EXPERIMENTAL #####


def format_cudart_err(err):
    return (
        f"{cudart.cudaGetErrorName(err)[1].decode('utf-8')}({int(err)}): "
        f"{cudart.cudaGetErrorString(err)[1].decode('utf-8')}"
    )


def check_cudart_err(args):
    if isinstance(args, tuple):
        assert len(args) >= 1
        err = args[0]
        if len(args) == 1:
            ret = None
        elif len(args) == 2:
            ret = args[1]
        else:
            ret = args[1:]
    else:
        err = args
        ret = None

    assert isinstance(err, cudart.cudaError_t), type(err)
    if err != cudart.cudaError_t.cudaSuccess:
        raise RuntimeError(format_cudart_err(err))

    return ret


class NvBufSurfacePlaneParams(Structure):
    _fields_ = [
        ("num_planes", c_uint),
        ("width", c_uint * max_planes),
        ("height", c_uint * max_planes),
        ("pitch", c_uint * max_planes),
        ("offset", c_uint * max_planes),
        ("psize", c_uint * max_planes),
        ("bytesPerPix", c_uint * max_planes),
        ("_reserved", c_void_p * max_planes * structure_padding),
    ]


class NvBufSurfaceMappedAddr(Structure):
    _fields_ = [
        ("addr", c_void_p * max_planes),
        ("eglImage", c_void_p),
        ("_reserved", c_void_p * structure_padding),
    ]


class NvBufSurfaceParams(Structure):
    _fields_ = [
        ("width", c_uint),
        ("height", c_uint),
        ("pitch", c_uint),
        ("colorFormat", c_int),
        ("layout", c_int),
        ("bufferDesc", c_ulong),
        ("dataSize", c_uint),
        ("dataPtr", c_void_p),
        ("planeParams", NvBufSurfacePlaneParams),
        ("mappedAddr", NvBufSurfaceMappedAddr),
        ("_reserved", c_void_p * structure_padding),
    ]


class NvBufSurface(Structure):
    _fields_ = [
        ("gpuId", c_uint),
        ("batchSize", c_uint),
        ("numFilled", c_uint),
        ("isContiguous", c_bool),
        ("memType", c_int),
        ("surfaceList", POINTER(NvBufSurfaceParams)),
        ("_reserved", c_void_p * structure_padding),
    ]

    def __init__(self, gst_map_info):
        memmove(
            addressof(self),
            gst_map_info.data,
            min(sizeof(self), len(gst_map_info.data)),
        )
        status = nvbufsurface.NvBufSurfaceMapEglImage(self, -1)

    def struct_copy_from(self, other_buf_surface):
        self.batchSize = other_buf_surface.batchSize
        self.numFilled = other_buf_surface.numFilled
        self.isContiguous = other_buf_surface.isContiguous
        self.memType = other_buf_surface.memType
        self.surfaceList = (NvBufSurfaceParams * other_buf_surface.numFilled)()
        for surface_ix in range(other_buf_surface.numFilled):
            self.surfaceList[surface_ix] = NvBufSurfaceParams()
            self.surfaceList[surface_ix].width = other_buf_surface.surfaceList[surface_ix].width
            self.surfaceList[surface_ix].height = other_buf_surface.surfaceList[surface_ix].height
            self.surfaceList[surface_ix].pitch = other_buf_surface.surfaceList[surface_ix].pitch
            self.surfaceList[surface_ix].colorFormat = other_buf_surface.surfaceList[surface_ix].colorFormat
            self.surfaceList[surface_ix].layout = other_buf_surface.surfaceList[surface_ix].layout
            self.surfaceList[surface_ix].bufferDesc = other_buf_surface.surfaceList[surface_ix].bufferDesc
            self.surfaceList[surface_ix].dataSize = other_buf_surface.surfaceList[surface_ix].dataSize
            self.surfaceList[surface_ix].planeParams = other_buf_surface.surfaceList[surface_ix].planeParams

    def mem_copy_from(self, other_buf_surface):
        copy_result = nvbufsurface.NvBufSurfaceCopy(byref(other_buf_surface), byref(self))
        assert copy_result == 0


########################

# tiler_sink_pad_buffer_probe  will extract metadata received on tiler src pad
# and modify the frame buffer using cupy
def tiler_sink_pad_buffer_probe(pad, info, u_data):
    gst_buffer = info.get_buffer()
    if not gst_buffer:
        print("Unable to get GstBuffer ")
        return

    is_mapped, map_info = gst_buffer.map(Gst.MapFlags.READ)
    if not is_mapped:
        raise RuntimeError("Cannot map an gst buffer.")

    try:
        owner = None
        surface = NvBufSurface(map_info)
        egl_img = surface.surfaceList[0].mappedAddr.eglImage
        if not egl_img:
            raise RuntimeError("Failed to map egl image.")

        graphics_resources = check_cudart_err(cudart.cudaGraphicsEGLRegisterImage(int(egl_img), 1))
        cuda_egl_frame = check_cudart_err(cudart.cudaGraphicsResourceGetMappedEglFrame(graphics_resources, 0, 0))
        dataptr = cuda_egl_frame.frame.pPitch[0].ptr
        width = cuda_egl_frame.planeDesc[0].width
        height = cuda_egl_frame.planeDesc[0].height
        pitch = cuda_egl_frame.planeDesc[0].pitch
        channels = cuda_egl_frame.planeDesc[0].numChannels
        size = width * height * channels
        strides = (pitch, channels, 1)
        unownedmem = cp.cuda.UnownedMemory(dataptr, size, owner)
        memptr = cp.cuda.MemoryPointer(unownedmem, 0)
        n_frame_gpu = cp.ndarray(
            shape=(height, width, channels),
            dtype=np.uint8,
            memptr=memptr,
            strides=strides,
            order="C",
        )

        vpi_image = vpi.asimage(n_frame_gpu)
        with vpi.Backend.CUDA:
            output = vpi_image.rescale((vpi_image.width // 2, vpi_image.height // 2))
            output = output.convert(vpi.Format.BGR8)
        print(f"{vpi_image.width}x{vpi_image.height} {vpi_image.format}")
        print(f"{output.width}x{output.height} {output.format}")

    finally:
        del vpi_image, output
        check_cudart_err(
            cudart.cudaGraphicsUnregisterResource(graphics_resources)
        )  # This should be called for proper cleanup, but causes illegal memory access error.
        nvbufsurface.NvBufSurfaceUnMapEglImage(surface, -1)
        nvbufsurface.NvBufSurfaceUnMap(surface, -1, -1)
        gst_buffer.unmap(map_info)

    return Gst.PadProbeReturn.OK


def cb_newpad(decodebin, decoder_src_pad, data):
    print("In cb_newpad\n")
    caps = decoder_src_pad.get_current_caps()
    gststruct = caps.get_structure(0)
    gstname = gststruct.get_name()
    source_bin = data
    features = caps.get_features(0)

    # Need to check if the pad created by the decodebin is for video and not
    # audio.
    if (gstname.find("video") != -1):
        # Link the decodebin pad only if decodebin has picked nvidia
        # decoder plugin nvdec_*. We do this by checking if the pad caps contain
        # NVMM memory features.
        if features.contains("memory:NVMM"):
            # Get the source bin ghost pad
            bin_ghost_pad = source_bin.get_static_pad("src")
            if not bin_ghost_pad.set_target(decoder_src_pad):
                sys.stderr.write("Failed to link decoder src pad to source bin ghost pad\n")
        else:
            sys.stderr.write(" Error: Decodebin did not pick nvidia decoder plugin.\n")


def decodebin_child_added(child_proxy, Object, name, user_data):
    print("Decodebin child added:", name, "\n")
    if name.find("decodebin") != -1:
        Object.connect("child-added", decodebin_child_added, user_data)

    if "source" in name:
        source_element = child_proxy.get_by_name("source")
        if source_element.find_property('drop-on-latency') != None:
            Object.set_property("drop-on-latency", True)

def create_source_bin(index, uri):
    print("Creating source bin")

    # Create a source GstBin to abstract this bin's content from the rest of the
    # pipeline
    bin_name = "source-bin-%02d" % index
    print(bin_name)
    nbin = Gst.Bin.new(bin_name)
    if not nbin:
        sys.stderr.write(" Unable to create source bin \n")

    # Source element for reading from the uri.
    # We will use decodebin and let it figure out the container format of the
    # stream and the codec and plug the appropriate demux and decode plugins.
    uri_decode_bin = Gst.ElementFactory.make("uridecodebin", "uri-decode-bin")
    if not uri_decode_bin:
        sys.stderr.write(" Unable to create uri decode bin \n")
    # We set the input uri to the source element
    uri_decode_bin.set_property("uri", uri)
    # Connect to the "pad-added" signal of the decodebin which generates a
    # callback once a new pad for raw data has beed created by the decodebin
    uri_decode_bin.connect("pad-added", cb_newpad, nbin)
    uri_decode_bin.connect("child-added", decodebin_child_added, nbin)

    # We need to create a ghost pad for the source bin which will act as a proxy
    # for the video decoder src pad. The ghost pad will not have a target right
    # now. Once the decode bin creates the video decoder and generates the
    # cb_newpad callback, we will set the ghost pad target to the video decoder
    # src pad.
    Gst.Bin.add(nbin, uri_decode_bin)
    bin_pad = nbin.add_pad(Gst.GhostPad.new_no_target("src", Gst.PadDirection.SRC))
    if not bin_pad:
        sys.stderr.write(" Failed to add ghost pad in source bin \n")
        return None
    return nbin

def main(args):
    global perf_data
    perf_data = PERF_DATA(len(args))
    number_sources = len(args)

    # Standard GStreamer initialization
    Gst.init(None)

    # Create gstreamer elements */
    # Create Pipeline element that will form a connection of other elements
    print("Creating Pipeline \n ")
    pipeline = Gst.Pipeline()
    is_live = False

    if not pipeline:
        sys.stderr.write(" Unable to create Pipeline \n")
    print("Creating streammux \n ")

    # Create nvstreammux instance to form batches from one or more sources.
    streammux = Gst.ElementFactory.make("nvstreammux", "Stream-muxer")
    if not streammux:
        sys.stderr.write(" Unable to create NvStreamMux \n")

    pipeline.add(streammux)
    for i in range(number_sources):
        print("Creating source_bin ", i, " \n ")
        uri_name = args[i]
        if uri_name.find("rtsp://") == 0:
            is_live = True
        source_bin = create_source_bin(i, uri_name)
        if not source_bin:
            sys.stderr.write("Unable to create source bin \n")
        pipeline.add(source_bin)
        padname = "sink_%u" % i
        sinkpad = streammux.get_request_pad(padname)
        if not sinkpad:
            sys.stderr.write("Unable to create sink pad bin \n")
        srcpad = source_bin.get_static_pad("src")
        if not srcpad:
            sys.stderr.write("Unable to create src pad bin \n")
        srcpad.link(sinkpad)
    print("Creating Pgie \n ")
    pgie = Gst.ElementFactory.make("nvinfer", "primary-inference")
    if not pgie:
        sys.stderr.write(" Unable to create pgie \n")
    # Add nvvidconv1 and filter1 to convert the frames to RGBA
    # which is easier to work with in Python.
    print("Creating nvvidconv1 \n ")
    nvvidconv1 = Gst.ElementFactory.make("nvvideoconvert", "convertor1")
    if not nvvidconv1:
        sys.stderr.write(" Unable to create nvvidconv1 \n")
    print("Creating filter1 \n ")
    caps1 = Gst.Caps.from_string("video/x-raw(memory:NVMM), format=RGBA")
    filter1 = Gst.ElementFactory.make("capsfilter", "filter1")
    if not filter1:
        sys.stderr.write(" Unable to get the caps filter1 \n")
    filter1.set_property("caps", caps1)
    print("Creating tiler \n ")
    tiler = Gst.ElementFactory.make("nvmultistreamtiler", "nvtiler")
    if not tiler:
        sys.stderr.write(" Unable to create tiler \n")
    print("Creating nvvidconv \n ")
    nvvidconv = Gst.ElementFactory.make("nvvideoconvert", "convertor")
    if not nvvidconv:
        sys.stderr.write(" Unable to create nvvidconv \n")
    print("Creating nvosd \n ")
    nvosd = Gst.ElementFactory.make("nvdsosd", "onscreendisplay")
    if not nvosd:
        sys.stderr.write(" Unable to create nvosd \n")

    print("Creating Fakesink \n")
    sink = Gst.ElementFactory.make("fakesink", "fakesink")
    if not sink:
        sys.stderr.write(" Unable to create fake sink \n")

    if is_live:
        print("Atleast one of the sources is live")
        streammux.set_property("live-source", 1)

    streammux.set_property('width', 1920)
    streammux.set_property('height', 1080)
    streammux.set_property('batch-size', number_sources)
    streammux.set_property('batched-push-timeout', MUXER_BATCH_TIMEOUT_USEC)
    pgie.set_property('config-file-path', "dstest_imagedata_cupy_config_ds7.1.txt")
    pgie_batch_size = pgie.get_property("batch-size")
    if (pgie_batch_size != number_sources):
        print("WARNING: Overriding infer-config batch-size", pgie_batch_size, " with number of sources ",
              number_sources, " \n")
        pgie.set_property("batch-size", number_sources)
    tiler_rows = int(math.sqrt(number_sources))
    tiler_columns = int(math.ceil((1.0 * number_sources) / tiler_rows))
    tiler.set_property("rows", tiler_rows)
    tiler.set_property("columns", tiler_columns)
    tiler.set_property("width", TILED_OUTPUT_WIDTH)
    tiler.set_property("height", TILED_OUTPUT_HEIGHT)

    sink.set_property("sync", 0)
    sink.set_property("qos", 0)

    print("Adding elements to Pipeline \n")
    pipeline.add(pgie)
    pipeline.add(tiler)
    pipeline.add(nvvidconv)
    pipeline.add(filter1)
    pipeline.add(nvvidconv1)
    pipeline.add(nvosd)
    pipeline.add(sink)

    print("Linking elements in the Pipeline \n")
    streammux.link(pgie)
    pgie.link(nvvidconv1)
    nvvidconv1.link(filter1)
    filter1.link(tiler)
    tiler.link(nvvidconv)
    nvvidconv.link(nvosd)
    nvosd.link(sink)

    # create an event loop and feed gstreamer bus mesages to it
    loop = GLib.MainLoop()
    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect("message", bus_call, loop)

    tiler_sink_pad = tiler.get_static_pad("sink")
    if not tiler_sink_pad:
        sys.stderr.write(" Unable to get src pad \n")
    else:
        tiler_sink_pad.add_probe(Gst.PadProbeType.BUFFER, tiler_sink_pad_buffer_probe, 0)
        # perf callback function to print fps every 5 sec
        GLib.timeout_add(5000, perf_data.perf_print_callback)

    # List the sources
    print("Now playing...")
    for i, source in enumerate(args):
        print(i, ": ", source)

    print("Starting pipeline \n")
    # start play back and listed to events		
    pipeline.set_state(Gst.State.PLAYING)
    try:
        loop.run()
    except:
        pass
    # cleanup
    print("Exiting app\n")
    pipeline.set_state(Gst.State.NULL)

def parse_args():
    parser = argparse.ArgumentParser(prog="deepstream_imagedata-multistream_cupy.py", 
                description="deepstream-imagedata-multistream-cupy takes multiple URI streams as input" \
                    " and retrieves the image buffer from GPU as a cupy array for in-place modification")
    parser.add_argument(
        "-i",
        "--input",
        help="Path to input streams",
        nargs="+",
        metavar="URIs",
        default=["a"],
        required=True,
    )

    args = parser.parse_args()
    stream_paths = args.input
    return stream_paths


if __name__ == '__main__':
    stream_paths = parse_args()
    sys.exit(main(stream_paths))
