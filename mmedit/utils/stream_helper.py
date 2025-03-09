# Copyright 2020 InterDigital Communications, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import struct
from pathlib import Path

import torch
# from torch.nn.modules.utils import consume_prefix_in_state_dict_if_present


def get_padding_size(height, width, p=64):
    new_h = (height + p - 1) // p * p
    new_w = (width + p - 1) // p * p
    # padding_left = (new_w - width) // 2
    padding_left = 0
    padding_right = new_w - width - padding_left
    # padding_top = (new_h - height) // 2
    padding_top = 0
    padding_bottom = new_h - height - padding_top
    return padding_left, padding_right, padding_top, padding_bottom


def get_downsampled_shape(height, width, p):
    new_h = (height + p - 1) // p * p
    new_w = (width + p - 1) // p * p
    return int(new_h / p + 0.5), int(new_w / p + 0.5)


def get_state_dict(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location=torch.device('cpu'))
    if "state_dict" in ckpt:
        ckpt = ckpt['state_dict']
    elif "net" in ckpt:
        ckpt = ckpt["net"]
    # for key in list(ckpt.keys()):
    #     ckpt[key.replace('generator.','')]=ckpt[key]
    ckpt_new={}
    for key in list(ckpt.keys()):
        ckpt_new[key.replace('generator.','')]=ckpt[key]
    return ckpt_new


def filesize(filepath: str) -> int:
    if not Path(filepath).is_file():
        raise ValueError(f'Invalid file "{filepath}".')
    return Path(filepath).stat().st_size


def write_uints(fd, values, fmt=">{:d}I"):
    fd.write(struct.pack(fmt.format(len(values)), *values))


def write_uchars(fd, values, fmt=">{:d}B"):
    fd.write(struct.pack(fmt.format(len(values)), *values))


def read_uints(fd, n, fmt=">{:d}I"):
    sz = struct.calcsize("I")
    return struct.unpack(fmt.format(n), fd.read(n * sz))


def read_uchars(fd, n, fmt=">{:d}B"):
    sz = struct.calcsize("B")
    return struct.unpack(fmt.format(n), fd.read(n * sz))


def write_bytes(fd, values, fmt=">{:d}s"):
    if len(values) == 0:
        return
    fd.write(struct.pack(fmt.format(len(values)), values))


def read_bytes(fd, n, fmt=">{:d}s"):
    sz = struct.calcsize("s")
    return struct.unpack(fmt.format(n), fd.read(n * sz))[0]


def write_ushorts(fd, values, fmt=">{:d}H"):
    fd.write(struct.pack(fmt.format(len(values)), *values))


def read_ushorts(fd, n, fmt=">{:d}H"):
    sz = struct.calcsize("H")
    return struct.unpack(fmt.format(n), fd.read(n * sz))


def encode_i(height, width, q_index, bit_stream, output):
    with Path(output).open("wb") as f:
        write_uints(f, (height, width))
        write_uchars(f, ((q_index << 2),))  # 6-bit index
        write_bytes(f, bit_stream)


def decode_i(inputpath):
    with Path(inputpath).open("rb") as f:
        header = read_uints(f, 2)
        height = header[0]
        width = header[1]
        flag = read_uchars(f, 1)[0]
        q_index = ((flag & 0xfc) >> 2)
        stream_length = os.path.getsize(inputpath) - 9

        bit_stream = read_bytes(f, stream_length)

    return height, width, q_index, bit_stream


def encode_p(string, q_index, fa_idx, output):
    assert fa_idx < 4
    with Path(output).open("wb") as f:
        write_uchars(f, ((q_index << 2) + fa_idx,))
        write_bytes(f, string)


def decode_p(inputpath):
    with Path(inputpath).open("rb") as f:
        flag = read_uchars(f, 1)[0]
        q_index = ((flag & 0xfc) >> 2)
        fa_idx = (flag & 0x03)

        string_length = os.path.getsize(inputpath) - 1
        string = read_bytes(f, string_length)

    return q_index, fa_idx, string
