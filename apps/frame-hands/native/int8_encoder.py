"""Isolated CPU ncnn INT8 encoder adapter; remaining learned model stays unchanged."""
import hashlib,json
from pathlib import Path
import numpy as np
import torch
from torch import nn

class Int8Encoder(nn.Module):
    def __init__(self,directory,threads=1):
        super().__init__();import ncnn
        if ncnn.__version__!='1.0.20260526':raise ValueError('Unpinned ncnn runtime')
        directory=Path(directory);meta=json.loads((directory/'manifest.json').read_text())
        for name in ('backbone.ncnn.param','backbone.ncnn.bin'):
            if hashlib.sha256((directory/name).read_bytes()).hexdigest()!=meta['sha256'][name]:raise ValueError('Changed INT8 encoder')
        self.ncnn=ncnn;self.net=ncnn.Net();self.net.opt.num_threads=threads;self.net.opt.use_vulkan_compute=False;self.net.opt.use_int8_inference=True
        self.net.opt.use_fp16_storage=False;self.net.opt.use_fp16_packed=False;self.net.opt.use_fp16_arithmetic=False;self.net.opt.use_bf16_storage=False
        if self.net.load_param(str(directory/'backbone.ncnn.param')) or self.net.load_model(str(directory/'backbone.ncnn.bin')):raise RuntimeError('INT8 model load failed')
        self.calls=0
    def forward(self,x):
        if x.dtype!=torch.float32 or x.ndim!=4 or tuple(x.shape[1:])!=(1,96,96) or not 1<=len(x)<=16:raise ValueError('Unexpected encoder input')
        rows=[]
        for sample in x.detach().cpu().numpy():
            with self.net.create_extractor() as ex:
                if ex.input('in0',self.ncnn.Mat(np.ascontiguousarray(sample)).clone()):raise RuntimeError('INT8 input failed')
                status,out=ex.extract('out0')
                if status:raise RuntimeError('INT8 extraction failed')
                value=np.array(out).reshape(72,6,6).copy()
                if not np.isfinite(value).all():raise ValueError('Nonfinite INT8 feature')
                rows.append(value)
        self.calls+=len(rows)
        return torch.from_numpy(np.stack(rows)).to(x.device)
