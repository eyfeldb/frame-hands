"""Guarded inspection of actual ncnn encoder layer capabilities, no capture/XR."""
import argparse,collections,hashlib,json,platform
from pathlib import Path
from workflow_guard import native_workflow


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--models',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    with native_workflow():
        import ncnn
        if ncnn.__version__!='1.0.20260526' or ncnn.get_gpu_count()<1:raise RuntimeError('Unexpected native runtime')
        manifest=json.loads((a.models/'manifest.json').read_text())
        for name in ('backbone.ncnn.param','backbone.ncnn.bin'):
            if hashlib.sha256((a.models/name).read_bytes()).hexdigest()!=manifest['sha256'][name]:raise ValueError('Changed model')
        net=ncnn.Net();net.opt.num_threads=1;net.opt.use_vulkan_compute=True
        net.opt.use_fp16_storage=False;net.opt.use_fp16_packed=False;net.opt.use_fp16_arithmetic=False;net.opt.use_bf16_storage=False
        if net.load_param(str(a.models/'backbone.ncnn.param')) or net.load_model(str(a.models/'backbone.ncnn.bin')):raise RuntimeError('Model load failed')
        rows=[dict(index=i,name=layer.name,type=layer.type,support_vulkan=layer.support_vulkan,support_vulkan_packing=layer.support_vulkan_packing) for i,layer in enumerate(net.layers())]
        result=dict(scope='Loaded encoder layer capability inspection, not GPU timestamps or measured synchronization attribution.',ncnn=ncnn.__version__,platform=platform.platform(),device=ncnn.get_gpu_info().device_name(),layers=rows,operator_counts=dict(collections.Counter(r['type'] for r in rows)),non_vulkan_compute_layers=[r for r in rows if not r['support_vulkan'] and r['type']!='Input'],python_has_vkcompute=hasattr(ncnn,'VkCompute'),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='layers'},indent=2))

if __name__=='__main__':main()
