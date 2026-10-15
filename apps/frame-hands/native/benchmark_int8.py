"""Guarded encoder-only FP32 Vulkan versus calibrated INT8 CPU1/CPU4 benchmark."""
import argparse,hashlib,json,platform,time
from pathlib import Path
import numpy as np
from workflow_guard import native_workflow


def thermal():
    rows={}
    for zone in Path('/sys/class/thermal').glob('thermal_zone*'):
        try:rows[(zone/'type').read_text().strip()]=(zone/'temp').read_text().strip()
        except OSError:pass
    return rows


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('models','encoder','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    with native_workflow():
        import ncnn,torch
        from int8_encoder import Int8Encoder
        if ncnn.__version__!='1.0.20260526' or ncnn.get_gpu_count()<1:raise RuntimeError('Unexpected runtime')
        torch.set_num_threads(1);manifest=json.loads((a.models/'manifest.json').read_text());names=['backbone.ncnn.param','backbone.ncnn.bin']+[f'backbone.{kind}-{i}.bin' for kind in ('input','output') for i in range(4)]
        for name in names:
            if hashlib.sha256((a.models/name).read_bytes()).hexdigest()!=manifest['sha256'][name]:raise ValueError('Changed fixture/model')
        net=ncnn.Net();net.opt.num_threads=1;net.opt.use_vulkan_compute=True;net.opt.use_fp16_storage=False;net.opt.use_fp16_packed=False;net.opt.use_fp16_arithmetic=False;net.opt.use_bf16_storage=False
        if net.load_param(str(a.models/'backbone.ncnn.param')) or net.load_model(str(a.models/'backbone.ncnn.bin')):raise RuntimeError('FP32 load failed')
        def fp32(x):
            with net.create_extractor() as ex:
                if ex.input('in0',ncnn.Mat(x[0].numpy()).clone()):raise RuntimeError('FP32 input failed')
                status,out=ex.extract('out0')
                if status:raise RuntimeError('FP32 inference failed')
                return torch.from_numpy(np.array(out).reshape(1,72,6,6).copy())
        methods={'fp32_vulkan':fp32,'int8_cpu1':Int8Encoder(a.encoder,1),'int8_cpu4':Int8Encoder(a.encoder,4)}
        inputs=[torch.from_numpy(np.fromfile(a.models/f'backbone.input-{i}.bin',dtype='<f4').reshape(1,1,96,96)) for i in range(4)]
        expected=[np.fromfile(a.models/f'backbone.output-{i}.bin',dtype='<f4').reshape(1,72,6,6) for i in range(4)]
        errors={name:[float(abs(method(x).numpy()-ref).max()) for x,ref in zip(inputs,expected)] for name,method in methods.items()}
        for _ in range(5):
            for method in methods.values():method(inputs[0])
        temps=thermal();times={name:[] for name in methods};order=list(methods)
        for i in range(40):
            for name in order if i%2==0 else order[::-1]:
                start=time.perf_counter();out=methods[name](inputs[i%4]);times[name].append((time.perf_counter()-start)*1000)
                if not torch.isfinite(out).all():raise ValueError('Nonfinite output')
        cpu_features=next((line.partition(':')[2].strip() for line in Path('/proc/cpuinfo').read_text().splitlines() if line.startswith('Features')),'unavailable')
        result=dict(scope='Four fixed synthetic tensors; encoder-only wall time. Not full tracking latency, camera/render or wearer acceptance. INT8 errors are feature values, not millimeters.',ncnn=ncnn.__version__,torch=torch.__version__,platform=platform.platform(),mode='Linux saved tensors; headset streaming state unrecorded',cpu_features=cpu_features,thermal_before=temps,thermal_after=thermal(),milliseconds_p50_p95_max={n:np.percentile(t,[50,95,100]).tolist() for n,t in times.items()},feature_max_errors=errors,raw_ms=times,encoder_manifest_sha256=hashlib.sha256((a.encoder/'manifest.json').read_bytes()).hexdigest(),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ('raw_ms','thermal_before','thermal_after')},indent=2))

if __name__=='__main__':main()
