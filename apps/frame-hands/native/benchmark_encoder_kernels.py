"""Guarded encoder-only FP32 kernel A/B; no camera, replay, viewer or model edits.

Hypothesis: Winograd transform/dispatch costs may outweigh arithmetic savings
for the encoder's small feature maps. Compare default against Winograd disabled;
leave precision, image size, learned weights and network topology unchanged.
"""
import argparse,hashlib,json,platform,time
from pathlib import Path
import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--models',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--iterations',type=int,default=80);a=p.parse_args()
    if not 20<=a.iterations<=200:p.error('iterations must be20..200')
    if a.output.exists():raise FileExistsError(a.output)
    from workflow_guard import native_workflow
    with native_workflow():
        import ncnn
        if ncnn.__version__!='1.0.20260526':raise ValueError('Reinspect kernel selection for changed ncnn version')
        if ncnn.get_gpu_count()<1:raise RuntimeError('No Vulkan GPU; no CPU fallback')
        manifest=json.loads((a.models/'manifest.json').read_text())
        required=['backbone.ncnn.param','backbone.ncnn.bin']+[f'backbone.{kind}-{i}.bin' for kind in ('input','output') for i in range(4)]
        for name in required:
            if hashlib.sha256((a.models/name).read_bytes()).hexdigest()!=manifest['sha256'][name]:raise ValueError('Changed fixture '+name)
        spec=manifest['blocks']['backbone'];inputs=[];expected=[]
        for i in range(4):
            inputs.append(np.fromfile(a.models/f'backbone.input-{i}.bin',dtype='<f4').reshape(spec['input_shape']))
            expected.append(np.fromfile(a.models/f'backbone.output-{i}.bin',dtype='<f4').reshape(spec['output_shape']))
        nets={};options={};errors={};times={name:[] for name in ('default','no_winograd')}
        def run(name,x):
            with nets[name].create_extractor() as ex:
                if ex.input('in0',ncnn.Mat(x[0]).clone()):raise RuntimeError('input failed')
                status,y=ex.extract('out0')
                if status:raise RuntimeError('inference failed')
                result=np.array(y).reshape(spec['output_shape']).copy()
                if not np.isfinite(result).all():raise ValueError('Nonfinite encoder output')
                return result
        try:
            for name in times:
                net=ncnn.Net();net.opt.num_threads=1;net.opt.use_vulkan_compute=True
                for flag in ('use_fp16_storage','use_fp16_packed','use_fp16_arithmetic','use_bf16_storage'):setattr(net.opt,flag,False)
                net.opt.use_winograd_convolution=name=='default'
                options[name]={flag:getattr(net.opt,flag) for flag in ('use_winograd_convolution','use_sgemm_convolution','use_fp16_storage','use_fp16_packed','use_fp16_arithmetic','use_bf16_storage','use_vulkan_compute','num_threads')}
                if net.load_param(str(a.models/'backbone.ncnn.param')) or net.load_model(str(a.models/'backbone.ncnn.bin')):raise RuntimeError('Model load failed')
                nets[name]=net
                errors[name]=[float(np.abs(run(name,x)-y).max()) for x,y in zip(inputs,expected)]
                if max(errors[name])>1e-4:raise ValueError('FP32 kernel parity failed before timing')
                for i in range(10):run(name,inputs[i%4])
            raw=[]
            # Alternate order every pair to reduce drift bias; identical saved
            # input for both arms within each pair. Compilation excluded.
            for i in range(a.iterations):
                order=('default','no_winograd') if i%2==0 else ('no_winograd','default')
                row={}
                for name in order:
                    start=time.perf_counter();run(name,inputs[i%4]);elapsed=(time.perf_counter()-start)*1000
                    times[name].append(elapsed);row[name]=elapsed
                raw.append(row)
            result=dict(scope=__doc__,platform=platform.platform(),ncnn=ncnn.__version__,options=options,iterations=a.iterations,max_absolute_errors=errors,ms_p50_p95_max={k:np.percentile(v,[50,95,100]).tolist() for k,v in times.items()},paired_default_minus_candidate_ms_p50_p95=np.percentile(np.asarray(times['default'])-times['no_winograd'],[50,95]).tolist(),samples=raw,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),model_manifest_sha256=hashlib.sha256((a.models/'manifest.json').read_bytes()).hexdigest())
            a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='samples'},indent=2))
        finally:
            for net in nets.values():net.clear()


if __name__=='__main__':main()
