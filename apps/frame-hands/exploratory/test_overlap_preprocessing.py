"""Check diagnostic preprocessing against extracted frozen C++ functions, offline."""
import argparse,ctypes,hashlib,json,subprocess
from pathlib import Path
import numpy as np
from probe_overlap_acquisition import native_clahe,hand_matrix


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    nets=(a.source/'nets.cpp').read_text();assert hashlib.sha256((a.source/'nets.cpp').read_bytes()).hexdigest()=='6406829eb961deae8871403c8c1b770e2014573344838ab2555d166c475ade51'
    body=nets[nets.index('constexpr int kPalmSize'):nets.index('// A crop as the models')]
    roi=nets[nets.index('Roi roi_from_points'):nets.index('Roi Landmarks::next_roi')]
    harness='#define _USE_MATH_DEFINES\n#include <cmath>\n#include <algorithm>\n#include <cstdint>\n#include <vector>\n#include "'+(a.source/'geom.h').resolve().as_posix()+'"\nstruct Roi { V2 center{}; double size=0, rotation=0; };\n'+body+roi+'''\nextern "C" __declspec(dllexport) void reference_clahe(uint8_t *image,int n){clahe(image,n,2.0);}
extern "C" __declspec(dllexport) void reference_roi(const double *points,float *matrix){V2 p[21];for(int i=0;i<21;i++)p[i]={points[i*2],points[i*2+1]};auto r=roi_from_points(p);crop_matrix(r.center,r.size,r.rotation,224,matrix);}
'''
    (out/'reference.cpp').write_text(harness)
    tool=Path('C:/Program Files (x86)/Microsoft Visual Studio/2022/BuildTools/VC/Tools/MSVC/14.44.35207');sdk=Path('C:/Program Files (x86)/Windows Kits/10');v='10.0.26100.0'
    includes=[tool/'include',sdk/'Include'/v/'ucrt',sdk/'Include'/v/'shared',sdk/'Include'/v/'um']
    command=[str(tool/'bin/Hostx64/x64/cl.exe'),'/nologo','/std:c++17','/EHsc','/O2','/MT','/LD','/DNOMINMAX']+['/I'+str(x) for x in includes]+[str(out/'reference.cpp'),'/Fe:'+str(out/'reference.dll'),'/link']+['/LIBPATH:'+str(x) for x in (tool/'lib/x64',sdk/'Lib'/v/'ucrt/x64',sdk/'Lib'/v/'um/x64')]
    subprocess.run(command,cwd=out,check=True)
    dll=ctypes.CDLL(str(out/'reference.dll'));bytep=ctypes.POINTER(ctypes.c_uint8);dp=ctypes.POINTER(ctypes.c_double);fp=ctypes.POINTER(ctypes.c_float)
    dll.reference_clahe.argtypes=[bytep,ctypes.c_int];dll.reference_clahe.restype=None;dll.reference_roi.argtypes=[dp,fp];dll.reference_roi.restype=None
    rng=np.random.default_rng(94271);images=[np.full((192,192),x,np.uint8) for x in (0,1,127,254,255)]+[rng.integers(0,256,(192,192),dtype=np.uint8) for _ in range(20)]
    pixel_differences=[]
    for image in images:
        native=image.copy();dll.reference_clahe(native.ctypes.data_as(bytep),192);python=native_clahe(image);pixel_differences.append(int(np.max(abs(native.astype(int)-python.astype(int)))))
    geometry=[]
    for _ in range(100):
        points=np.ascontiguousarray(rng.normal(400,100,(21,2)));matrix=np.empty((2,3),np.float32);dll.reference_roi(points.ctypes.data_as(dp),matrix.ctypes.data_as(fp));mine,_=hand_matrix(points);geometry.append(float(np.max(abs(mine-matrix))))
    # C++ crop_matrix rounds its linear coefficients before computing translation.
    assert max(pixel_differences)==0
    assert max(geometry)<.0001
    result=dict(clahe_cases=len(images),max_pixel_difference=max(pixel_differences),roi_cases=len(geometry),max_matrix_difference=max(geometry),native_source_sha256=hashlib.sha256((a.source/'nets.cpp').read_bytes()).hexdigest(),dll_sha256=hashlib.sha256((out/'reference.dll').read_bytes()).hexdigest(),compiler='MSVC14.44.35207 /O2 /fp:precise default; SDK10.0.26100.0',scope='Native CLAHE and ROI formulas checked only. OpenCV/ncnn affine sampler and x64/ARM inference parity remain unproven.')
    (out/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
