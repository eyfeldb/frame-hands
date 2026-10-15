#pragma once
#include "../pc-input/retarget_chain.h"
#include "../pc-input/model_protocol.h"
#include <fstream>
#include <vector>
#include <stdexcept>
namespace fhdisplay {
using fhs::V;using fhs::Q;using fhs::Transform;
struct Matrix {float m[16]{};}; // row-major
inline Matrix matrix(Transform t){
    auto x=t.q.rotate({1,0,0}),y=t.q.rotate({0,1,0}),z=t.q.rotate({0,0,1});
    return {{float(x.x),float(y.x),float(z.x),float(t.p.x),float(x.y),float(y.y),float(z.y),float(t.p.y),float(x.z),float(y.z),float(z.z),float(t.p.z),0,0,0,1}};
}
inline Matrix multiply(const Matrix&a,const Matrix&b){Matrix c;for(int i=0;i<4;++i)for(int j=0;j<4;++j)for(int k=0;k<4;++k)c.m[4*i+j]+=a.m[4*i+k]*b.m[4*k+j];return c;}
inline V apply(const Matrix&m,V v,bool normal=false){return {m.m[0]*v.x+m.m[1]*v.y+m.m[2]*v.z+(normal?0:m.m[3]),m.m[4]*v.x+m.m[5]*v.y+m.m[6]*v.z+(normal?0:m.m[7]),m.m[8]*v.x+m.m[9]*v.y+m.m[10]*v.z+(normal?0:m.m[11])};}
struct Vertex {float p[3],n[3];uint32_t bone[4];float weight[4];};
struct DrawVertex {float p[3],n[3];};
static_assert(sizeof(Vertex)==56&&sizeof(DrawVertex)==24);
struct Mesh {
    uint32_t side=0;std::array<Matrix,31> inverse{};std::vector<Vertex> vertices;std::vector<uint32_t> indices;
    template<class Path>void load(const Path& path,uint32_t expected){
        std::ifstream f(path,std::ios::binary);char magic[8]{};uint32_t nv=0,ni=0,nb=0;
        f.read(magic,8);f.read(reinterpret_cast<char*>(&side),4);f.read(reinterpret_cast<char*>(&nv),4);f.read(reinterpret_cast<char*>(&ni),4);f.read(reinterpret_cast<char*>(&nb),4);
        if(!f||std::memcmp(magic,"FHMSH01",8)||side!=expected||!nv||nv>65000||!ni||ni>300000||ni%3||nb!=31)throw std::runtime_error("Invalid glove asset header");
        f.read(reinterpret_cast<char*>(inverse.data()),sizeof inverse);vertices.resize(nv);indices.resize(ni);
        f.read(reinterpret_cast<char*>(vertices.data()),nv*sizeof(Vertex));f.read(reinterpret_cast<char*>(indices.data()),ni*4);
        if(!f||f.peek()!=EOF)throw std::runtime_error("Invalid glove asset length");
        for(auto i:indices)if(i>=nv)throw std::runtime_error("Invalid glove index");
        for(const auto&m:inverse)for(float v:m.m)if(!std::isfinite(v))throw std::runtime_error("Nonfinite bind");
        for(const auto&v:vertices){double sum=0;for(int k=0;k<3;++k)if(!std::isfinite(v.p[k])||std::abs(v.p[k])>1||!std::isfinite(v.n[k]))throw std::runtime_error("Invalid vertex");for(int k=0;k<4;++k){if(v.bone[k]>=31||!std::isfinite(v.weight[k])||v.weight[k]<0)throw std::runtime_error("Invalid weight");sum+=v.weight[k];}if(std::abs(sum-1)>.001)throw std::runtime_error("Weight sum");}
    }
    std::vector<DrawVertex> skin(const fhs::Skeleton& pose)const{
        auto bones=fhs::forward(pose.bones);std::array<Matrix,31> transforms;
        Matrix world=matrix({pose.position,pose.rotation});
        for(int b=0;b<31;++b)transforms[b]=multiply(multiply(world,matrix(bones[b])),inverse[b]);
        std::vector<DrawVertex> out(vertices.size());
        for(size_t i=0;i<vertices.size();++i){auto& v=vertices[i];V p{},n{};
            for(int k=0;k<4;++k)if(v.weight[k]){auto& m=transforms[v.bone[k]];p=p+apply(m,{v.p[0],v.p[1],v.p[2]})*v.weight[k];n=n+apply(m,{v.n[0],v.n[1],v.n[2]},true)*v.weight[k];}
            n=n.unit();out[i]={{float(p.x),float(p.y),float(p.z)},{float(n.x),float(n.y),float(n.z)}};
        }return out;
    }
};
}
