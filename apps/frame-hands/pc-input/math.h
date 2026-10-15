#pragma once
#include <algorithm>
#include <array>
#include <cmath>
namespace fhs {
struct V {
    double x=0,y=0,z=0;
    V operator+(V b)const{return{x+b.x,y+b.y,z+b.z};}
    V operator-(V b)const{return{x-b.x,y-b.y,z-b.z};}
    V operator*(double k)const{return{x*k,y*k,z*k};}
    double dot(V b)const{return x*b.x+y*b.y+z*b.z;}
    V cross(V b)const{return{y*b.z-z*b.y,z*b.x-x*b.z,x*b.y-y*b.x};}
    double len()const{return std::sqrt(dot(*this));}
    V unit()const{auto l=len();return l>1e-9?*this*(1/l):V{};}
    bool finite()const{return std::isfinite(x)&&std::isfinite(y)&&std::isfinite(z);}
};
struct Q {
    double w=1,x=0,y=0,z=0;
    Q inv()const{return{w,-x,-y,-z};}
    Q unit()const{double n=std::sqrt(w*w+x*x+y*y+z*z);return n>1e-9?Q{w/n,x/n,y/n,z/n}:Q{};}
    Q operator*(Q b)const{return{w*b.w-x*b.x-y*b.y-z*b.z,w*b.x+x*b.w+y*b.z-z*b.y,w*b.y-x*b.z+y*b.w+z*b.x,w*b.z+x*b.y-y*b.x+z*b.w};}
    V rotate(V v)const{Q q=*this*Q{0,v.x,v.y,v.z}*inv();return{q.x,q.y,q.z};}
};
inline Q swing(V a,V b){
    a=a.unit();b=b.unit();double d=std::clamp(a.dot(b),-1.,1.);
    if(d<-.999999){V axis=a.cross(std::abs(a.x)<.8?V{1,0,0}:V{0,1,0}).unit();return{0,axis.x,axis.y,axis.z};}
    V c=a.cross(b);return Q{1+d,c.x,c.y,c.z}.unit();
}
inline Q basis(V x,V y,V z){
    const double m[3][3]={{x.x,y.x,z.x},{x.y,y.y,z.y},{x.z,y.z,z.z}};
    Q q;double t=m[0][0]+m[1][1]+m[2][2];
    if(t>0){double s=std::sqrt(1+t)*2;q={s/4,(m[2][1]-m[1][2])/s,(m[0][2]-m[2][0])/s,(m[1][0]-m[0][1])/s};}
    else {int i=m[1][1]>m[0][0]?1:0;if(m[2][2]>m[i][i])i=2;int j=(i+1)%3,k=(i+2)%3;
        double s=std::sqrt(1+m[i][i]-m[j][j]-m[k][k])*2;double v[3]{};v[i]=s/4;v[j]=(m[i][j]+m[j][i])/s;v[k]=(m[i][k]+m[k][i])/s;
        q={(m[k][j]-m[j][k])/s,v[0],v[1],v[2]};}
    return q.unit();
}
inline Q palm(const std::array<V,21>& p){V y=(p[9]-p[0]).unit();V x=(p[5]-p[17]).unit();V z=x.cross(y).unit();x=y.cross(z).unit();return basis(x,y,z);}
}
