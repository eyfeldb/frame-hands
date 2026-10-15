#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <ctime>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <dirent.h>

namespace fhg {
constexpr double degrees=180.0/3.14159265358979323846;
struct Vec {
    double x=0,y=0,z=0;
    Vec operator+(Vec b)const{return{x+b.x,y+b.y,z+b.z};}
    Vec operator-(Vec b)const{return{x-b.x,y-b.y,z-b.z};}
    Vec operator*(double s)const{return{x*s,y*s,z*s};}
    double dot(Vec b)const{return x*b.x+y*b.y+z*b.z;}
    double length()const{return std::sqrt(dot(*this));}
    bool finite()const{return std::isfinite(x)&&std::isfinite(y)&&std::isfinite(z);}
    Vec unit()const{double n=length();return n>1e-6?*this*(1/n):Vec{};}
};
using Angles=std::array<double,2>;
inline Angles angles(Vec v){v=v.unit();return{std::atan2(-v.x,-v.z)*degrees,std::asin(std::clamp(v.y,-1.,1.))*degrees};}
inline Vec direction(Angles a){double y=a[0]/degrees,p=a[1]/degrees;return{-std::sin(y)*std::cos(p),std::sin(p),-std::cos(y)*std::cos(p)};}
inline Angles fixation_center(const std::vector<Angles>& samples){
    if(samples.empty())return {};
    Angles center{};
    for(int k=0;k<2;++k){std::vector<double> values;for(auto a:samples)values.push_back(a[k]);std::sort(values.begin(),values.end());size_t n=values.size();center[k]=(values[n/2]+values[(n-1)/2])/2;}
    return center;
}
inline double fixation_spread(const std::vector<Angles>& samples){
    if(samples.empty())return 180;
    auto middle=direction(fixation_center(samples));std::vector<double> errors;
    for(auto a:samples)errors.push_back(std::acos(std::clamp(direction(a).dot(middle),-1.,1.))*degrees);
    std::sort(errors.begin(),errors.end());return errors[size_t((errors.size()-1)*.9)];
}
template<class M> Vec rotate(const M& m,Vec v){return{m.m[0][0]*v.x+m.m[0][1]*v.y+m.m[0][2]*v.z,m.m[1][0]*v.x+m.m[1][1]*v.y+m.m[1][2]*v.z,m.m[2][0]*v.x+m.m[2][1]*v.y+m.m[2][2]*v.z};}
template<class M> Vec inverse_rotate(const M& m,Vec v){return{m.m[0][0]*v.x+m.m[1][0]*v.y+m.m[2][0]*v.z,m.m[0][1]*v.x+m.m[1][1]*v.y+m.m[2][1]*v.z,m.m[0][2]*v.x+m.m[1][2]*v.y+m.m[2][2]*v.z};}
template<class M> Vec inverse_point(const M& m,Vec v){return inverse_rotate(m,v-Vec{m.m[0][3],m.m[1][3],m.m[2][3]});}
inline std::array<double,6> features(Angles a){double x=a[0]/20,y=a[1]/20;return{1,x,y,x*x,x*y,y*y};}
// Bind profiles to this boot and actual eye-tracker process lifetime. No Steam config.
inline std::string eye_session(){
    std::string boot;std::ifstream("/proc/sys/kernel/random/boot_id")>>boot;
    DIR* directory=opendir("/proc");if(!directory)return {};
    std::string result;
    while(auto e=readdir(directory)){
        if(e->d_name[0]<'0'||e->d_name[0]>'9')continue;
        std::string base=std::string("/proc/")+e->d_name,name,line;
        std::getline(std::ifstream(base+"/comm"),name);
        if(name!="eyetracking")continue;
        std::getline(std::ifstream(base+"/stat"),line);auto end=line.rfind(')');if(end==std::string::npos)continue;
        std::istringstream fields(line.substr(end+2));std::string token;
        for(int field=3;field<=22;++field)fields>>token;
        if(fields&&!boot.empty()){if(!result.empty()){result.clear();break;}result=boot+":"+e->d_name+":"+token;}
    }
    closedir(directory);return result;
}
struct Calibration {
    int terms=0;long long created=0;std::string session;
    double minimum[2]{},maximum[2]{},coefficient[2][6]{};
    bool load(const std::string& path,const std::string& current,long long now,bool preview=false){
        std::ifstream in(path);std::string magic,extra;in>>magic>>session>>created>>terms;
        for(int k=0;k<2;++k)in>>minimum[k]>>maximum[k];
        for(auto& row:coefficient)for(double& v:row)in>>v;
        if(!in||magic!=(preview?"FHGAZE_PREVIEW1":"FHGAZE1")||session.empty()||current.empty()||session!=current||created>now||now-created>900||(terms!=1&&terms!=3&&terms!=6))return false;
        if(in>>extra)return false;
        for(int k=0;k<2;++k){
            if(!std::isfinite(minimum[k])||!std::isfinite(maximum[k])||minimum[k]>=maximum[k]||minimum[k]<-60||maximum[k]>60)return false;
            for(int j=0;j<6;++j)if(!std::isfinite(coefficient[k][j])||std::abs(coefficient[k][j])>60)return false;
        }
        return true;
    }
    bool apply(Angles raw,Angles& corrected)const{
        if(terms!=1&&terms!=3&&terms!=6)return false;
        for(int k=0;k<2;++k)if(!std::isfinite(raw[k])||raw[k]<minimum[k]-2||raw[k]>maximum[k]+2)return false;
        auto f=features(raw);corrected=raw;
        for(int k=0;k<2;++k){double delta=0;for(int j=0;j<terms;++j)delta+=coefficient[k][j]*f[j];if(!std::isfinite(delta)||std::abs(delta)>12)return false;corrected[k]+=delta;}
        return std::abs(corrected[0])<80&&std::abs(corrected[1])<80;
    }
};
}
