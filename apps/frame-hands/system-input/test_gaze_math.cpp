#include "gaze_math.h"
#include <cassert>
#include <cstdio>
#include <limits>
#include <unistd.h>
int main(int argc,char** argv){
    if(argc==4){fhg::Calibration c;if(!c.load(argv[1],argv[2],std::stoll(argv[3])))return 2;fhg::Angles out{};if(!c.apply({4,-3},out))return 3;std::printf("[%.9f,%.9f]\n",out[0],out[1]);return 0;}
    for(double yaw=-70;yaw<=70;yaw+=7)for(double pitch=-60;pitch<=60;pitch+=6){
        fhg::Angles a{yaw,pitch};auto b=fhg::angles(fhg::direction(a));assert(std::abs(a[0]-b[0])<1e-10&&std::abs(a[1]-b[1])<1e-10);
    }
    assert(fhg::fixation_spread({})==180);
    assert(fhg::fixation_spread({{0,0},{0,0},{0,0}})<.00001);
    assert(std::abs(fhg::fixation_spread({{-2,0},{2,0},{-2,0},{2,0}})-2)<.00001);
    // Known 90-degree world yaw: head-local forward maps to world left.
    struct Matrix{double m[3][4];} m{{{0,0,1,2},{0,1,0,3},{-1,0,0,4}}};
    auto direction=fhg::rotate(m,{0,0,-1});assert(direction.x==-1&&direction.z==0);
    auto local=fhg::inverse_point(m,{1,3,4});assert(local.x==0&&local.y==0&&local.z==-1);
    std::string path="/tmp/fh-gaze-test-"+std::to_string(getpid());
    {std::ofstream out(path);out<<"FHGAZE1 session 1000 1 -20 20 -15 15 3 0 0 0 0 0 -2 0 0 0 0 0\n";}
    fhg::Calibration c;assert(c.load(path,"session",1010));fhg::Angles corrected{};
    assert(c.apply({4,-3},corrected)&&corrected[0]==7&&corrected[1]==-5);
    assert(!c.apply({23,0},corrected));assert(!c.apply({0,std::numeric_limits<double>::quiet_NaN()},corrected));
    assert(!c.load(path,"other-session",1010));assert(!c.load(path,"session",1901));assert(!c.load(path,"session",999));
    {std::ofstream out(path,std::ios::app);out<<"trailing-invalid\n";}assert(!c.load(path,"session",1010));unlink(path.c_str());
    {std::ofstream out(path);out<<"FHGAZE_PREVIEW1 session 1000 1 -20 20 -15 15 3 0 0 0 0 0 -2 0 0 0 0 0\n";}
    assert(!c.load(path,"session",1010));assert(c.load(path,"session",1010,true));unlink(path.c_str());
    std::puts("PASS gaze coordinate round trips, rotated head frame, profile age/session/format checks and calibration bounds");
}
