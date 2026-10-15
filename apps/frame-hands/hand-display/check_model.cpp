#include <openvr.h>
#include <cstdio>
#include <chrono>
#include <thread>
#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif
int main(int argc,char**argv){
 if(argc!=3)return 2;
#ifdef _WIN32
 auto lib=LoadLibraryA(argv[1]);auto symbol=[&](const char*n){return reinterpret_cast<void*>(GetProcAddress(lib,n));};
#else
 auto lib=dlopen(argv[1],RTLD_NOW|RTLD_LOCAL);auto symbol=[&](const char*n){return dlsym(lib,n);};
#endif
 if(!lib)return 3;
 auto init=reinterpret_cast<uint32_t(*)(vr::EVRInitError*,vr::EVRApplicationType,const char*)>(symbol("VR_InitInternal2"));
 auto get=reinterpret_cast<void*(*)(const char*,vr::EVRInitError*)>(symbol("VR_GetGenericInterface"));
 auto quit=reinterpret_cast<void(*)()>(symbol("VR_ShutdownInternal"));if(!init||!get||!quit)return 4;
 vr::EVRInitError error{};init(&error,vr::VRApplication_Background,nullptr);if(error)return 5;
 auto models=static_cast<vr::IVRRenderModels*>(get(vr::IVRRenderModels_Version,&error));if(!models){quit();return 6;}
 vr::RenderModel_t* model=nullptr;vr::EVRRenderModelError result=vr::VRRenderModelError_Loading;
 for(int i=0;i<100&&result==vr::VRRenderModelError_Loading;++i){result=models->LoadRenderModel_Async(argv[2],&model);if(result==vr::VRRenderModelError_Loading)std::this_thread::sleep_for(std::chrono::milliseconds(20));}
 std::printf("render_model result=%d vertices=%u triangles=%u\n",int(result),model?model->unVertexCount:0,model?model->unTriangleCount:0);
 if(model)models->FreeRenderModel(model);quit();return result?1:0;
}
