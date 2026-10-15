#define main unused_probe_main
#include "../pc-input/probe_openvr.cpp"
#undef main
int main(int argc,char** argv){
    if(argc!=4)return 2;
    auto lib=LoadLibraryA(argv[1]);if(!lib)return 3;
    auto init=symbol<uint32_t(*)(EVRInitError*,EVRApplicationType,const char*)>(lib,"VR_InitInternal2");
    auto iface=symbol<void*(*)(const char*,EVRInitError*)>(lib,"VR_GetGenericInterface");
    auto close=symbol<void(*)()>(lib,"VR_ShutdownInternal");
    if(!init||!iface||!close)return 4;
    EVRInitError error;init(&error,VRApplication_Utility,nullptr);
    if(error){std::printf("init_error=%d\n",int(error));return 5;}
    auto apps=static_cast<IVRApplications*>(iface(IVRApplications_Version,&error));
    if(!apps){close();return 6;}
    const std::string command=argv[2];EVRApplicationError result=VRApplicationError_None;
    if(command=="register")result=apps->AddApplicationManifest(argv[3],false);
    else if(command=="remove")result=apps->RemoveApplicationManifest(argv[3]);
    else if(command!="inspect"){close();return 7;}
    bool installed=apps->IsApplicationInstalled("local.framehands.piano");
    char name[256]{};EVRApplicationError property;
    apps->GetApplicationPropertyString("local.framehands.piano",VRApplicationProperty_Name_String,name,sizeof(name),&property);
    std::printf("result=%d installed=%d name=%s\n",int(result),installed,name);
    close();return result==VRApplicationError_None?0:8;
}
