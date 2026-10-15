// Register only our background pose receiver; never opens a game or camera.
#define main unused_probe_main
#include "../../pc-input/probe_openvr.cpp"
#undef main
int main(int argc,char** argv){
 if(argc!=4)return 2;
 auto lib=LoadLibraryA(argv[1]);if(!lib)return 3;
 auto init=symbol<uint32_t(*)(EVRInitError*,EVRApplicationType,const char*)>(lib,"VR_InitInternal2");
 auto iface=symbol<void*(*)(const char*,EVRInitError*)>(lib,"VR_GetGenericInterface");
 auto close=symbol<void(*)()>(lib,"VR_ShutdownInternal");if(!init||!iface||!close)return 4;
 EVRInitError error;init(&error,VRApplication_Utility,nullptr);if(error)return 5;
 auto apps=static_cast<IVRApplications*>(iface(IVRApplications_Version,&error));if(!apps){close();return 6;}
 const char* key="local.framehands.nativewireless";std::string cmd=argv[2];EVRApplicationError result=VRApplicationError_None;
 if(cmd=="install"){result=apps->AddApplicationManifest(argv[3],false);if(!result)result=apps->SetApplicationAutoLaunch(key,true);}
 else if(cmd=="launch"){
  auto pid=apps->GetApplicationProcessId(key);HANDLE process=pid?OpenProcess(SYNCHRONIZE,FALSE,pid):nullptr;
  bool running=process&&WaitForSingleObject(process,0)==WAIT_TIMEOUT;if(process)CloseHandle(process);
  if(!running)result=apps->LaunchApplication(key);
 }
 else if(cmd=="remove"){result=apps->SetApplicationAutoLaunch(key,false);if(!result)result=apps->RemoveApplicationManifest(argv[3]);}
 else if(cmd!="inspect"){close();return 7;}
 std::printf("result=%d installed=%d autolaunch=%d pid=%u\n",int(result),apps->IsApplicationInstalled(key),apps->GetApplicationAutoLaunch(key),apps->GetApplicationProcessId(key));
 close();return result?8:0;
}
