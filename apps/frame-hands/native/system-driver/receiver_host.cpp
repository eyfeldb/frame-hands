// SteamVR-owned receiver lifetime. Never starts cameras or games.
#define main unused_probe_main
#include "../../pc-input/probe_openvr.cpp"
#undef main
#include <fstream>
#include <filesystem>
static std::string quote(const char* value){
 std::string out="\"";unsigned slashes=0;
 for(const char* p=value;*p;++p){if(*p=='\\'){++slashes;continue;}
  if(*p=='\"')out.append(slashes*2+1,'\\');else out.append(slashes,'\\');
  slashes=0;out+=*p;}
 out.append(slashes*2,'\\');return out+'"';
}
int main(int argc,char**argv){
 if(argc!=7)return 2;
 HANDLE mutex=CreateMutexA(nullptr,TRUE,"Local\\FrameHandsNativeWirelessHost");
 if(!mutex||GetLastError()==ERROR_ALREADY_EXISTS){if(mutex)CloseHandle(mutex);return 0;}
 auto lib=LoadLibraryA(argv[1]);if(!lib)return 3;
 auto init=symbol<uint32_t(*)(EVRInitError*,EVRApplicationType,const char*)>(lib,"VR_InitInternal2");
 auto iface=symbol<void*(*)(const char*,EVRInitError*)>(lib,"VR_GetGenericInterface");
 auto shutdown=symbol<void(*)()>(lib,"VR_ShutdownInternal");if(!init||!iface||!shutdown)return 4;
 EVRInitError error;init(&error,VRApplication_Background,nullptr);if(error)return 5;
 auto system=static_cast<IVRSystem*>(iface(IVRSystem_Version,&error));
 auto apps=static_cast<IVRApplications*>(iface(IVRApplications_Version,&error));if(!system||!apps)return 6;
 auto identify=apps->IdentifyApplication(GetCurrentProcessId(),"local.framehands.nativewireless");if(identify)return 7;
 // Only lifecycle/error text. No camera frames, poses, key material or command lines.
 std::error_code ec;if(std::filesystem::file_size(argv[6],ec)>131072&&!ec){std::filesystem::remove(std::string(argv[6])+".previous",ec);std::filesystem::rename(argv[6],std::string(argv[6])+".previous",ec);}
 auto log=[&](const std::string& msg){std::ofstream f(argv[6],std::ios::app);f<<GetTickCount64()<<" "<<msg<<'\n';};
 std::string cmd=quote(argv[2])+" "+quote(argv[3])+" --connection "+quote(argv[4])+" --stop-file "+quote(argv[5]);
 HANDLE child=nullptr,job=nullptr;ULONGLONG retry=0;bool quit=false;
 log("receiver host started pid="+std::to_string(GetCurrentProcessId()));
 while(!quit&&GetFileAttributesA(argv[5])==INVALID_FILE_ATTRIBUTES){
  VREvent_t event{};while(system->PollNextEvent(&event,sizeof event))if(event.eventType==VREvent_Quit){system->AcknowledgeQuit_Exiting();quit=true;}
  if(quit)break;
  if(child&&WaitForSingleObject(child,0)==WAIT_OBJECT_0){DWORD code=0;GetExitCodeProcess(child,&code);log("receiver exited code="+std::to_string(code));CloseHandle(child);child=nullptr;CloseHandle(job);job=nullptr;retry=GetTickCount64()+2000;}
  if(!child&&GetTickCount64()>=retry){
   SECURITY_ATTRIBUTES sa{sizeof(sa),nullptr,TRUE};HANDLE sink=CreateFileA("NUL",GENERIC_READ|GENERIC_WRITE,FILE_SHARE_READ|FILE_SHARE_WRITE,&sa,OPEN_EXISTING,0,nullptr);
   HANDLE errors=CreateFileA(argv[6],FILE_APPEND_DATA,FILE_SHARE_READ|FILE_SHARE_WRITE,&sa,OPEN_ALWAYS,FILE_ATTRIBUTE_NORMAL,nullptr);
   STARTUPINFOA si{};si.cb=sizeof(si);si.dwFlags=STARTF_USESTDHANDLES;si.hStdInput=sink;si.hStdOutput=sink;si.hStdError=errors;
   PROCESS_INFORMATION pi{};job=CreateJobObjectA(nullptr,nullptr);JOBOBJECT_EXTENDED_LIMIT_INFORMATION limits{};limits.BasicLimitInformation.LimitFlags=JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
   bool ready=job&&SetInformationJobObject(job,JobObjectExtendedLimitInformation,&limits,sizeof limits);
   auto command=cmd;
   if(ready&&CreateProcessA(argv[2],command.data(),nullptr,nullptr,TRUE,CREATE_NO_WINDOW|CREATE_SUSPENDED,nullptr,nullptr,&si,&pi)){
    if(AssignProcessToJobObject(job,pi.hProcess)){child=pi.hProcess;ResumeThread(pi.hThread);log("receiver child started pid="+std::to_string(pi.dwProcessId));}
    else{log("cannot contain receiver child");TerminateProcess(pi.hProcess,8);CloseHandle(pi.hProcess);}
    CloseHandle(pi.hThread);
   }else log("receiver launch failed error="+std::to_string(GetLastError()));
   CloseHandle(sink);CloseHandle(errors);if(!child){if(job)CloseHandle(job);job=nullptr;retry=GetTickCount64()+2000;}
  }
  Sleep(100);
 }
 if(child){WaitForSingleObject(child,1500);CloseHandle(child);}if(job)CloseHandle(job);
 log(quit?"SteamVR requested exit":"receiver stop requested");shutdown();CloseHandle(mutex);return 0;
}
