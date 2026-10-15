#pragma once
#include <string>
#include <cstdint>
#ifdef _WIN32
#include <windows.h>
#define FHS_EXPORT __declspec(dllexport)
namespace fhs_platform {
using Path=std::wstring;
inline uint64_t now_ms(){
#ifdef FRAME_HANDS_DRIVER_MODEL
    LARGE_INTEGER tick{},frequency{};QueryPerformanceCounter(&tick);QueryPerformanceFrequency(&frequency);
    return uint64_t(tick.QuadPart)*1000/uint64_t(frequency.QuadPart);
#else
    return GetTickCount64();
#endif
}
template<class T> bool read_exact(const Path& path,T& value){
    HANDLE f=CreateFileW(path.c_str(),GENERIC_READ,FILE_SHARE_READ|FILE_SHARE_WRITE|FILE_SHARE_DELETE,nullptr,OPEN_EXISTING,FILE_ATTRIBUTE_NORMAL,nullptr);
    if(f==INVALID_HANDLE_VALUE)return false;
    DWORD bytes=0;LARGE_INTEGER size{};
    bool ok=GetFileSizeEx(f,&size)&&size.QuadPart==sizeof value&&ReadFile(f,&value,sizeof value,&bytes,nullptr)&&bytes==sizeof value;
    CloseHandle(f);return ok;
}
inline Path sibling(const Path& p,const char* name){return p.substr(0,p.find_last_of(L"\\/"))+L"\\"+Path(name,name+std::char_traits<char>::length(name));}
inline Path mailbox_path(){
    HMODULE module=nullptr;GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS|GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,reinterpret_cast<LPCWSTR>(&now_ms),&module);
    wchar_t path[32768]{};if(!GetModuleFileNameW(module,path,32768))return {};
    return sibling(path,"input.bin");
}
}
#else
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>
#include <time.h>
#define FHS_EXPORT __attribute__((visibility("default")))
namespace fhs_platform {
using Path=std::string;
inline uint64_t now_ms(){timespec t{};if(clock_gettime(CLOCK_MONOTONIC,&t))return 0;return uint64_t(t.tv_sec)*1000+uint64_t(t.tv_nsec)/1000000;}
inline Path mailbox_path(){return "/run/user/"+std::to_string(getuid())+"/frame-hands-native/input.bin";}
inline Path sibling(const Path& p,const char* name){return p.substr(0,p.find_last_of('/'))+"/"+name;}
template<class T> bool read_exact(const Path& path,T& value){
    int fd=open(path.c_str(),O_RDONLY|O_CLOEXEC|O_NOFOLLOW|O_NONBLOCK);
    if(fd<0)return false;
    struct stat st{};
    bool ok=fstat(fd,&st)==0&&S_ISREG(st.st_mode)&&st.st_uid==getuid()&&!(st.st_mode&0022)&&st.st_size==sizeof value;
    if(ok)ok=pread(fd,&value,sizeof value,0)==sizeof value;
    close(fd);return ok;
}
}
#endif
