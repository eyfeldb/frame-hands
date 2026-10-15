#pragma once
#include <stdexcept>
#include <cstdint>
#include <vector>
#include <string>
#include <array>
#include <cstdio>
#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#include <GL/gl.h>
#include <GL/glext.h>
#define FH_GL_FUNCTIONS(X) \
 X(PFNGLCREATESHADERPROC,glCreateShader) X(PFNGLSHADERSOURCEPROC,glShaderSource) X(PFNGLCOMPILESHADERPROC,glCompileShader) X(PFNGLGETSHADERIVPROC,glGetShaderiv) X(PFNGLGETSHADERINFOLOGPROC,glGetShaderInfoLog) \
 X(PFNGLCREATEPROGRAMPROC,glCreateProgram) X(PFNGLATTACHSHADERPROC,glAttachShader) X(PFNGLBINDATTRIBLOCATIONPROC,glBindAttribLocation) X(PFNGLLINKPROGRAMPROC,glLinkProgram) X(PFNGLGETPROGRAMIVPROC,glGetProgramiv) X(PFNGLUSEPROGRAMPROC,glUseProgram) \
 X(PFNGLGETUNIFORMLOCATIONPROC,glGetUniformLocation) X(PFNGLUNIFORMMATRIX4FVPROC,glUniformMatrix4fv) X(PFNGLUNIFORM3FPROC,glUniform3f) \
 X(PFNGLGENBUFFERSPROC,glGenBuffers) X(PFNGLBINDBUFFERPROC,glBindBuffer) X(PFNGLBUFFERDATAPROC,glBufferData) X(PFNGLVERTEXATTRIBPOINTERPROC,glVertexAttribPointer) X(PFNGLENABLEVERTEXATTRIBARRAYPROC,glEnableVertexAttribArray) \
 X(PFNGLGENFRAMEBUFFERSPROC,glGenFramebuffers) X(PFNGLBINDFRAMEBUFFERPROC,glBindFramebuffer) X(PFNGLFRAMEBUFFERTEXTURE2DPROC,glFramebufferTexture2D) X(PFNGLCHECKFRAMEBUFFERSTATUSPROC,glCheckFramebufferStatus) \
 X(PFNGLGENRENDERBUFFERSPROC,glGenRenderbuffers) X(PFNGLBINDRENDERBUFFERPROC,glBindRenderbuffer) X(PFNGLRENDERBUFFERSTORAGEMULTISAMPLEPROC,glRenderbufferStorageMultisample) X(PFNGLFRAMEBUFFERRENDERBUFFERPROC,glFramebufferRenderbuffer) X(PFNGLBLITFRAMEBUFFERPROC,glBlitFramebuffer)
#define DECLARE(type,name) inline type name=nullptr;
FH_GL_FUNCTIONS(DECLARE)
#undef DECLARE
#else
#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <GLES3/gl3.h>
#endif
#include "mesh.h"
namespace fhdisplay {
inline void require(bool b,const char* text){if(!b)throw std::runtime_error(text);}
struct Graphics {
    static constexpr int width=1024,height=1024;
    GLuint texture=0,fbo=0,msaa=0,color=0,depth=0,program=0,vertex=0,index=0;
    GLint projection=0,view=0,tint=0;
#ifdef _WIN32
    HWND window=nullptr;HDC dc=nullptr;HGLRC context=nullptr;
#else
    EGLDisplay display=EGL_NO_DISPLAY;EGLContext context=EGL_NO_CONTEXT;EGLSurface surface=EGL_NO_SURFACE;
#endif
    void init(){
#ifdef _WIN32
        WNDCLASSW wc{};wc.style=CS_OWNDC;wc.lpfnWndProc=DefWindowProcW;wc.hInstance=GetModuleHandleW(nullptr);wc.lpszClassName=L"FrameHandsHiddenRenderer";RegisterClassW(&wc);
        window=CreateWindowW(wc.lpszClassName,L"Frame hands renderer",WS_POPUP,0,0,1,1,nullptr,nullptr,wc.hInstance,nullptr);require(window,"hidden GL window");
        dc=GetDC(window);PIXELFORMATDESCRIPTOR p{};p.nSize=sizeof p;p.nVersion=1;p.dwFlags=PFD_DRAW_TO_WINDOW|PFD_SUPPORT_OPENGL;p.iPixelType=PFD_TYPE_RGBA;p.cColorBits=32;p.cDepthBits=24;
        int format=ChoosePixelFormat(dc,&p);require(format&&SetPixelFormat(dc,format,&p),"GL pixel format");context=wglCreateContext(dc);require(context&&wglMakeCurrent(dc,context),"GL context");
#define LOAD(type,name) name=reinterpret_cast<type>(reinterpret_cast<void*>(wglGetProcAddress(#name)));require(reinterpret_cast<uintptr_t>(name)>3&&reinterpret_cast<uintptr_t>(name)!=~uintptr_t(0),#name);
        FH_GL_FUNCTIONS(LOAD)
#undef LOAD
        const std::string version="#version 120\n",attribute="attribute",varyout="varying",varyin="varying";
#else
        auto platform=reinterpret_cast<PFNEGLGETPLATFORMDISPLAYEXTPROC>(eglGetProcAddress("eglGetPlatformDisplayEXT"));require(platform,"EGL platform");
        display=platform(EGL_PLATFORM_SURFACELESS_MESA,EGL_DEFAULT_DISPLAY,nullptr);require(eglInitialize(display,nullptr,nullptr)&&eglBindAPI(EGL_OPENGL_ES_API),"EGL initialize");
        EGLint attrs[]={EGL_SURFACE_TYPE,EGL_PBUFFER_BIT,EGL_RENDERABLE_TYPE,EGL_OPENGL_ES3_BIT,EGL_RED_SIZE,8,EGL_GREEN_SIZE,8,EGL_BLUE_SIZE,8,EGL_ALPHA_SIZE,8,EGL_NONE};EGLConfig config;EGLint count;
        require(eglChooseConfig(display,attrs,&config,1,&count)&&count,"EGL config");EGLint size[]={EGL_WIDTH,1,EGL_HEIGHT,1,EGL_NONE},ver[]={EGL_CONTEXT_CLIENT_VERSION,3,EGL_NONE};
        surface=eglCreatePbufferSurface(display,config,size);context=eglCreateContext(display,config,EGL_NO_CONTEXT,ver);require(eglMakeCurrent(display,surface,surface,context),"EGL current");
        const std::string version="#version 300 es\nprecision highp float;\n",attribute="in",varyout="out",varyin="in";
#endif
        std::printf("renderer=%s\n",glGetString(GL_RENDERER));
        auto make=[&](GLenum type,const std::string& src){GLuint shader=glCreateShader(type);auto text=src.c_str();glShaderSource(shader,1,&text,nullptr);glCompileShader(shader);GLint ok;glGetShaderiv(shader,GL_COMPILE_STATUS,&ok);if(!ok){char msg[1024]{};glGetShaderInfoLog(shader,sizeof msg,nullptr,msg);std::fprintf(stderr,"%s\n",msg);}require(ok,"Shader compilation");return shader;};
        auto vs=make(GL_VERTEX_SHADER,version+attribute+" vec3 pos;\n"+attribute+" vec3 normal;\n"+varyout+" vec3 n; uniform mat4 projection;uniform mat4 view;void main(){n=mat3(view)*normal;gl_Position=projection*view*vec4(pos,1.0);}");
        std::string output;
#ifdef _WIN32
        output="gl_FragColor";
#else
        output="fragment";
#endif
        auto fs=make(GL_FRAGMENT_SHADER,version+varyin+" vec3 n;uniform vec3 tint;\n"+
#ifdef _WIN32
            ""
#else
            "out vec4 fragment;\n"
#endif
            +"void main(){vec3 nn=normalize(n);float light=0.36+0.48*max(dot(nn,normalize(vec3(0.35,0.7,0.6))),0.0)+0.20*max(dot(nn,normalize(vec3(-0.7,0.1,0.5))),0.0);"+output+"=vec4(tint*light,1.0);}");
        program=glCreateProgram();glAttachShader(program,vs);glAttachShader(program,fs);glBindAttribLocation(program,0,"pos");glBindAttribLocation(program,1,"normal");glLinkProgram(program);GLint linked;glGetProgramiv(program,GL_LINK_STATUS,&linked);require(linked,"Shader link");
        projection=glGetUniformLocation(program,"projection");view=glGetUniformLocation(program,"view");tint=glGetUniformLocation(program,"tint");
        glGenBuffers(1,&vertex);glGenBuffers(1,&index);
        glGenTextures(1,&texture);glBindTexture(GL_TEXTURE_2D,texture);glTexImage2D(GL_TEXTURE_2D,0,GL_RGBA8,width*2,height,0,GL_RGBA,GL_UNSIGNED_BYTE,nullptr);
        glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_LINEAR);glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_LINEAR);glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_WRAP_S,GL_CLAMP_TO_EDGE);glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_WRAP_T,GL_CLAMP_TO_EDGE);
        glGenFramebuffers(1,&fbo);glBindFramebuffer(GL_FRAMEBUFFER,fbo);glFramebufferTexture2D(GL_FRAMEBUFFER,GL_COLOR_ATTACHMENT0,GL_TEXTURE_2D,texture,0);require(glCheckFramebufferStatus(GL_FRAMEBUFFER)==GL_FRAMEBUFFER_COMPLETE,"Resolve FBO");
        glGenFramebuffers(1,&msaa);glBindFramebuffer(GL_FRAMEBUFFER,msaa);glGenRenderbuffers(1,&color);glBindRenderbuffer(GL_RENDERBUFFER,color);glRenderbufferStorageMultisample(GL_RENDERBUFFER,4,GL_RGBA8,width*2,height);glFramebufferRenderbuffer(GL_FRAMEBUFFER,GL_COLOR_ATTACHMENT0,GL_RENDERBUFFER,color);
        glGenRenderbuffers(1,&depth);glBindRenderbuffer(GL_RENDERBUFFER,depth);glRenderbufferStorageMultisample(GL_RENDERBUFFER,4,GL_DEPTH_COMPONENT24,width*2,height);glFramebufferRenderbuffer(GL_FRAMEBUFFER,GL_DEPTH_ATTACHMENT,GL_RENDERBUFFER,depth);require(glCheckFramebufferStatus(GL_FRAMEBUFFER)==GL_FRAMEBUFFER_COMPLETE,"MSAA FBO");
        glDisable(GL_CULL_FACE);glEnable(GL_DEPTH_TEST);require(glGetError()==GL_NO_ERROR,"GL initialization");
    }
    static std::array<float,16> column(const Matrix&m){std::array<float,16> out{};for(int r=0;r<4;++r)for(int c=0;c<4;++c)out[c*4+r]=m.m[r*4+c];return out;}
    void begin(){glBindFramebuffer(GL_FRAMEBUFFER,msaa);glClearColor(0,0,0,0);glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT);glUseProgram(program);}
    void eye(int side,const Matrix&p,const Matrix&v){glViewport(side*width,0,width,height);auto pp=column(p),vv=column(v);glUniformMatrix4fv(projection,1,GL_FALSE,pp.data());glUniformMatrix4fv(view,1,GL_FALSE,vv.data());}
    void draw(const Mesh& mesh,const std::vector<DrawVertex>& data){
        glUniform3f(tint,mesh.side?.94f:.66f,mesh.side?.76f:.85f,mesh.side?.56f:.96f);
        glBindBuffer(GL_ARRAY_BUFFER,vertex);glBufferData(GL_ARRAY_BUFFER,data.size()*sizeof(DrawVertex),data.data(),GL_STREAM_DRAW);
        glVertexAttribPointer(0,3,GL_FLOAT,GL_FALSE,sizeof(DrawVertex),nullptr);glVertexAttribPointer(1,3,GL_FLOAT,GL_FALSE,sizeof(DrawVertex),reinterpret_cast<void*>(12));glEnableVertexAttribArray(0);glEnableVertexAttribArray(1);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER,index);glBufferData(GL_ELEMENT_ARRAY_BUFFER,mesh.indices.size()*4,mesh.indices.data(),GL_STREAM_DRAW);glDrawElements(GL_TRIANGLES,int(mesh.indices.size()),GL_UNSIGNED_INT,nullptr);
    }
    void finish(){glBindFramebuffer(GL_READ_FRAMEBUFFER,msaa);glBindFramebuffer(GL_DRAW_FRAMEBUFFER,fbo);glBlitFramebuffer(0,0,width*2,height,0,0,width*2,height,GL_COLOR_BUFFER_BIT,GL_NEAREST);glBindFramebuffer(GL_FRAMEBUFFER,fbo);glFlush();require(glGetError()==GL_NO_ERROR,"Glove render");}
    std::vector<unsigned char> pixels(){std::vector<unsigned char> result(width*2*height*4);glReadPixels(0,0,width*2,height,GL_RGBA,GL_UNSIGNED_BYTE,result.data());return result;}
    ~Graphics(){
#ifdef _WIN32
        if(context){wglMakeCurrent(nullptr,nullptr);wglDeleteContext(context);}if(dc&&window)ReleaseDC(window,dc);if(window)DestroyWindow(window);
#else
        if(display!=EGL_NO_DISPLAY){eglMakeCurrent(display,EGL_NO_SURFACE,EGL_NO_SURFACE,EGL_NO_CONTEXT);if(context!=EGL_NO_CONTEXT)eglDestroyContext(display,context);if(surface!=EGL_NO_SURFACE)eglDestroySurface(display,surface);eglTerminate(display);}
#endif
    }
};
}
