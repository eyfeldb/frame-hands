/* Compile the real discovery implementation with metadata-only syscall stubs. */
#define ioctl fixture_ioctl
#define open fixture_open
#define close fixture_close
#include "camd/xrcams.c"
#undef ioctl
#undef open
#undef close
#include <assert.h>
static int mode, calls;
static const int left[16]={155,157,159,162,164,166,168,170,172,174,176,180,183,185,187,189};
static const int right[16]={193,195,197,199,201,203,205,207,210,212,214,216,218,220,222,224};
int fixture_open(const char *path,int flags,...){assert((flags&O_ACCMODE)==O_RDONLY);return strstr(path,"video0")?100:103;}
int fixture_close(int fd){assert(fd==100 || fd==103);return 0;}
int fixture_ioctl(int fd,unsigned long request,...)
{
    assert(request==VIDIOC_QUERYBUF);calls++;
    va_list args;va_start(args,request);struct v4l2_buffer *b=va_arg(args,void*);va_end(args);
    if(mode==6){errno=EIO;return -1;}
    if(b->index>=16){errno=EINVAL;return -1;}
    unsigned i=b->index;
    int p=(fd==100?left:right)[i];
    int p1=p+1;if(p==159)p1=161;
            if(p==176)p1=178;
            if(p==180)p1=182;
            if(p==207)p1=209;
    if(mode==3 && i==1){p=left[0];p1=p+1;}
    if(mode==4 && fd==103 && i==0){p=left[0];p1=p+1;}
    if(mode==5 && calls>17 && fd==100 && i==0){p=left[1];p1=p+1;}
    b->memory=V4L2_MEMORY_DMABUF;b->length=2;
    b->m.planes[0].m.fd=p;b->m.planes[0].length=1769472;
    b->m.planes[1].m.fd=p1;b->m.planes[1].length=151552;
    if(mode==2)b->m.planes[0].data_offset=1;
    return 0;
}
static xr_state_t st;
static void setup(void)
{
    memset(&st,0,sizeof(st));nfdents=0;calls=0;
    st.ncameras=2;
    for(int c=0;c<2;c++) {
        xr_camera_t *cam=&st.cameras[c];cam->nplanes=2;cam->width=1056;cam->height=1024;
        cam->pixfmt=V4L2_PIX_FMT_NV12;cam->planesize[0]=1769472;cam->planesize[1]=151552;
        snprintf(cam->sensor,sizeof(cam->sensor),"og01a1b 4-%s",c?"0060":"0036");
        snprintf(cam->path,sizeof(cam->path),"/dev/video%d",c?3:0);
        fdents[nfdents++]=(fdent_t){.xfd=c?192:154,.kind=FD_SUBDEV_SENSOR};
        snprintf(fdents[nfdents-1].sensor,XR_SENSOR_LEN,"%s",cam->sensor);
        for(int i=0;i<16;i++) {
            int p=(c?right:left)[i],p1=p+1;
            if(p==159)p1=161;
            if(p==176)p1=178;
            if(p==180)p1=182;
            if(p==207)p1=209;
            fdents[nfdents++]=(fdent_t){.xfd=p,.kind=FD_DMABUF,.size=1769472,.ino=1000+p};
            if(p==159)fdents[nfdents++]=(fdent_t){.xfd=160,.kind=FD_DMABUF,.size=33554432,.ino=1160};
            fdents[nfdents++]=(fdent_t){.xfd=p1,.kind=FD_DMABUF,.size=151552,.ino=1000+p1};
        }
    }
    build_groups(&st);
    /* Actual old grouping drops two early buffers and the interleaved plane pair. */
    assert(st.ngroups==2 && st.groups[0].nbufs==13 && st.groups[1].nbufs==16);
}
int main(void)
{
    char err[256];mode=0;setup();assert(exact_forward_groups(&st,err,sizeof(err)));
    assert(st.ngroups==2 && st.groups[0].nbufs==16 && st.groups[1].nbufs==16);
    assert(st.groups[0].buf[2].xfd==159 && st.groups[0].buf[2].xfd1==161);
    for(mode=1;mode<=6;mode++) {
        setup();if(mode==1)fdents[1].kind=FD_VIDEO;
        assert(!exact_forward_groups(&st,err,sizeof(err)));
        assert(st.ngroups==2 && st.groups[0].nbufs==13); /* atomic failure */
    }
    mode=0;setup();fdents[3].ino=fdents[1].ino;assert(!exact_forward_groups(&st,err,sizeof(err)));
    puts("PASS: interleaved 13/16 -> exact 16/16; missing descriptor, malformed metadata, duplicates, cross-camera aliases, changing queues and ioctl failure rejected");
    return 0;
}
