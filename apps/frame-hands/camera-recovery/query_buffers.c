/* Read-only V4L2 queue metadata probe. No pixels, QBUF/DQBUF, REQBUFS,
 * STREAMON/OFF, control changes, mapping, or elevated privileges. */
#include <errno.h>
#include <fcntl.h>
#include <linux/videodev2.h>
#include <stdio.h>
#include <string.h>
#include <sys/ioctl.h>
#include <unistd.h>

int main(void)
{
    const int nodes[]={0,3};
    for(unsigned camera=0;camera<2;camera++) {
        char path[64];snprintf(path,sizeof(path),"/dev/video%d",nodes[camera]);
        int fd=open(path,O_RDONLY|O_NONBLOCK|O_CLOEXEC);
        if(fd<0) {printf("{\"node\":%d,\"open_errno\":%d}\n",nodes[camera],errno);continue;}
        for(int pass=0;pass<2;pass++) {
            for(unsigned index=0;index<33;index++) {
                struct v4l2_plane planes[VIDEO_MAX_PLANES]={0};
                struct v4l2_buffer b={0};
                b.type=V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
                b.index=index;b.m.planes=planes;b.length=VIDEO_MAX_PLANES;
                int rc=ioctl(fd,VIDIOC_QUERYBUF,&b),err=rc<0?errno:0;
                printf("{\"node\":%d,\"pass\":%d,\"index\":%u,\"errno\":%d",nodes[camera],pass,index,err);
                if(!rc) {
                    printf(",\"memory\":%u,\"flags\":%u,\"sequence\":%u,\"planes\":[",b.memory,b.flags,b.sequence);
                    for(unsigned p=0;p<b.length && p<VIDEO_MAX_PLANES;p++)
                        printf("%s{\"fd\":%d,\"length\":%u,\"used\":%u,\"offset\":%u}",p?",":"",planes[p].m.fd,planes[p].length,planes[p].bytesused,planes[p].data_offset);
                    printf("]");
                }
                printf("}\n");
                if(rc<0)break;
            }
            if(!pass)usleep(100000);
        }
        close(fd);
    }
    return 0;
}
