/* Read-only exact DMA-BUF identity. Frame kernel support is measured separately. */
#pragma once
#include <errno.h>
#include <linux/videodev2.h>
#include <stdbool.h>
#include <sys/ioctl.h>

static bool queue_metadata_valid(const struct v4l2_buffer *b,
                                 const struct v4l2_plane *p, unsigned index,
                                 unsigned plane0, unsigned plane1)
{
    return b->type==V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE && b->index==index &&
           b->memory==V4L2_MEMORY_DMABUF && b->length==2 &&
           p[0].m.fd>=0 && p[1].m.fd>=0 && p[0].m.fd!=p[1].m.fd &&
           p[0].length==plane0 && p[1].length==plane1 &&
           p[0].data_offset==0 && p[1].data_offset==0;
}

/* 0=valid, errno otherwise. QUERYBUF never dequeues or changes the queue. */
static int queue_buffer_fd(int video_fd, unsigned index, unsigned plane0,
                           unsigned plane1, int *source_fd)
{
    struct v4l2_plane p[VIDEO_MAX_PLANES]={0};
    struct v4l2_buffer b={0};
    b.type=V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;b.index=index;
    b.length=VIDEO_MAX_PLANES;b.m.planes=p;
    int rc;
    do { rc=ioctl(video_fd,VIDIOC_QUERYBUF,&b); } while(rc<0 && errno==EINTR);
    if(rc<0)return errno;
    if(!queue_metadata_valid(&b,p,index,plane0,plane1))return EPROTO;
    *source_fd=p[0].m.fd;
    return 0;
}

static int unique_queue_slot(const int *source_fds, int count, int requested)
{
    int found=-1;
    for(int i=0;i<count;i++) if(source_fds[i]==requested) {
        if(found>=0)return -1;
        found=i;
    }
    return found;
}
