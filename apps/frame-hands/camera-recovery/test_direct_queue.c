#include <assert.h>
#include <stdio.h>
#include <string.h>
#include "camd/direct_queue.h"
int main(void)
{
    struct v4l2_buffer b={0};struct v4l2_plane p[2]={0};
    b.index=7;b.type=V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
    b.memory=V4L2_MEMORY_DMABUF;b.length=2;
    p[0].m.fd=125;p[1].m.fd=126;p[0].length=1769472;p[1].length=151552;
    assert(queue_metadata_valid(&b,p,7,1769472,151552));
    assert(!queue_metadata_valid(&b,p,8,1769472,151552));
    b.memory=V4L2_MEMORY_MMAP;assert(!queue_metadata_valid(&b,p,7,1769472,151552));b.memory=V4L2_MEMORY_DMABUF;
    b.length=1;assert(!queue_metadata_valid(&b,p,7,1769472,151552));b.length=2;
    p[0].data_offset=1;assert(!queue_metadata_valid(&b,p,7,1769472,151552));p[0].data_offset=0;
    assert(!queue_metadata_valid(&b,p,7,1769471,151552));
    p[1].m.fd=125;assert(!queue_metadata_valid(&b,p,7,1769472,151552));p[1].m.fd=126;
    int fds[32];for(int i=0;i<32;i++)fds[i]=125+2*i;
    // Arbitrarily shuffled physical allocation: index-to-fd identity wins over order.
    for(int i=0;i<32;i++)assert(unique_queue_slot(fds,32,fds[(i*13)%32])==(i*13)%32);
    assert(unique_queue_slot(fds,32,999)==-1);
    fds[31]=fds[0];assert(unique_queue_slot(fds,32,fds[0])==-1);
    int output=999;
    assert(queue_buffer_fd(-1,0,1769472,151552,&output)==EBADF && output==999);
    puts("PASS exact queue: shuffled identities resolve; mismatched metadata, duplicate and absent fds, and failed queries rejected");
}
