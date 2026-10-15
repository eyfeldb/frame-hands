/* Exact forward-camera discovery; never infer plane pairs from FD adjacency.
 * Included after xrcams.c's descriptor scan and legacy grouping helpers. */
#include "direct_queue.h"

static bool exact_forward(const xr_camera_t *c)
{
    return strstr(c->sensor,"og01a1b") && c->pixfmt==V4L2_PIX_FMT_NV12 &&
           c->width==1056 && c->height==1024 && c->nplanes==2;
}

static const fdent_t *exact_descriptor(int fd, size_t size)
{
    const fdent_t *found=NULL;
    for(int i=0;i<nfdents;i++) if(fdents[i].xfd==fd) {
        if(found || fdents[i].kind!=FD_DMABUF || fdents[i].size!=size || !fdents[i].ino)
            return NULL;
        found=&fdents[i];
    }
    return found;
}

static int exact_pair(int fd,unsigned index,const xr_camera_t *c,int pair[2])
{
    struct v4l2_plane p[VIDEO_MAX_PLANES]={0};
    struct v4l2_buffer b={0};
    b.type=V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;b.index=index;
    b.length=VIDEO_MAX_PLANES;b.m.planes=p;
    int rc;
    do {rc=ioctl(fd,VIDIOC_QUERYBUF,&b);} while(rc<0 && errno==EINTR);
    if(rc<0)return errno;
    if(!queue_metadata_valid(&b,p,index,c->planesize[0],c->planesize[1]))return EPROTO;
    for(int k=0;k<2;k++) {
        if(!exact_descriptor(p[k].m.fd,p[k].length))return ESTALE;
        pair[k]=p[k].m.fd;
    }
    return 0;
}

static bool exact_forward_groups(xr_state_t *st,char *err,size_t errn)
{
    /* Stage everything before publishing. Keep non-forward legacy groups. */
    xr_group_t *groups=calloc(XR_MAX_GROUPS,sizeof(*groups));
    if(!groups){set_err(err,errn,"exact discovery: allocation failed");return false;}
    int count=0;
    for(int i=0;i<st->ngroups;i++) {
        bool forward=false;
        for(int c=0;c<st->ncameras;c++) if(exact_forward(&st->cameras[c]) &&
            st->groups[i].planesize[0]==st->cameras[c].planesize[0] &&
            st->groups[i].planesize[1]==st->cameras[c].planesize[1])forward=true;
        if(!forward)groups[count++]=st->groups[i];
    }
    for(int ci=0;ci<st->ncameras;ci++) {
        xr_camera_t *c=&st->cameras[ci];
        if(!exact_forward(c))continue;
        if(count==XR_MAX_GROUPS){set_err(err,errn,"exact discovery: group limit");goto fail;}
        int fd=open(c->path,O_RDONLY|O_NONBLOCK|O_CLOEXEC);
        if(fd<0){set_err(err,errn,"%s: query open: %s",c->path,strerror(errno));goto fail;}
        xr_group_t *g=&groups[count];g->cam=c;
        snprintf(g->sensor,sizeof(g->sensor),"%s",c->sensor);
        g->planesize[0]=c->planesize[0];g->planesize[1]=c->planesize[1];
        int failure=0;
        for(unsigned index=0;index<32;index++) {
            int pair[2];int e=exact_pair(fd,index,c,pair);
            if(e==EINVAL && index>=2)break;
            if(e){failure=e;break;}
            const fdent_t *a=exact_descriptor(pair[0],c->planesize[0]);
            const fdent_t *b=exact_descriptor(pair[1],c->planesize[1]);
            if(a->ino==b->ino){failure=EPROTO;break;}
            for(int j=0;j<=count;j++)for(int k=0;k<groups[j].nbufs;k++) {
                const xr_bufref_t *old=&groups[j].buf[k];
                const fdent_t *oa=exact_descriptor(old->xfd,old->size);
                const fdent_t *ob=exact_descriptor(old->xfd1,old->size1);
                if((oa && (oa->ino==a->ino || oa->ino==b->ino)) ||
                   (ob && (ob->ino==a->ino || ob->ino==b->ino)))failure=EPROTO;
            }
            if(failure)break;
            g->buf[g->nbufs++]=(xr_bufref_t){pair[0],pair[1],c->planesize[0],c->planesize[1]};
        }
        if(!failure && (g->nbufs<2 || g->nbufs==32))failure=EPROTO;
        /* A second complete metadata pass rejects identities that changed mid-scan. */
        for(int i=0;!failure && i<g->nbufs;i++) {
            int pair[2];failure=exact_pair(fd,(unsigned)i,c,pair);
            if(!failure && (pair[0]!=g->buf[i].xfd || pair[1]!=g->buf[i].xfd1))failure=ESTALE;
        }
        close(fd);
        if(failure){set_err(err,errn,"%s: exact discovery: %s",c->path,strerror(failure));goto fail;}
        count++;
    }
    memcpy(st->groups,groups,(size_t)count*sizeof(*groups));st->ngroups=count;
    free(groups);return true;
fail:
    free(groups);return false;
}
