"""Apply the reviewed sensor-layout adaptation to the pinned Frametop sources.

The pixel format and dimensions are device-observed. Left/right names are
provisional until independently checked against factory epipolar geometry.
"""
import argparse
from pathlib import Path


def replace_once(path, before, after):
    text=path.read_text(encoding='utf-8')
    if after in text:
        return
    if text.count(before)!=1:
        raise RuntimeError(f'Pinned source changed: {path}')
    path.write_text(text.replace(before,after),encoding='utf-8',newline='\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('hands_source',type=Path)
    root=parser.parse_args().hands_source
    replace_once(root/'camd/camd.c',
        'if (lay.fmt == XR_FMT_GREY8 && strstr(cam->sensor, opt_sensor))',
        'if ((lay.fmt == XR_FMT_GREY8 || (lay.fmt == XR_FMT_NV12 && strstr(cam->sensor, "og01a1b"))) && strstr(cam->sensor, opt_sensor))')
    replace_once(root/'camd/camd.c',
        'c->need = (size_t)c->lay.pitch * (c->color ? c->lay.height : c->lay.rows);',
        '// NV12 tracking cameras need only their 8-bit luma rows, never chroma.\n'
        '    c->need = (size_t)c->lay.pitch * ((c->color || c->lay.fmt == XR_FMT_NV12) ? c->lay.height : c->lay.rows);')
    replace_once(root/'camd/camd.c',
        '/* Mean of an 8-bit image on a sparse grid. */',
        '''/* Avoid stride aliasing: 1152*1024/512 samples only column zero.
 * Sample 32 columns by 16 rows of active mono pixels, excluding padding.
 * Packed 10-bit color keeps its original byte-layout sampling. */
static void sample_camera(const cam_t *c, const uint8_t *p, uint64_t *out)
{
    if (c->color) { sample_words(p, c->need, out); return; }
    for (unsigned i = 0; i < NSAMP; ++i) {
        unsigned y = ((2 * (i / 32) + 1) * c->lay.height) / 32;
        unsigned x = (((2 * (i % 32) + 1) * c->lay.width) / 64) & ~7u;
        if (x + 8 > c->lay.width) x = c->lay.width - 8;
        memcpy(&out[i], p + (size_t)y * c->lay.pitch + x, 8);
    }
}

/* Mean of an 8-bit image on a sparse grid. */''')
    path=root/'camd/camd.c'
    text=path.read_text(encoding='utf-8')
    # Only camera sample sites; the packed-color fallback remains sample_words.
    for pointer,output in [('c->map[s]','c->samp[s]'),('c->map[s]','cur'),
                           ('c->map[slot]','cur'),('c->map[s]','d'),('src','before'),('src','after')]:
        text=text.replace(f'sample_words({pointer}, c->need, {output});',
                          f'sample_camera(c, {pointer}, {output});')
    path.write_text(text,encoding='utf-8',newline='\n')
    replace_once(path,
        '            sample_camera(c, c->map[s], c->samp[s]);',
        '            buf_sync(c->fd[s], DMA_BUF_SYNC_START);\n'
        '            sample_camera(c, c->map[s], c->samp[s]);\n'
        '            buf_sync(c->fd[s], DMA_BUF_SYNC_END);')
    replace_once(path,
        '        sample_camera(c, c->map[s], cur);\n\n        changed[s] = 0;',
        '        buf_sync(c->fd[s], DMA_BUF_SYNC_START);\n'
        '        sample_camera(c, c->map[s], cur);\n'
        '        buf_sync(c->fd[s], DMA_BUF_SYNC_END);\n\n        changed[s] = 0;')
    replace_once(root/'track/main.cpp',
        'if (!std::strcmp(name, "msm_vfe3_video0")) return "slam_left";\n'
        '    if (!std::strcmp(name, "msm_vfe4_video0")) return "slam_right";\n'
        '    if (!std::strcmp(name, "msm_vfe2_video0")) return "upper_left";\n'
        '    if (!std::strcmp(name, "msm_vfe2_video1")) return "upper_right";',
        '// Ilya Frame 20260922.6101926: dimensions verified; side ordering\n'
        '    // must pass the camera geometry probe before interaction.\n'
        '    if (!std::strcmp(name, "msm_vfe0_video0")) return "slam_right";\n'
        '    if (!std::strcmp(name, "msm_vfe1_video0")) return "slam_left";\n'
        '    if (!std::strcmp(name, "msm_vfe3_video0")) return "upper_left";\n'
        '    if (!std::strcmp(name, "msm_vfe4_video0")) return "upper_right";')
    replace_once(root/'track/main.cpp',
        'if (!name || !calib.count(name)) continue;',
        'if (!name || !calib.count(name)) continue;\n'
        '        if (ring.camera(i).width != unsigned(calib[name].width) || ring.camera(i).height != unsigned(calib[name].height)) {\n'
        '            std::fprintf(stderr, "Camera/calibration dimension mismatch for %s; refusing invalid geometry\\n", name);\n'
        '            return 1;\n'
        '        }')
    replace_once(root/'track/tracker.cpp',
        '\n}\n\nvoid Tracker::add_tiles(const Camera &cam, double frac, int gx, int gy) {',
        '''
    // Opt-in recovery search for rotated hands; retain calibrated crops first.
    // This changes acquisition only, not landmark coordinates or camera geometry.
    if (const char *flag = std::getenv("FRAME_HANDS_ROTATION_SEARCH"); flag && std::string(flag) == "1") {
        const size_t count = tiles_.size();
        for (size_t i = 0; i < count; ++i) {
            for (double angle : {-M_PI/2, -M_PI/4, M_PI/4, M_PI/2, M_PI}) {
                Tile t = tiles_[i];
                t.rotation += angle;
                t.weight *= 0.6;
                tiles_.push_back(t);
            }
        }
        search_budget_ = 6;
    }
}

void Tracker::add_tiles(const Camera &cam, double frac, int gx, int gy) {''')
    # The diagnostic oracle must validate alternative orientations before NMS;
    # a confident palm proposal can still produce invalid finger landmarks.
    replace_once(root/'track/tracker.cpp',
        'if (!near) crops.push_back(v);',
        'if (!near || std::getenv("FRAME_HANDS_ROTATION_SEARCH")) crops.push_back(v);')
    replace_once(root/'track/tracker.cpp',
        'for (auto *list : {&views_, &fresh})\n'
        '                    for (View &v : *list) near = near || (v.cam == picked[i]->cam && norm(v.roi.center - roi.center) < 0.5 * roi.size);',
        '''for (View &v : views_)
                    near = near || (v.cam == picked[i]->cam && norm(v.roi.center - roi.center) < 0.5 * roi.size);
                for (View &v : fresh)
                    near = near || (v.cam == picked[i]->cam && norm(v.roi.center - roi.center) < 0.5 * roi.size &&
                        (!std::getenv("FRAME_HANDS_ROTATION_SEARCH") ||
                         std::fabs(std::remainder(v.roi.rotation-roi.rotation,2*M_PI)) < M_PI/12));''')
    replace_once(root/'track/tracker.cpp',
        '''        for (View &v : fresh)
            if (v.lm.presence >= min_presence_) {
                v.roi = v.lm.next_roi();
                v.frames = 1;
                views_.push_back(v);
            }''',
        '''        // Suppress duplicates after landmark validation, keeping the best pose.
        std::sort(fresh.begin(),fresh.end(),[](const View &a,const View &b) {
            return a.lm.presence > b.lm.presence;
        });
        for (View &v : fresh)
            if (v.lm.presence >= min_presence_) {
                bool duplicate=false;
                for (const View &o : views_)
                    duplicate = duplicate || (o.cam==v.cam && same_hand(o.lm,v.lm,hand_size(v.lm)));
                if (duplicate) continue;
                v.roi = v.lm.next_roi();
                v.frames = 1;
                views_.push_back(v);
            }''')
    # Optional exact-frame image/landmark audit; never enabled by default.
    replace_once(root/'track/io.h',
        'std::string run_dir();',
        'std::string run_dir();\n'
        'std::vector<const Hand *> select_publication_hands(const std::vector<const Hand *> &in);')
    replace_once(root/'track/io.cpp',
        '''void Publisher::write(const std::vector<const Hand *> &in, uint64_t capture_ns) {
    std::vector<const Hand *> hands = in;
    std::sort(hands.begin(), hands.end(), [](const Hand *a, const Hand *b) { return a->frames > b->frames; });
    if (hands.size() > FH_HANDS_MAX_HANDS) hands.resize(FH_HANDS_MAX_HANDS);''',
        '''std::vector<const Hand *> select_publication_hands(const std::vector<const Hand *> &in) {
    std::vector<const Hand *> ranked=in, selected;
    std::sort(ranked.begin(),ranked.end(),[](const Hand *a,const Hand *b) {
        return a->frames!=b->frames ? a->frames>b->frames : a->id<b->id;
    });
    bool sides[2]={false,false};
    for(const Hand *hand:ranked) {
        const int side=hand->right()?1:0;
        if(sides[side]) continue;
        sides[side]=true;selected.push_back(hand);
    }
    return selected;
}

void Publisher::write(const std::vector<const Hand *> &in, uint64_t capture_ns) {
    // Duplicate side hypotheses remain in diagnostic logs, but cannot consume
    // both slots or invalidate the otherwise usable opposite hand for readers.
    const std::vector<const Hand *> hands=select_publication_hands(in);''')
    replace_once(root/'track/tracker.h',
        '    double right_score = 0.5;   // the model\'s right-hand score (these images aren\'t mirrored)',
        '    bool side_from_open = false; // optional identity evidence from an open hand\n'
        '    double right_score = 0.5;   // the model\'s right-hand score (these images aren\'t mirrored)')
    replace_once(root/'track/tracker.cpp',
        '    for (View *v : views) hand.right_score += 0.2 * (v->lm.right - hand.right_score);',
        '''    const char *open_side=std::getenv("FRAME_HANDS_OPEN_SIDE");
    if(open_side && std::string(open_side)=="1") {
        double weighted=0, weight=0, all=0, all_weight=0;
        for(const View *v:views) {
            auto point=[&](int i){return V3{v->lm.world[i][0],v->lm.world[i][1],v->lm.world[i][2]};};
            double curl=0;
            for(int knuckle:{5,9,13,17}) {
                const double length=norm(point(knuckle)-point(0));
                curl+=length>1e-4 ? norm(point(knuckle+3)-point(0))/length/4 : 0;
            }
            all+=v->lm.presence*v->lm.right;all_weight+=v->lm.presence;
            if(curl>=1.4 && v->lm.presence>=0.8) {
                weighted+=v->lm.presence*v->lm.right;weight+=v->lm.presence;
            }
        }
        if(weight>0) {
            hand.right_score+=0.2*(weighted/weight-hand.right_score);
            if(hand.right_score<0.2 || hand.right_score>0.8)hand.side_from_open=true;
        } else if(!hand.side_from_open && all_weight>0) {
            hand.right_score+=0.2*(all/all_weight-hand.right_score);
        }
    } else {
        for (View *v : views) hand.right_score += 0.2 * (v->lm.right - hand.right_score);
    }''')
    replace_once(root/'track/tracker.cpp',
        '                    if (m[x] >= 0 && c[x][m[x]] >= 0) s += c[x][m[x]] - kPairBonus;',
        '''                    if (m[x] >= 0 && c[x][m[x]] >= 0) {
                        s += c[x][m[x]] - kPairBonus;
                        const char *continuity=std::getenv("FRAME_HANDS_PAIR_CONTINUITY");
                        if(continuity && std::string(continuity)=="1" &&
                           A[x]->hand>0 && A[x]->hand==B[m[x]]->hand &&
                           hands_.count(A[x]->hand) && hands_.at(A[x]->hand).frames>=5)
                            s-=1.0;
                    }''')
    replace_once(root/'track/nets.cpp',
        '    const float min_logit = std::log(0.5f / 0.5f);  // score 0.5',
        '''    float palm_threshold=0.5f;
    if(const char *value=std::getenv("FRAME_HANDS_PALM_THRESHOLD")) {
        const float proposed=std::atof(value);
        if(proposed>=0.1f && proposed<=0.5f) palm_threshold=proposed;
    }
    const float min_logit=std::log(palm_threshold/(1.0f-palm_threshold));''')
    replace_once(root/'track/tracker.cpp',
        '                if (!near) fresh.push_back({picked[i]->cam, roi, 0, {}, false, 0});',
        '''                // Diagnostic option: a large existing crop can cover the other
                // hand. Validate joints before rejecting nearby palm proposals.
                const char *late_nms=std::getenv("FRAME_HANDS_LATE_NMS");
                if (!near || (late_nms && std::string(late_nms)=="1"))
                    fresh.push_back({picked[i]->cam, roi, 0, {}, false, 0});''')
    # OpenCV Zoo's reference landmark preprocessing bounds the rotated palm
    # keypoints, then shifts upward by 0.4 heights and expands by 3.0.
    # Keep this isolated from the live default until recorded-image evaluation.
    replace_once(root/'track/nets.cpp',
        '    const double h = size[1];',
        '''    const char *keypoint_roi = std::getenv("FRAME_HANDS_KEYPOINT_ROI");
    if (keypoint_roi && std::string(keypoint_roi) == "1") {
        const double c=std::cos(rot), s=std::sin(rot);
        V2 lo{1e9,1e9}, hi{-1e9,-1e9};
        for (const V2 &point : kp) {
            const V2 d=point-center;
            const V2 q{c*d[0]+s*d[1],-s*d[0]+c*d[1]};
            for(int k=0;k<2;++k) {lo[k]=std::min(lo[k],q[k]);hi[k]=std::max(hi[k],q[k]);}
        }
        const double width=hi[0]-lo[0], height=hi[1]-lo[1];
        V2 mid=(lo+hi)*0.5;
        mid[1]-=0.4*height;
        return {center+V2{c*mid[0]-s*mid[1],s*mid[0]+c*mid[1]},3.0*std::max(width,height),rot};
    }
    const double h = size[1];''')
    replace_once(root/'track/main.cpp',
        '    Tracker tracker(used, nets, pool);',
        '''    Tracker tracker(used, nets, pool);
    std::ofstream frame_audit;
    if (const char *audit_path = std::getenv("FRAME_HANDS_AUDIT")) {
        frame_audit.open(audit_path);
        if (!frame_audit) { std::fprintf(stderr,"Cannot open frame audit\\n"); return 1; }
        frame_audit.precision(10);
    }''')
    replace_once(root/'track/main.cpp',
        '        if (publish) pub.write(hands, capture), gestures.write(pinch, grip, capture);',
        r'''        if (publish) pub.write(hands, capture), gestures.write(pinch, grip, capture);
        if (frame_audit.is_open()) {
            frame_audit << "set " << tmin << " " << capture << " " << hands.size() << "\n";
            for (const Hand *h : hands)
                frame_audit << "hand " << h->id << " " << h->right_score << " " << h->nviews << "\n";
            for (const Seen &v : views) {
                frame_audit << "view " << v.cam << " " << v.hand << " " << v.lm.presence << " " << v.lm.right;
                for (int k=0;k<21;++k) frame_audit << " " << v.lm.pts[k][0] << " " << v.lm.pts[k][1];
                frame_audit << "\n";
            }
            frame_audit << "end\n";
        }''')
    # Keep the numerical camera-check utility consistent with this device route.
    replace_once(root/'track/replay.cpp', '#include <set>', '#include <set>\n#include <fstream>')
    replace_once(root/'track/replay.cpp',
        '    Tracker tracker(used, nets, pool);',
        '''    Tracker tracker(used, nets, pool);
    std::ofstream frame_audit;
    if (const char *audit_path=std::getenv("FRAME_HANDS_AUDIT")) {
        frame_audit.open(audit_path);
        if(!frame_audit) return 1;
        frame_audit.precision(10);
    }''')
    replace_once(root/'track/replay.cpp',
        '            const std::vector<Seen> seen = tracker.views_now();',
        r'''            const std::vector<Seen> seen = tracker.views_now();
            if(frame_audit.is_open()) {
                frame_audit << "set " << t << " " << t << " " << out.size() << "\n";
                for(const Hand *h:out) frame_audit << "hand " << h->id << " " << h->right_score << " " << h->nviews << "\n";
                for(const Seen &v:seen) {
                    frame_audit << "view " << v.cam << " " << v.hand << " " << v.lm.presence << " " << v.lm.right;
                    for(int k=0;k<21;++k) frame_audit << " " << v.lm.pts[k][0] << " " << v.lm.pts[k][1];
                    frame_audit << "\n";
                }
                frame_audit << "end\n";
            }''')
    # Separate opt-in cadence experiment: fresh camera updates at 30 Hz and
    # smaller search batches each update instead of six crops every 200 ms.
    replace_once(root/'track/tracker.cpp',
        'double Tracker::interval() const {',
        '''double Tracker::interval() const {
    const char *full_rate = std::getenv("FRAME_HANDS_FULL_RATE");
    if (full_rate && std::string(full_rate) == "1") return 1 / 30.0;''')
    if '    std::set<std::string> deficient_cameras;' not in (root/'track/tracker.cpp').read_text():
        replace_once(root/'track/tracker.cpp',
            '    if (tracked.size() < 2 && (t_ns - search_ns_) / 1e9 >= kSearchInterval - 0.01) {',
            '''    const char *full_rate_flag = std::getenv("FRAME_HANDS_FULL_RATE");
    const bool full_rate = full_rate_flag && std::string(full_rate_flag) == "1";
    const double search_interval = full_rate ? 1 / 30.0 : kSearchInterval;
    if (tracked.size() < 2 && (t_ns - search_ns_) / 1e9 >= search_interval - 0.01) {''')
    replace_once(root/'track/tracker.cpp',
        '        const int budget = tracked.empty() ? search_budget_ : std::max(1, search_budget_ - 1);',
        '        const int budget = full_rate ? 3 : (tracked.empty() ? search_budget_ : std::max(1, search_budget_ - 1));')
    for relative in ['tools/check_sides.py']:
        replace_once(root/relative,
            "PIPES = {'msm_vfe3_video0': 'slam_left', 'msm_vfe4_video0': 'slam_right'}",
            "PIPES = {'msm_vfe1_video0': 'slam_left', 'msm_vfe0_video0': 'slam_right'}")
    # Diagnostic only: test whether conservative crop-center margins suppress
    # a useful second view. Presence, off-axis and triangulation gates stay intact.
    replace_once(root/'track/tracker.cpp',
        '    const double m = 0.12;',
        '''    const char *edge_handoff = std::getenv("FRAME_HANDS_EDGE_HANDOFF");
    const double m = edge_handoff && std::string(edge_handoff)=="1" ? 0.03 : 0.12;''')
    replace_once(root/'track/tracker.cpp',
        '    if (tracked.size() < 2 && (t_ns - search_ns_) / 1e9 >= search_interval - 0.01) {',
        '''    std::set<std::string> deficient_cameras;
    const char *stereo_search = std::getenv("FRAME_HANDS_STEREO_SEARCH");
    if(stereo_search && std::string(stereo_search)=="1" && tracked.size()==2) {
        for(const auto &[name,image]:images) {
            std::set<int> ids;
            for(const View &v:views_) if(v.fresh && v.cam->name==name) ids.insert(v.hand);
            if(ids.size()<2) deficient_cameras.insert(name);
        }
    }
    if ((tracked.size() < 2 || !deficient_cameras.empty()) && (t_ns - search_ns_) / 1e9 >= search_interval - 0.01) {''')
    replace_once(root/'track/tracker.cpp',
        '                    (!best || t.credit > best->credit))',
        '''                    (deficient_cameras.empty() || deficient_cameras.count(t.cam->name)) &&
                    (!best || t.credit > best->credit))''')
    # Optional replay-only raw triangulated/mono pose suffix, after the existing
    # model-world and filtered coordinates. Never enabled by the live runner.
    replace_once(root/'track/replay.cpp',
        "                    std::fputc('\\n', pp);",
        '''                    if(const char *raw=std::getenv("FRAME_HANDS_POSE_RAW_APPEND");raw && std::string(raw)=="1")
                        for(int k=0;k<21;++k)for(int j=0;j<3;++j)std::fprintf(pp," %.6f",h->pts[k][j]);
                    std::fputc('\\n', pp);''')
    # Opt-in identity evidence, independent of pose filtering/prediction. The
    # classifier can flip on curled hands and suppress a still-tracked opposite
    # hand at publication. Require agreeing open stereo observations to latch.
    (root/'track/hand_identity.h').write_text('''#pragma once
#include <cstdint>
struct HandIdentity {
    bool latched=false, right=false;
    int candidate=-1, streak=0, stereo_streak=0, corrections=0;
    int64_t last=0;
    void observe(int vote, int64_t now, bool stereo=true) {
        if(now<=last) return; // repeated polls are not independent evidence
        if(vote<0) {candidate=-1;streak=0;stereo_streak=0;last=now;return;}
        if(vote!=candidate || now-last>100000000) {streak=0;stereo_streak=0;}
        candidate=vote;last=now;++streak;
        stereo_streak=stereo ? stereo_streak+1 : 0;
        const bool enough=stereo_streak>=(latched ? 30 : 6) || streak>=(latched ? 60 : 12);
        if(enough && (!latched || right!=bool(vote))) {
            if(latched) ++corrections;
            latched=true;right=bool(vote);
        }
    }
};
''',encoding='utf-8',newline='\n')
    replace_once(root/'track/tracker.h', '#include "nets.h"',
        '#include "nets.h"\n#include "hand_identity.h"')
    replace_once(root/'track/tracker.h',
        '    bool side_from_open = false; // optional identity evidence from an open hand',
        '    HandIdentity identity; // opt-in stable stereo handedness\n'
        '    bool side_from_open = false; // optional identity evidence from an open hand')
    # Upgrade the earlier stereo-only experiment before matching the full block.
    if '    const char *identity_flag=' in (root/'track/tracker.cpp').read_text():
        replace_once(root/'track/tracker.cpp', '        bool reliable=views.size()>=2;',
            '        bool reliable=!views.empty();')
        replace_once(root/'track/tracker.cpp', '        hand.identity.observe(reliable ? vote : -1,t_ns);',
            '        hand.identity.observe(reliable ? vote : -1,t_ns,views.size()>=2);')
    replace_once(root/'track/tracker.cpp',
        '    const char *open_side=std::getenv("FRAME_HANDS_OPEN_SIDE");',
        '''    const char *identity_flag=std::getenv("FRAME_HANDS_IDENTITY_LATCH");
    const bool identity_latch=identity_flag && std::string(identity_flag)=="1";
    if(identity_latch) {
        int vote=-1;
        bool reliable=!views.empty();
        for(const View *v:views) {
            auto point=[&](int i){return V3{v->lm.world[i][0],v->lm.world[i][1],v->lm.world[i][2]};};
            double curl=0;
            for(int knuckle:{5,9,13,17}) {
                const double length=norm(point(knuckle)-point(0));
                curl+=length>1e-4 ? norm(point(knuckle+3)-point(0))/length/4 : 0;
            }
            const int side=v->lm.right>0.85 ? 1 : (v->lm.right<0.15 ? 0 : -1);
            if(v->lm.presence<0.8 || curl<1.4 || side<0 || (vote>=0 && vote!=side)) reliable=false;
            if(side>=0) vote=side;
        }
        hand.identity.observe(reliable ? vote : -1,t_ns,views.size()>=2);
    }
    const char *open_side=std::getenv("FRAME_HANDS_OPEN_SIDE");''')
    replace_once(root/'track/tracker.cpp',
        '    return true;\n}\n\n// How badly two views',
        '''    if(identity_latch && hand.identity.latched)
        hand.right_score=hand.identity.right ? 0.99 : 0.01;
    return true;
}

// How badly two views''')


if __name__=='__main__':main()
