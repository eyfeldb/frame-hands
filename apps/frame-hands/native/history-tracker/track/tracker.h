// Multi-camera hand tracking (a port of frame-hands' Python prototype; the scheduling is
// described in hands/README.md). All 3D is metres in the head frame.
#pragma once

#include "calib.h"
#include "nets.h"
#include "hand_identity.h"
#include "joint_identity.h"
#include "../native/tracker_rig_hook.h"

#include <condition_variable>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

// Runs batches of jobs on a few threads, each pinned to a core.
class Pool {
public:
    // One thread per entry of cpus, pinned there (round-robin if threads > cpus).
    Pool(int threads, const std::vector<int> &cpus);
    ~Pool();
    void run(std::vector<std::function<void()>> &jobs);

private:
    void loop(int cpu);
    std::vector<std::thread> threads_;
    std::mutex mu_;
    std::condition_variable wake_, done_;
    std::vector<std::function<void()>> *jobs_ = nullptr;
    size_t next_ = 0, finished_ = 0;
    bool stop_ = false;
};

struct Hand {
    int id = 0;
    V3 pts[21]{};               // as measured this frame; the tracker steers crops by these
    V3 smooth[21]{};            // filtered (One Euro, see Tracker::smooth): publish these
    bool has_pts = false;
    V3 prev_pts[21]{};          // pts of the previous update (for history extrapolation)
    int64_t prev_ns = 0;
    bool has_prev = false;
    double residual = -1;       // rms ray distance of the triangulation (m); -1: one view
    int nviews = 0;
    HandIdentity identity; // opt-in stable stereo handedness
    bool side_from_open = false; // optional identity evidence from an open hand
    double right_score = 0.5;   // the model's right-hand score (these images aren't mirrored)
    double scale = 1.0;         // this user's hand size / the model's world landmarks
    const Camera *cal_cam[2] = {nullptr, nullptr};   // FRAME_HANDS_MONO_DEPTH_CAL: per-camera single-view
    double cal_gain[2] = {1.0, 1.0};                 // depth calibration, stereo / single-view distance (EMA)
    int64_t seen_ns = 0;
    int64_t first_ns = 0;       // created at (FRAME_HANDS_GHOST_PRUNE)
    int64_t stereo_ns = 0;      // last update seen by two cameras (FRAME_HANDS_GHOST_PRUNE)
    int frames = 0;
    double speed = 0;           // palm centre, m/s, smoothed
    int64_t last_ns = 0;
    V3 last_palm{};
    V3 dpalm{};                 // the filter's palm velocity, m/s
    int64_t smooth_ns = 0;
    bool output_mono=false;
    V3 output_pts[21]{}, last_valid_output[21]{};
    int64_t last_valid_output_ns=0;
    const V3 *published() const { return output_mono ? output_pts : smooth; }
    int published_views() const { return output_mono ? 1 : nviews; }
    bool right() const { return right_score > 0.5; }
};

struct Stats {
    int palm_calls = 0, hand_calls = 0, sets = 0;
    double palm_ms = 0, hand_ms = 0, step_ms = 0;   // summed batch times
    int palm_batches = 0, hand_batches = 0;
    // why views and hands come and go
    int lost = 0;         // a tracked view's landmarks fell below min presence
    int handoff_miss = 0; // a view projected from the hand's 3D (new camera or retry) found no hand
    int dups = 0;         // the same hand twice in one camera
    int splits = 0;       // a hand's views disagreed in 3D and were split
    int created = 0, merged = 0, forgotten = 0;   // merged: views re-paired across cameras
    int ghosts = 0;       // same-handed duplicate tracks dropped (FRAME_HANDS_GHOST_PRUNE)
    // diagnostics: on stereo frames, each view's single-view palm distance / the stereo one
    std::vector<std::pair<int, double>> mono_ratio;   // (hand id, ratio)
};

// A hand the landmark model found in one camera (Tracker::views_now, Tracker::exhaustive).
struct Seen {
    std::string cam;
    int hand = 0;           // the tracker's hand; 0 in exhaustive()
    Roi roi;
    Landmarks lm;
    V3 wrist{};             // exhaustive(): single-view 3D guess at the model's hand size
};

class Tracker {
public:
    Tracker(const std::map<std::string, Camera> &cams, const Nets &nets, Pool &pool, int max_views = 2);
    // images: calibration name -> frame. Returns the hands seen in this set.
    std::vector<const Hand *> step(const std::map<std::string, Image> &images, int64_t t_ns);
    // Seconds until the next frame set is worth processing (30 Hz fast hands, 15 Hz slow, 5 Hz none).
    double interval() const;
    Stats stats;
    const std::vector<fhn::TrackerRigOutput>& rig_outputs() const {return rig_hook_.outputs();}
    void publish_rig(int64_t raw,uint64_t capture,uint64_t now){rig_hook_.publish(raw,capture,now);}
    void rig_capture_context(int64_t raw,uint64_t mono){rig_hook_.capture_context(raw,mono);}
    size_t views() const { return views_.size(); }
    std::vector<Seen> views_now() const;   // the views this step updated
    // Forget this camera's views, when it stops being tracked with (the lighting switched
    // cameras); views in cameras a set lacks otherwise wait for their next frame.
    void drop_camera(const std::string &name);
    // Every search tile in every camera, then landmarks on every palm: slow; for checking
    // what the scheduler misses (ft-handreplay --oracle).
    std::vector<Seen> exhaustive(const std::map<std::string, Image> &images);
    // Landmark presence a tracked view needs to stay (new views need min presence, 0.5). In
    // bright rooms the camera exposes for the room, the hands come out dim, and presence
    // dips under 0.5 for a frame at a time.
    void set_keep_presence(double p) { keep_presence_ = p; }
    // One view's 3D hand: each landmark along its ray, as far as how big the palm looks says
    // for a hand `scale` times the model's (Hand::scale). False if the palm is degenerate.
    bool single_view(const Camera &cam, const Landmarks &lm, double scale, V3 out[21]) const;

private:
    struct View {
        const Camera *cam;
        Roi roi;
        int hand = 0;           // 0: not assigned yet
        Landmarks lm;
        bool has_lm = false;
        int frames = 0;
        bool fresh = false;     // lm is from this step
        V2 hist[21]{};          // extrapolated previous pose projected into this camera (history input)
        bool has_hist = false;
    };
    // History conditioning (FRAME_HANDS_HISTORY=1 with a history-trained hand model).
    bool history_ = false;
    bool depth_cal_ = false;    // FRAME_HANDS_MONO_DEPTH_CAL=1: calibrate single-view depth against stereo
    void extrapolate(const Hand &hand, int64_t t_ns, V3 out[21]) const;
    void set_history(View &v, const Hand &hand, int64_t t_ns) const;
    struct Tile {
        const Camera *cam;
        V2 center;
        double size, rotation, weight, credit = 0;
    };
    void run_landmarks(const std::map<std::string, Image> &images, std::vector<View *> &views);
    bool hand_3d(Hand &hand, std::vector<View *> views, int64_t t_ns);
    double pair_cost(const View &a, const View &b) const;
    double size_misfit(const std::vector<const View *> &views, const V3 *pts) const;
    void associate();
    static void smooth(Hand &h, int64_t t_ns);
    bool inside(const Camera &cam, V2 uv) const;
    void add_tiles(const Camera &cam, double frac, int gx, int gy);

    std::map<std::string, const Camera *> cams_;
    const Nets &nets_;
    Pool &pool_;
    int max_views_, hand_budget_ = 4, search_budget_ = 3;
    double min_presence_ = 0.5, keep_presence_ = 0.5;
    std::vector<View> views_;
    std::map<int, Hand> hands_;
    JointIdentity joint_identity_;
    fhn::TrackerRigHook rig_hook_;
    std::vector<Tile> tiles_;
    int next_id_ = 1;
    int64_t last_ns_ = 0, search_ns_ = 0;
    // FRAME_HANDS_GHOST_PRUNE: where a dropped duplicate track was seen; palm proposals there are
    // ignored until `until`, so a forearm or monitor edge is not re-acquired every search.
    struct GhostZone {
        const Camera *cam;
        V2 centre;
        double size;
        int64_t until;
    };
    std::vector<GhostZone> ghost_zones_;
};
