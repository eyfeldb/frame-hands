"""Exact raw/baseline capture join; optional recorded delivery cadence only."""

def exact_capture_pairs(raw_frames, baseline_rows, delivered_only=False, stats=None):
    """Never reuse or retime a neighboring prediction for an unmatched image.

    Default requires a prediction for every selected raw image. Delivered mode
    skips unmatched raw images, retaining the baseline's recorded cadence.
    Missing raw images inside the selected span and unordered baselines fail.
    """
    if stats is None:stats={}
    stats.update(raw_frames_available=0,raw_frames_without_baseline=0,
                 raw_unmatched_inside_span=0,raw_before_first_baseline=0,
                 raw_after_last_baseline=0,matched_captures=0)
    source=iter(baseline_rows);previous=None
    def advance():
        nonlocal previous
        row=next(source,None)
        if row is not None:
            stamp=row['capture_raw_ns']
            if previous is not None and stamp<=previous:
                raise ValueError('Baseline captures must be strictly increasing')
            previous=stamp
        return row
    current=advance();first_raw=None;last_raw=None;first_match=None;last_match=None
    for index,raw in enumerate(raw_frames):
        seconds,stamp,images=raw
        if last_raw is not None and stamp<=last_raw:raise ValueError('Raw captures must be strictly increasing')
        last_raw=stamp
        if first_raw is None:first_raw=stamp
        stats['raw_frames_available']+=1
        while current is not None and current['capture_raw_ns']<stamp:
            if current['capture_raw_ns']>=first_raw:
                raise ValueError('Baseline has no exact raw capture')
            current=advance()
        if current is None or current['capture_raw_ns']!=stamp:
            if not delivered_only:raise ValueError('Baseline lacks exact current capture')
            stats['raw_frames_without_baseline']+=1
            continue
        if first_match is None:first_match=index
        last_match=index;stats['matched_captures']+=1
        yield raw,current
        current=advance()
    if first_match is not None:
        stats['raw_before_first_baseline']=first_match
        stats['raw_after_last_baseline']=stats['raw_frames_available']-last_match-1
        stats['raw_unmatched_inside_span']=last_match-first_match+1-stats['matched_captures']
    else:stats['raw_before_first_baseline']=stats['raw_frames_available']
