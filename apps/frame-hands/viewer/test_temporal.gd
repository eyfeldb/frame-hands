extends SceneTree
const Temporal = preload("res://temporal_hands.gd")

func hand(id: int, right: bool, x: float) -> Dictionary:
    var points := PackedVector3Array()
    for k in range(21): points.append(Vector3(x+float(k%4)*0.005,float(k/4)*0.008,-0.5))
    return {"id":id,"right":right,"stereo":true,"points":points}

func observation(seq: int, time: float, hands: Array) -> Dictionary:
    return {"valid":true,"sequence":seq,"capture_seconds":time,"hands":hands}

func _init() -> void:
    var filter = Temporal.new()
    var first = observation(1,1.0,[hand(10,false,-0.2),hand(20,true,0.2)])
    assert(filter.sample(first,1.01,1.02).hands.size()==2)
    var next = observation(2,1.033333,[hand(10,true,-0.19),hand(20,false,0.21)])
    var a = filter.sample(next,1.04,1.045)
    assert(a.hands[0].id==10 and not a.hands[0].right,"A classifier-side flip must not swap the left identity")
    assert(a.hands[1].id==20 and a.hands[1].right)
    var b = filter.sample(next,1.05,1.059)
    assert(b.hands[0].points[0].x>a.hands[0].points[0].x,"Repeated 30 Hz observation must animate between display frames")
    assert(filter.accepted_updates==2,"Repeated render polls must not count as measurements")
    var missing = observation(3,1.066667,[hand(20,true,0.22)])
    assert(filter.sample(missing,1.08,1.09).hands.size()==2,"A missed observation must not immediately erase the hand")
    var faded = filter.sample(missing,1.20,1.21)
    assert(faded.hands[0].held and faded.hands[0].opacity<1.0)
    assert(not faded.hands[0].input_valid,"Held display poses must not be usable as input")
    var expired = filter.sample(observation(4,1.31,[]),1.31,1.32)
    assert(expired.hands.size()==1,"The older missing hand must expire after 250 ms")
    assert(filter.sample(observation(5,1.34,[]),1.34,1.35).hands.is_empty())
    var reacquire = filter.sample(observation(6,1.36,[hand(99,true,-0.18),hand(88,false,0.23)]),1.37,1.38)
    assert(reacquire.hands.size()==2)
    assert(reacquire.hands[0].id==99 and not reacquire.hands[0].right,"Spatial reacquisition should preserve the recent left identity across ID changes")
    assert(reacquire.hands[1].id==88 and reacquire.hands[1].right)
    assert(not filter.sample({"valid":false},1.38,1.39).valid)
    assert(filter.slots[0].is_empty() and filter.slots[1].is_empty(),"Invalid XR/packet data must clear held poses")
    filter = Temporal.new()
    filter.sample(observation(1,1.0,[hand(5,false,0.2)]),1.01,1.02)
    for k in range(4):
        filter.sample(observation(k+2,1.033333+k/30.0,[hand(5,true,0.2)]),1.04+k/30.0,1.05+k/30.0)
    var corrected = filter.sample(observation(9,1.2,[hand(5,true,0.2),hand(6,false,-0.2)]),1.21,1.22)
    assert(corrected.hands.size()==2 and corrected.hands[0].id==6 and corrected.hands[1].id==5,"Do not permanently lock an initially incorrect classifier label")
    filter = Temporal.new()
    filter.sample(observation(1,1.0,[hand(10,false,0)]),1.01,1.02)
    filter.sample(observation(2,1.033333,[hand(10,false,0.03)]),1.04,1.05)
    var stopped := observation(3,1.066667,[hand(10,false,0.03)])
    var stationary = filter.sample(stopped,1.07,1.11)
    assert(absf(stationary.hands[0].points[0].x-0.03)<0.001,"A new stationary measurement should stop prediction momentum")
    # A crossing retains IDs even when left/right spatial order reverses.
    filter = Temporal.new()
    for frame in range(21):
        var t := 2.0+frame/30.0
        var x := -0.10+frame*0.01
        var result = filter.sample(observation(frame,t,[hand(10,false,x),hand(20,true,-x)]),t+0.01,t+0.02)
        assert(result.hands.size()==2)
        assert(result.hands[0].id==10 and result.hands[1].id==20)
    # Sustained linear motion: positions change at 72 Hz from 30 Hz inputs.
    filter = Temporal.new()
    var changes := 0
    var last_x := -100.0
    for frame in range(144):
        var render_time := 4.0+frame/72.0
        var capture := 4.0+floorf((render_time-4.0)*30.0)/30.0
        var result = filter.sample(observation(int(round((capture-4.0)*30)),capture,[hand(1,false,(capture-4.0)*0.1)]),render_time,render_time+0.01)
        var x: float = result.hands[0].points[0].x
        if absf(x-last_x)>0.00001: changes+=1
        last_x=x
    assert(changes>120,"Display updates should exceed 60 per second over this two-second simulation")
    print("PASS: side-flip continuity, 72 Hz animation from 30 Hz observations, dropout fade/expiry, ID-change recovery, crossings, invalid-data reset; changed render positions=",changes)
    quit()
