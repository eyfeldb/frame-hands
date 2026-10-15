extends SceneTree
const Smoother = preload("res://smooth_hands.gd")
const PARENTS = [-1,0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19]

func hand(id: int, side: bool, x: float, angle: float=0.0) -> Dictionary:
    var p := PackedVector3Array()
    p.resize(21)
    p[0]=Vector3(x,0,-0.5)
    for finger in range(5):
        var first := 1+finger*4
        p[first]=p[0]+Vector3((finger-2)*0.018,0.05,0)
        p[first+1]=p[first]+Vector3(0,0.03,0)
        var direction := Vector3(0,cos(angle),sin(angle))
        p[first+2]=p[first+1]+direction*0.025
        p[first+3]=p[first+2]+direction*0.018
    return {"id":id,"right":side,"stereo":true,"points":p}

func _init() -> void:
    var filter = Smoother.new()
    var before: PackedVector3Array = hand(1,false,0).points
    filter.sample({"valid":true,"hands":[hand(1,false,0),hand(2,true,0.3)]},1.0/72)
    var target := hand(1,false,0.06,1.2)
    var observation := {"valid":true,"hands":[target,hand(2,true,0.3)]}
    var previous_x := 0.0
    for step in range(10):
        var result = filter.sample(observation,1.0/72)
        assert(result.hands.size()==2,"Smoothing must not reject a valid optical hand")
        var p: PackedVector3Array = result.hands[0].points
        assert(p[0].x>=previous_x and p[0].x<=0.060001,"A move-and-stop must approach the measured wrist monotonically without overshoot")
        previous_x=p[0].x
        for k in range(1,21):
            var parent: int = PARENTS[k]
            assert(absf(p[k].distance_to(p[parent])-target.points[k].distance_to(target.points[parent]))<0.000001,"Interpolation must preserve unchanged bone lengths")
    assert(previous_x>0.059,"Step response should settle promptly")
    assert(filter.sample({"valid":true,"hands":[hand(2,true,0.3)]},1.0/72).hands.size()==1,"This isolated comparison must preserve baseline loss behavior")
    assert(filter.sample({"valid":false},1.0/72).hands.is_empty())
    var changes := 0
    previous_x=-1
    filter=Smoother.new()
    for frame in range(144):
        var time := frame/72.0
        var capture := floorf(time*30.0)/30.0
        var result = filter.sample({"valid":true,"hands":[hand(1,false,capture*0.1)]},1.0/72)
        var x: float = result.hands[0].points[0].x
        if absf(x-previous_x)>0.000001: changes+=1
        assert(x<=capture*0.1+0.000001,"Smoothing must not extrapolate beyond the newest measurement")
        previous_x=x
    assert(changes>120)
    print("PASS: no overshoot, unchanged hand availability/labels, bone lengths, loss/reset, 30 Hz input with changing display poses=",changes,"/144")
    quit()
