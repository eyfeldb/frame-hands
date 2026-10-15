import base64,math,struct,unittest
from pc_transport import Converter,HEADER,HAND,unpack,map_point,rigid
I=[1,0,0,0,0,1,0,0,0,0,1,0]
def packet(seq=1,capture=990,render=1000,now=1001,points=None,budget=170):
    points=points or ([.1,.2,-.3]*21)
    hand=HAND.pack(7,1,capture,capture+budget,*points)
    data=HEADER.pack(b'FHMODEL2',2,592,seq,render,capture)+hand+bytes(HAND.size)
    return dict(state='native',version=1,native_now_ms=now,packet=base64.b64encode(data).decode(),head=I,mode=[1,1])
class Tests(unittest.TestCase):
    def test_identity_and_conservative_age(self):
        d=unpack(packet(),2000,2006);self.assertEqual(d['pose_age_ms'],4)
        data=Converter().convert(d,dict(valid=True,capture=I),2000,2006,99,2007)
        header=HEADER.unpack_from(data);h=HAND.unpack_from(data,40)
        self.assertEqual(header,(b'FHMODEL2',2,592,99,1999,1989))
        self.assertEqual(h[:4],(7,1,1989,2159))
        for got,want in zip(h[4:],[.1,.2,-.3]*21):self.assertAlmostEqual(got,want,places=6)
    def test_world_alignment(self):
        # Native and PC origins differ in rotation/translation; head-local point is unchanged.
        for angle in [i*.07 for i in range(90)]:
            c,s=math.cos(angle),math.sin(angle)
            native=rigid([c,0,s,3,0,1,0,2,-s,0,c,-4])
            pc=rigid([0,-1,0,-2,1,0,0,1,0,0,1,7])
            point=[.2,-.4,-.6]
            world=[sum(native[i][j]*point[j] for j in range(3))+native[i][3] for i in range(3)]
            expected=[sum(pc[i][j]*point[j] for j in range(3))+pc[i][3] for i in range(3)]
            for x,y in zip(map_point(world,native,pc),expected):self.assertAlmostEqual(x,y,places=12)
    def test_old_capture_never_renews(self):
        c=Converter();head=dict(valid=True,capture=I)
        first=c.convert(unpack(packet(),2000,2010),head,2000,2010,1,2011)
        later=c.convert(unpack(packet(seq=2,render=1020,now=1021),2025,2027),head,2025,2027,2,2028)
        self.assertEqual(HAND.unpack_from(first,40)[2:4],HAND.unpack_from(later,40)[2:4])
        late=c.convert(unpack(packet(seq=3,render=1190,now=1191),2190,2195),head,2190,2195,3,2196)
        self.assertEqual(HAND.unpack_from(late,40)[1],0)
    def test_duplicates_and_loss(self):
        c=Converter();d=unpack(packet(),2000,2006);head=dict(valid=True,capture=I)
        self.assertIsNone(c.convert(d,dict(valid=False),2000,2006,1,2007))
        self.assertIsNotNone(c.convert(d,head,2000,2006,2,2007))
        self.assertIsNone(c.convert(d,head,2000,2006,3,2008))
        self.assertIsNone(unpack(packet(),2000,2050))
        self.assertIsNone(unpack(packet(now=1100),2000,2005))
        self.assertIsNone(unpack(dict(state='waiting'),2000,2001))
    def test_bad_data(self):
        for points in ([float('nan')]*63,[float('inf')]*63):
            with self.assertRaises(ValueError):unpack(packet(points=points),2000,2005)
        p=packet();p['head']=[-1,0,0,0,0,1,0,0,0,0,1,0]
        with self.assertRaises(ValueError):unpack(p,2000,2005)
        p=packet();raw=base64.b64decode(p['packet']);p['packet']=base64.b64encode(raw[:316]+raw[40:316]).decode()
        with self.assertRaises(ValueError):unpack(p,2000,2005)
    def test_long_grace_conserves_expiry(self):
        c=Converter();head=dict(valid=True,capture=I)
        first=c.convert(unpack(packet(budget=350),2000,2006),head,2000,2006,1,2007)
        held=c.convert(unpack(packet(seq=2,render=1290,now=1291,budget=350),2290,2295),head,2290,2295,2,2296)
        self.assertEqual(HAND.unpack_from(held,40)[:4],HAND.unpack_from(first,40)[:4])
        expired=c.convert(unpack(packet(seq=3,render=1345,now=1346,budget=350),2345,2350),head,2345,2350,3,2351)
        self.assertEqual(HAND.unpack_from(expired,40)[1],0)
        with self.assertRaises(ValueError):unpack(packet(budget=351),2000,2005)
        self.assertIsNone(unpack(packet(budget=350),2000,2050)) # transport age unchanged
    def test_native_reboot(self):
        c=Converter();head=dict(valid=True,capture=I)
        self.assertIsNotNone(c.convert(unpack(packet(seq=500),2000,2005),head,2000,2005,1,2006))
        p=packet(seq=1,capture=90,render=100,now=101)
        self.assertIsNotNone(c.convert(unpack(p,5000,5005),head,5000,5005,2,5006))
if __name__=='__main__':unittest.main()
