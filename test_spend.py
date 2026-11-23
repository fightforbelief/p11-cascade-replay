import itertools
import unittest
from spend_replay import GateReplay,certificate,direct_value,efficient
from published_frontiers import best,direct_options
from fractions import Fraction as F


class SpendTests(unittest.TestCase):
    def test_interior_minimum(self):
        # At each deterministic endpoint the excess is one. Mixing alignments
        # reaches the direct-policy kink, where the excess falls to minus one.
        vertices=[(1,1),(3,5)]
        direct=[(1,0),(2,4)]
        def oracle(price,minimize):
            return (min if minimize else max)(vertices,key=lambda p:p[1]-price*p[0])
        c=certificate(direct,oracle)
        self.assertAlmostEqual(c["minimum_pp"],-1.)
        self.assertAlmostEqual(c["maximum_pp"],1.)
        self.assertAlmostEqual(c["minimum_dual_pp"],-1.)

    def test_oracles_against_full_enumeration(self):
        archive={"a":([(1,1),(4,0),(2,0)],[(4,0),(2,1),(5,1)]),
                 "b":([(4,1),(1,1),(5,0)],[(2,0),(1,0),(7,1)])}
        qs=sorted(archive);r=GateReplay(archive,qs,2,1)
        perms=list(itertools.permutations(range(3)))
        for common in (True,False):
            alignments=[(p,p) for p in perms] if common else itertools.product(perms,repeat=2)
            vertices=[]
            for align in alignments:
                c=a=0
                for q,p in zip(qs,align):
                    first,second=archive[q]
                    for i,(t,y) in enumerate(first):
                        t2,y2=second[p[i]]
                        c+=t+(t2 if t>2 else 0)
                        a+=100*(y2 if t>2 else y)
                vertices.append((c/6,a/6))
            oracle=r.common if common else r.local
            for lam in (0,.5,2,10,100):
                for minimize in (True,False):
                    def u(p):
                        return p[1]-lam*p[0]
                    self.assertAlmostEqual(u(oracle(lam,minimize)),(min if minimize else max)(map(u,vertices)))
            # Independent primal exhaustive solution: all segment/kink intersections.
            direct=[(0,10),(7,90),(12,94)]
            f=lambda p:p[1]-direct_value(efficient(direct),p[0])
            vals=[f(p) for p in vertices]
            for p,q in itertools.combinations(sorted(set(vertices)),2):
                for x,y in direct:
                    if p[0]<x<q[0]:
                        vals.append(f((x,p[1]+(q[1]-p[1])*(x-p[0])/(q[0]-p[0]))))
            bounds=certificate(direct,oracle)
            self.assertAlmostEqual(min(vals),bounds["minimum_pp"])
            self.assertAlmostEqual(max(vals),bounds["maximum_pp"])

    def test_direct_frontier(self):
        points={"a":(F(1),F(1,5)),"b":(F(2),F(2,5)),"c":(F(2),F(3,5)),"d":(F(3),F(1,2))}
        for i in range(10,41):
            b=F(i,10)
            self.assertAlmostEqual(float(best(direct_options(points,b))[0]),direct_value(efficient(points.values()),b))
        self.assertIsNone(best(direct_options(points,F(1,2))))
