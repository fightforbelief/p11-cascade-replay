import itertools
import random
import unittest
from statistics import fmean, pstdev
from replay import assignment, compare, erase_count


class ReplayTest(unittest.TestCase):
    def test_assignment(self):
        rng=random.Random(101)
        for size in range(2,7):
            for _ in range(5):
                matrix=[[rng.randrange(-9,10) for _ in range(size)] for _ in range(size)]
                def cost(perm):
                    return sum(matrix[i][j] for i,j in enumerate(perm))
                perms=itertools.permutations(range(size))
                chosen=assignment(matrix)
                self.assertEqual(sorted(chosen),list(range(size)))
                self.assertEqual(cost(chosen),min(map(cost,perms)))

    def test_identified_ranges_and_randomization_laws(self):
        cells=[([1,-1,.5],[0,2,7],-3),([0,1,-1],[6,1,3],1)]
        out=compare(cells)
        perms=list(itertools.permutations(range(3)))
        def value(pair):
            total=0
            for (weights,values,c),p in zip(cells,pair):
                total+=c+sum(w*values[j] for w,j in zip(weights,p))
            return total/6
        for name,samples in [('local',[value(p) for p in itertools.product(perms,repeat=2)]),
                             ('common',[value((p,p)) for p in perms])]:
            for k,fn in [('min',min),('max',max),('sd',pstdev)]:
                self.assertAlmostEqual(out[f'{name}_{k}'],fn(samples))
            self.assertAlmostEqual(out['product'],fmean(samples))
        self.assertAlmostEqual(out['identity'],value(((0,1,2),)*2))
        for edge in ('min','max'):
            perm=out[f'common_{edge}_permutation']
            self.assertAlmostEqual(value((perm,perm)),out[f'common_{edge}'])
        candidates=[sum(p!=perms[0] for p in pair) for pair in itertools.product(perms,repeat=2)
                    if value(pair)*out['identity']<=0]
        self.assertEqual(out['erase_questions'],min(candidates) if candidates else None)

    def test_global_cancellation(self):
        result=compare([([1,0],[1,0],0),([1,0],[0,1],0)])
        self.assertEqual(result['common_min'],result['common_max'])
        self.assertGreater(result['local_max'],result['local_min'])
        self.assertEqual(result['common_sd'],0)

    def test_erasure(self):
        self.assertIsNone(erase_count(-3,[1,1]))
        self.assertEqual(erase_count(-3,[1,2,0]),2)
        self.assertEqual(erase_count(0,[0]),0)

    def test_joint_utility(self):
        # A correct fallback costs 1, an incorrect one costs 0.
        w=[1,0]
        acc=compare([(w,[1,0],0)])
        cost=compare([(w,[1,0],0)])
        utility=compare([(w,[0,0],0)])
        self.assertEqual(utility['local_min'],0)
        self.assertLess(acc['local_min']-cost['local_max'],utility['local_min'])

if __name__=="__main__":
    unittest.main()
