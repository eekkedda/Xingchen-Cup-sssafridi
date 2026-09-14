import unittest
from decimal import Decimal as D
from analyze_results import parse_time, total_stats, report, load_data
from pathlib import Path

class TimingTests(unittest.TestCase):
    def test_ms_quantization(self):
        t = parse_time('1.40 ms')
        self.assertEqual((t.us, t.quantum_us, t.low, t.high),
                         (D('1400'), D('10'), D('1395'), D('1405')))
    def test_unicode_units(self):
        self.assertEqual(parse_time('3.90 μs').us, D('3.90'))
        self.assertEqual(parse_time('3.90 µs').quantum_us, D('0.01'))
    def test_invalid(self):
        for value in ('0 us', '-1 ms', 'NaN us', '5', 5, 'inf s'):
            with self.assertRaises(ValueError):
                parse_time(value)
    def test_history_totals(self):
        data=load_data(Path(__file__).with_name('历史跑分.json'))
        self.assertEqual(total_stats(data['Round6'])[0], D('4108.24'))
        self.assertEqual(total_stats(data['Round5'])[0], D('4120.42'))
        self.assertEqual(total_stats(data['Baseline'])[0], D('4273.32'))
    def test_round5_vs6_ambiguity(self):
        data=load_data(Path(__file__).with_name('历史跑分.json'))
        text=report(data, 'Round5', 'Round6')
        self.assertIn('显示量化覆盖', text)
        self.assertIn('+12.180', text)
    def test_coherent_total_median(self):
        runs=[{'1':parse_time('1.00 us'),'2':parse_time('9.00 us')},
              {'1':parse_time('9.00 us'),'2':parse_time('1.00 us')},
              {'1':parse_time('4.00 us'),'2':parse_time('4.00 us')}]
        self.assertEqual(total_stats(runs)[0], D('10'))

if __name__ == '__main__':
    unittest.main()
