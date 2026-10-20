import concurrent.futures as cf
import sqlite3
import threading
import time
import unittest
from unittest.mock import patch
import paper_discovery as discovery
from test_paper_discovery import DiscoveryTests

class CapacityTests(unittest.TestCase):
    setUp=DiscoveryTests.setUp
    tearDown=DiscoveryTests.tearDown
    def test_four_filters_do_not_evict_first_and_pages_are_independent(self):
        original=discovery._compute_explore_people
        with patch.object(discovery,'_compute_explore_people',wraps=original) as compute:
            first=discovery.explore_people(self.library,topics=['ocean'],limit=1)
            for year in [2020,2021,2022,2023]: discovery.explore_people(self.library,from_year=year)
            second=discovery.explore_people(self.library,topics=['ocean'],offset=1,limit=1)
            self.assertEqual(compute.call_count,5)
            self.assertNotEqual(first['people'],second['people'])
            self.assertEqual(second['offset'],1)
    def test_parallel_identical_requests_share_one_compute(self):
        original=discovery._compute_explore_people
        entered=threading.Event(); release=threading.Event()
        def slow(*a,**kw): entered.set();release.wait(5);return original(*a,**kw)
        with patch.object(discovery,'_compute_explore_people',side_effect=slow) as compute, cf.ThreadPoolExecutor(8) as pool:
            jobs=[pool.submit(discovery.explore_people,self.library,topics=['ocean']) for _ in range(8)]
            self.assertTrue(entered.wait(5));time.sleep(.05);release.set()
            results=[j.result(5) for j in jobs]
            self.assertEqual(compute.call_count,1);self.assertTrue(all(r==results[0] for r in results))
    def test_external_correction_without_audit_event_invalidates_cache(self):
        first=discovery.explore_people(self.library,topics=['ocean'])
        with sqlite3.connect(self.state) as db: db.execute("INSERT INTO excluded_links VALUES('test-v1',1,'W1003')")
        second=discovery.explore_people(self.library,topics=['ocean'])
        self.assertEqual(first['total'],2);self.assertEqual(second['total'],1)
    def test_edit_during_compute_does_not_return_stale_result(self):
        original=discovery._compute_explore_people
        def edit(*a,**kw):
            value=original(*a,**kw)
            with sqlite3.connect(self.state) as db: db.execute("INSERT INTO excluded_links VALUES('test-v1',1,'W1003')")
            return value
        with patch.object(discovery,'_compute_explore_people',side_effect=edit):
            with self.assertRaisesRegex(ValueError,'review state changed'): discovery.explore_people(self.library,topics=['ocean'])
        self.assertFalse(self.library._explore_pending)
        self.assertEqual(discovery.explore_people(self.library,topics=['ocean'])['total'],1)
    def test_failed_computation_can_be_retried(self):
        with patch.object(discovery,'_compute_explore_people',side_effect=RuntimeError('temporary')):
            with self.assertRaises(RuntimeError): discovery.explore_people(self.library,topics=['ocean'])
        self.assertFalse(self.library._explore_pending)
        self.assertEqual(discovery.explore_people(self.library,topics=['ocean'])['total'],2)
    def test_oversized_result_is_served_without_retention(self):
        with patch.object(discovery,'_EXPLORE_CACHE_BYTES',1):
            self.assertEqual(discovery.explore_people(self.library,topics=['ocean'])['total'],2)
        self.assertFalse(self.library._explore_cache);self.assertEqual(self.library._explore_bytes,0)
    def test_lru_eviction_obeys_size_budget(self):
        with patch.object(discovery,'_EXPLORE_CACHE_ENTRIES',2):
            for year in [2020,2021,2022]: discovery.explore_people(self.library,from_year=year)
        self.assertEqual(len(self.library._explore_cache),2)
        self.assertEqual(self.library._explore_bytes,sum(self.library._explore_sizes.values()))

if __name__=='__main__': unittest.main()
