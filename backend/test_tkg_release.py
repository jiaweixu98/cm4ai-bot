import copy
import json
import tempfile
import unittest
from pathlib import Path
from scripts.tkg_release import repair_layout, enrich_affiliation
from tkg_publications import affiliation_state, displayable_affiliations, resolve_snapshot


class ReleaseTests(unittest.TestCase):
    def test_neighbor_rows_follow_identity_through_merge(self):
        source = {'ids': [10, 30, 40], 'neighbors': {'neighbors': {'0': [1, 2, 3], '1': [0, 1], '2': [1, 2]}}}
        result = repair_layout(copy.deepcopy(source), [10, 20, 30, 40], {20: 30})
        self.assertEqual(result['neighbors']['neighbors'], {'0': [1, 2], '1': [0], '2': [1]})
        self.assertEqual(source['neighbors']['neighbors']['0'], [1, 2, 3])

    def test_wrong_index_space_fails_closed(self):
        with self.assertRaises(ValueError):
            repair_layout({'ids': [30, 10], 'neighbors': {}}, [10, 30], {})

    def test_job_titles_follow_the_recorded_period(self):
        history=[{'start_year':2000,'end_year':2005,'role_title':'Assistant professor','ror_id':'ror'},
                 {'start_year':2006,'end_year':None,'role_title':'Professor','ror_id':'ror'}]
        row={'institution':'Institute','start_year':2000,'end_year':2005}
        enrich_affiliation(row,history)
        self.assertEqual(row['role_title'],'Assistant professor')
        unknown={'institution':'Institute','start_year':1990,'end_year':1995}
        enrich_affiliation(unknown,history)
        self.assertNotIn('role_title',unknown)

    def test_affiliation_states_and_no_single_token_guess(self):
        history = [{'institution': 'Harvard University', 'verification': 'verified_orcid'},
                   {'institution': 'University of Alabama at Birmingham', 'verification': 'inferred'}]
        self.assertEqual(affiliation_state('Harvard University', '', history), 'confirmed')
        self.assertEqual(affiliation_state('University of Alabama at Birmingham', '', history), 'in_history')
        self.assertEqual(affiliation_state('Harvard Medical School', '', history), 'unrecognised')
        self.assertEqual(affiliation_state('', '', history), 'unknown')

    def test_pointer_requires_matching_version(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            (base / 'release').mkdir()
            (base / 'active-snapshot.json').write_text(json.dumps({'path': 'release', 'snapshot_version': 'v1'}))
            (base / 'release/snapshot_manifest.json').write_text(json.dumps({'snapshot_version': 'v2'}))
            with self.assertRaises(RuntimeError):
                resolve_snapshot(directory)
            (base / 'release/snapshot_manifest.json').write_text(json.dumps({'snapshot_version': 'v1'}))
            self.assertEqual(resolve_snapshot(directory), str(base / 'release'))

    def test_display_filter_and_tolerant_numbers(self):
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parent / 'scripts'))
        from build_tkg_snapshot import to_int
        row={'affiliations':[{'institution':'A','source':'orcid'},{'institution':'B','source':'openalex_low_conf'},{'institution':' '}]}
        self.assertEqual([a['institution'] for a in displayable_affiliations(row)],['A'])
        self.assertEqual([to_int(v) for v in ('2020','2020.0','',None,' 7 ')],[2020,2020,None,None,7])

    def test_pending_edits_are_retained_for_review_but_not_displayed(self):
        row={'affiliations':[{'institution':'ORCID Institute','source':'orcid','is_current':True}]}
        reviews=[{'group_id':'orcid institute','state':'pending','override':{'institution':'Edited Institute'}},
                 {'group_id':'addition','state':'pending','override':{'institution':'Added Institute'}}]
        self.assertEqual([a['institution'] for a in displayable_affiliations(row,reviews)],['ORCID Institute'])
        reviews[1]['state']='approved'
        self.assertEqual([a['institution'] for a in displayable_affiliations(row,reviews)],['ORCID Institute','Added Institute'])

    def test_refresh_uses_the_display_filter(self):
        import inspect
        from scripts import tkg_release
        self.assertIn('displayable_affiliations(row)',inspect.getsource(tkg_release.refresh_release))


if __name__ == '__main__':
    unittest.main()
