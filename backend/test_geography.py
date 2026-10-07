import json
import os
import tempfile
import unittest
from unittest.mock import patch

import geography
from geography import Geography, normalize

US = {'country_code': 'US', 'country_name': 'United States', 'subdivision_code': 'CA',
      'subdivision_name': 'California', 'city': 'San Diego', 'continent_code': 'NA', 'continent_name': 'North America'}
CA = {'country_code': 'CA', 'country_name': 'Canada', 'subdivision_code': 'ON', 'subdivision_name': 'Ontario',
      'city': 'Toronto', 'continent_code': 'NA', 'continent_name': 'North America'}
DE = {'country_code': 'DE', 'country_name': 'Germany', 'subdivision_code': 'BY', 'subdivision_name': 'Bavaria',
      'city': 'Munich', 'continent_code': 'EU', 'continent_name': 'Europe'}
NL = {'country_code': 'NL', 'country_name': 'The Netherlands', 'subdivision_code': 'NH',
      'subdivision_name': 'North Holland', 'city': 'Amsterdam', 'continent_code': 'EU', 'continent_name': 'Europe'}


class GeographyTests(unittest.TestCase):
    def hits(self, typed):
        places = Geography({}, live=None)
        parts = places.parse(typed)
        return [name for name, row in (('US', US), ('CA', CA), ('DE', DE), ('NL', NL)) if parts and places.match(row, parts)]

    def test_normalisation(self):
        self.assertEqual(normalize('  U.S. '), 'us')
        self.assertEqual(normalize('U.S.A.'), 'usa')
        self.assertEqual(normalize('Türkiye'), 'turkiye')
        self.assertEqual(normalize('Côte d’Ivoire'), 'cote d ivoire')

    def test_countries_codes_and_aliases(self):
        for typed in ('US', 'U.S.', 'USA', 'america', 'United States', 'the united states', 'usa '):
            self.assertEqual(self.hits(typed), ['US'], typed)
        for typed in ('Germany', 'DE', 'DEU', 'germany'):
            self.assertEqual(self.hits(typed), ['DE'], typed)
        self.assertEqual(self.hits('Netherlands'), ['NL'])
        self.assertEqual(self.hits('The Netherlands'), ['NL'])
        self.assertEqual(self.hits('Holland'), ['NL'])
        self.assertEqual(self.hits('Canada'), ['CA'])

    def test_two_letter_code_alone_is_a_country(self):
        self.assertEqual(self.hits('CA'), ['CA'])      # Canada, not California
        self.assertEqual(self.hits('US-CA'), ['US'])    # explicit state
        self.assertEqual(self.hits('Boston, MA'), [])
        self.assertEqual(self.hits('San Diego, CA'), ['US'])

    def test_subdivisions_cities_and_continents(self):
        self.assertEqual(self.hits('California'), ['US'])
        self.assertEqual(self.hits('Ontario'), ['CA'])
        self.assertEqual(self.hits('San Diego'), ['US'])
        self.assertEqual(self.hits('North America'), ['US', 'CA'])
        self.assertEqual(self.hits('Europe'), ['DE', 'NL'])
        self.assertEqual(self.hits('Bay Area'), [])
        self.assertEqual(self.hits(''), [])

    def test_table_loading_and_newest_default(self):
        with tempfile.TemporaryDirectory() as directory:
            old = os.path.join(directory, 'ror_locations-v1.json')
            new = os.path.join(directory, 'ror_locations-v2.json')
            for path, version in ((old, 'v1'), (new, 'v2')):
                with open(path, 'w') as handle:
                    json.dump({'manifest': {'dump_version': version}, 'locations': {'012345678': US}}, handle)
            os.utime(old, (1, 1))
            with patch.object(geography, '_ROOT', os.path.dirname(directory)), \
                    patch.dict(os.environ, {'ROR_LOCATIONS_PATH': ''}):
                self.assertIsNone(geography._reference_path())  # no data/reference under that root
            with patch.dict(os.environ, {'ROR_LOCATIONS_PATH': new}):
                places = Geography.load()
            self.assertEqual(places.version, 'v2')
            found, complete = places.resolve(['https://ror.org/012345678', 'not-a-ror'])
            self.assertEqual(list(found), ['012345678']); self.assertTrue(complete)

    def test_live_lookups_are_bounded_and_failures_cached(self):
        calls = []
        places = Geography({}, live=lambda key: calls.append(key) or None)
        rors = [f'0abcdef{i:02d}' for i in range(geography.LIVE_LOOKUP_LIMIT + 5)]
        found, complete = places.resolve(rors)
        self.assertEqual((found, complete, len(calls)), ({}, False, geography.LIVE_LOOKUP_LIMIT))
        places.resolve(rors[:geography.LIVE_LOOKUP_LIMIT])
        self.assertEqual(len(calls), geography.LIVE_LOOKUP_LIMIT)  # cached, including the misses


if __name__ == '__main__':
    unittest.main()
