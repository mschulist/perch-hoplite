from perch_hoplite.agile.xenocanto_known_species import XenocantoKnownSpecies
from absl.testing import absltest


class XenocantoKnownSpeciesTest(absltest.TestCase):
    xenocanto = XenocantoKnownSpecies("c4bc930547fe8264ecec91ee864d2d62a8a893fe", 10)

    def test_westan(self):
        westan_name = "Piranga ludoviciana"

        westan_recordings = self.xenocanto.get_recordings(westan_name)

        self.assertTrue(len(westan_recordings) > 0)
