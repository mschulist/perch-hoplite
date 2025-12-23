from perch_hoplite.agile.query_known_species import QueryKnownSpeciesInterface
from perch_hoplite.audio_io import load_xc_audio
import requests
import numpy as np
import time

XENO_CANTO_URL = "https://xeno-canto.org/api/3/recordings"


class XenocantoKnownSpecies(QueryKnownSpeciesInterface):
    def __init__(self, api_key: str, max_len: float = 60, sample_rate: float = 32000):
        self.api_key = api_key
        self.max_len = 60
        self.sample_rate = sample_rate

    def _get_xc_ids(self, species: str, call_type: str) -> list[str]:
        url = f'{XENO_CANTO_URL}?key={self.api_key}&query=sp:"{species}"+type:"{call_type}"+len:"1-{self.max_len}"'

        # continue trying if rate limited
        status_code = 0
        response = None
        while status_code != 200:
            response = requests.get(url)
            status_code = response.status_code
            print(status_code)
            if status_code == 200:
                break
            time.sleep(0.25)

        if not response:
            raise ValueError(
                f"Failed to get response from xeno-canto API for {species}: {call_type}."
            )
        response_json = response.json()

        return [rec["id"] for rec in response_json["recordings"]]

    def get_recordings(self, species, call_type=None, length=None, n_recordings=1):
        xc_ids = self._get_xc_ids(species, call_type)[:n_recordings]
        recs: dict[str, np.ndarray] = {
            xc_id: load_xc_audio(f"xc{xc_id}", self.sample_rate) for xc_id in xc_ids
        }
        return recs
