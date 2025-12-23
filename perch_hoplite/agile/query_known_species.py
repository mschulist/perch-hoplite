import abc
import enum
import numpy as np


class QueryKnownSpeciesInterface(abc.ABC):
    @abc.abstractmethod
    def get_recordings(
        self,
        species: str,
        call_type: enum.Enum | None = None,
        length: int | None = None,
        n_recordings: int = 1,
    ) -> dict[str, np.ndarray]:
        """
        Gets known recordings of the given species.

        :param species: Species code, implemenation specific
        :type species: str
        :param call_type: Type of call, also implementation specific
        :type call_type: enum.Enum | None
        :param length: The length of the recording
        :type length: int | None
        :param n_recordings: The number of recordings to get
        :type n_recordings: int

        :return: Dictionary mapping the recording id to the recording np.ndarray
        :rtype: dict[str, np.ndarray]
        """
