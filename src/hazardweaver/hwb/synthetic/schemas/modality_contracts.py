"""Vendored PyHazards modality shapes (from PYHAZARDS_DATA_ONTOLOGY.md)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

# confirmed_by_audit: micro shapes from capability audit


@dataclass(frozen=True)
class GraphStreamflowContract:
    history: int = 4
    nodes: int = 6
    features: int = 2

    @property
    def x_shape(self) -> Tuple[int, int, int, int]:
        return (-1, self.history, self.nodes, self.features)

    @property
    def adj_shape(self) -> Tuple[int, int]:
        return (self.nodes, self.nodes)

    @property
    def y_shape(self) -> Tuple[int, int, int]:
        return (-1, self.nodes, 1)


@dataclass(frozen=True)
class TCTrackContract:
    history: int = 6
    features: int = 8

    @property
    def shape(self) -> Tuple[int, int, int]:
        return (-1, self.history, self.features)


@dataclass(frozen=True)
class RasterContract:
    height: int = 16
    width: int = 16
    channels: int = 1

    @property
    def shape(self) -> Tuple[int, int, int, int]:
        return (-1, self.channels, self.height, self.width)


GRAPH_STREAMFLOW = GraphStreamflowContract()
TC_TRACK = TCTrackContract()
RASTER_16 = RasterContract()
