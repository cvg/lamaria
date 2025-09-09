from __future__ import annotations

import json
import os
from typing import List, Optional, Tuple
from pathlib import Path
import numpy as np
import pycolmap

import projectaria_tools.core.mps as mps
from projectaria_tools.core import data_provider
from projectaria_tools.core.sensor_data import TimeDomain, TimeQueryOptions
from projectaria_tools.core.stream_id import StreamId

from ... import logger
from ..config.loaders import load_cfg
from ...utils.utils import find_closest_timestamp

class VrsToColmap:
    def __init__(self, cfg=None):
        self.cfg = cfg or load_cfg()
        self._init_io()

    def _init_io(self):
        self.left_img_dir = self.cfg.image_stream_path / "left"
        self.right_img_dir = self.cfg.image_stream_path / "right"

        self.empty_recons: pycolmap.Reconstruction = None
        self.keyframed_recons: pycolmap.Reconstruction = None

        self.vrs_provider = data_provider.create_vrs_data_provider(
            self.cfg.vrs_file_path
        )
        data_paths = mps.MpsDataPathsProvider(self.cfg.mps_path).get_data_paths()
        self.mps_data_provider = mps.MpsDataProvider(data_paths)

        self.left_cam_stream_id = StreamId(self.cfg.streams.left_cam_stream_id)
        self.right_cam_stream_id = StreamId(self.cfg.streams.right_cam_stream_id)
        self.imu_stream_id = StreamId(self.cfg.streams.imu_right_stream_id)

    def _list_images(self, folder: Path, ext: str =".jpg") -> List[str]:
        if not folder.is_dir():
            return []
        return sorted(n for n in folder.iterdir() if n.suffix == ext)

    def _ts_from_vrs(self, sid: StreamId) -> List[int]:
        """Timestamps in nanoseconds"""
        return sorted(self.vrs_provider.get_timestamps_ns(sid, TimeDomain.DEVICE_TIME))
    
    def _match_timestamps(self, max_diff=1e6) -> List[Tuple[int, int]]:
        L = self._ts_from_vrs(self.left_cam_stream_id)
        R = self._ts_from_vrs(self.right_cam_stream_id)
        matched = []

        if len(L) < len(R):
            for l in L:
                r = find_closest_timestamp(R, l, max_diff)
                if r is not None:
                    matched.append((l, r))
        else:
            for r in R:
                l = find_closest_timestamp(L, r, max_diff)
                if l is not None:
                    matched.append((l, r))
        return matched
    
    def _load_cameras(self) -> None:
        pass

