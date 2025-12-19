from dataclasses import dataclass


@dataclass(frozen=True)
class DetectionParams:
    fire_candidate_ratio_threshold: float = 0.01
    fire_area_ratio_threshold: float = 0.0005
    smoke_distance_threshold: float = 0.16
    smoke_area_ratio_threshold: float = 0.0235
    morph_kernel_size: int = 3
