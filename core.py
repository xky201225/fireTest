import cv2
import numpy as np
import math
from typing import Dict, Any

from config import DetectionParams
import utils


def detect_flame(bgr: np.ndarray, params: DetectionParams) -> Dict[str, Any]:
    b = bgr[:, :, 0].astype(np.uint8)
    g = bgr[:, :, 1].astype(np.uint8)
    r = bgr[:, :, 2].astype(np.uint8)

    r_mean = float(np.mean(r))
    candidate = (
        (r.astype(np.float32) > r_mean)
        & (r > g)
        & (g > b)
        & (r > 200)
        & (g < 200)
        & (b < 100)
    )

    candidate_ratio = float(np.count_nonzero(candidate)) / float(candidate.size)
    if candidate_ratio < params.fire_candidate_ratio_threshold:
        empty = np.zeros(candidate.shape, dtype=np.uint8)
        return {
            "mask": empty,
            "candidate_ratio": candidate_ratio,
            "max_area": 0,
            "detected": False,
            "r_mean": r_mean,
        }

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (params.morph_kernel_size, params.morph_kernel_size)
    )
    mask_u8 = utils.to_uint8_mask(candidate)
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, kernel, iterations=1)
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel, iterations=1)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    max_area = 0
    if num_labels > 1:
        areas = stats[1:, cv2.CC_STAT_AREA]
        max_area = int(np.max(areas)) if areas.size else 0

    area_threshold = int(params.fire_area_ratio_threshold * mask_u8.size)
    detected = max_area > area_threshold
    return {
        "mask": mask_u8,
        "candidate_ratio": candidate_ratio,
        "max_area": max_area,
        "detected": detected,
        "r_mean": r_mean,
        "area_threshold": area_threshold,
    }


def detect_smoke(bgr: np.ndarray, params: DetectionParams) -> Dict[str, Any]:
    h, s, i = utils.rgb_to_hsi_from_bgr(bgr)

    b = bgr[:, :, 0].astype(np.int16)
    g = bgr[:, :, 1].astype(np.int16)
    r = bgr[:, :, 2].astype(np.int16)
    alpha = np.maximum(np.maximum(np.abs(r - g), np.abs(g - b)), np.abs(b - r))

    ref = (h >= 175.0) & (h <= 185.0) & (alpha >= 0) & (alpha <= 20)
    if np.count_nonzero(ref) == 0:
        empty = np.zeros((bgr.shape[0], bgr.shape[1]), dtype=np.uint8)
        return {
            "mask": empty,
            "ref_count": 0,
            "max_area": 0,
            "detected": False,
        }

    h_ref = float(np.mean(h[ref]))
    s_ref = float(np.mean(s[ref]))
    i_ref = float(np.mean(i[ref]))

    h_rad = np.deg2rad(h)
    h_ref_rad = math.radians(h_ref)

    v1 = (i_ref - i) ** 2
    v2 = (s_ref * math.cos(h_ref_rad) - s * np.cos(h_rad)) ** 2
    v3 = (s_ref * math.sin(h_ref_rad) - s * np.sin(h_rad)) ** 2
    d = np.sqrt(v1 + v2 + v3)

    smoke = d <= params.smoke_distance_threshold
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (params.morph_kernel_size, params.morph_kernel_size)
    )
    mask_u8 = utils.to_uint8_mask(smoke)
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, kernel, iterations=1)
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel, iterations=1)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    max_area = 0
    if num_labels > 1:
        areas = stats[1:, cv2.CC_STAT_AREA]
        max_area = int(np.max(areas)) if areas.size else 0

    area_threshold = int(params.smoke_area_ratio_threshold * mask_u8.size)
    detected = max_area > area_threshold
    return {
        "mask": mask_u8,
        "ref_count": int(np.count_nonzero(ref)),
        "max_area": max_area,
        "detected": detected,
        "h_ref": h_ref,
        "s_ref": s_ref,
        "i_ref": i_ref,
        "area_threshold": area_threshold,
        "distance_threshold": params.smoke_distance_threshold,
    }


def overlay_mask(bgr: np.ndarray, fire_mask_u8: np.ndarray, smoke_mask_u8: np.ndarray) -> np.ndarray:
    overlay = bgr.copy()
    fire = fire_mask_u8 > 0
    smoke = smoke_mask_u8 > 0

    overlay[fire] = (0, 0, 255)
    overlay[smoke] = (255, 0, 0)

    blended = cv2.addWeighted(bgr, 0.65, overlay, 0.35, 0.0)
    return blended
