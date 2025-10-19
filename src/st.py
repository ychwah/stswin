# text_aware_preproc.py
import numpy as np
import cv2
from typing import Dict, Tuple

# ---- Robust linear solver fallback (pypardiso -> scipy) ----------------------------------
try:
    from pypardiso import spsolve  # fast if available
except Exception:
    from scipy.sparse.linalg import spsolve

from scipy.sparse import spdiags, csr_matrix

# ==========================================================================================
#                         RTV-style structure extraction (improved)
# ==========================================================================================

def _as_float01(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(np.float32)
    if arr.max() > 1.0:
        arr = arr / 255.0
    return arr


def _to_3ch(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 2:
        return np.repeat(arr[..., None], 3, axis=2)
    if arr.ndim == 3 and arr.shape[2] == 1:
        return np.repeat(arr, 3, axis=2)
    return arr


def conv2_sep(im: np.ndarray, sigma: float) -> np.ndarray:
    ksize = max(int(round(5 * sigma)), 1)
    if ksize % 2 == 0:
        ksize += 1
    g = cv2.getGaussianKernel(ksize, sigma)
    ret = cv2.filter2D(im, -1, g)
    ret = cv2.filter2D(ret, -1, g.T)
    return ret


def lpfilter(FImg: np.ndarray, sigma: float) -> np.ndarray:
    FBImg = np.zeros_like(FImg, dtype=np.float32)
    for ic in range(FImg.shape[2]):
        FBImg[:, :, ic] = conv2_sep(FImg[:, :, ic], sigma)
    return FBImg


def computeTextureWeights(fin: np.ndarray, sigma: float, sharpness: float) -> Tuple[np.ndarray, np.ndarray]:
    # fin expected in [0,1], 3ch
    fx = np.diff(fin, axis=1)
    fx = np.pad(fx, ((0, 0), (0, 1), (0, 0)), mode="constant")
    fy = np.diff(fin, axis=0)
    fy = np.pad(fy, ((0, 1), (0, 0), (0, 0)), mode="constant")

    vareps_s = float(sharpness)
    vareps = 1e-3

    wto = np.maximum(np.sum(np.sqrt(fx * fx + fy * fy), axis=2) / fin.shape[2], vareps_s) ** -1

    fbin = lpfilter(fin, sigma)
    gfx = np.diff(fbin, axis=1)
    gfx = np.pad(gfx, ((0, 0), (0, 1), (0, 0)), mode="constant")
    gfy = np.diff(fbin, axis=0)
    gfy = np.pad(gfy, ((0, 1), (0, 0), (0, 0)), mode="constant")

    wtbx = np.maximum(np.sum(np.abs(gfx), axis=2) / fin.shape[2], vareps) ** -1
    wtby = np.maximum(np.sum(np.abs(gfy), axis=2) / fin.shape[2], vareps) ** -1

    retx = wtbx * wto
    rety = wtby * wto
    retx[:, -1] = 0.0
    rety[-1, :] = 0.0
    return retx, rety


def solveLinearEquation(IN: np.ndarray, wx: np.ndarray, wy: np.ndarray, lambda_: float) -> np.ndarray:
    r, c, ch = IN.shape
    k = r * c

    dx = -lambda_ * wx.ravel(order="F")
    dy = -lambda_ * wy.ravel(order="F")

    B = np.vstack((dx, dy))
    d = [-r, -1]
    A = spdiags(B, d, k, k)

    e = dx
    w = np.pad(dx[:-r], (r, 0), "constant")
    s = dy
    n = np.pad(dy[:-1], (1, 0), "constant")
    D = 1.0 - (e + w + s + n)
    A = A + A.T + spdiags(D, 0, k, k)

    A = csr_matrix(A, dtype=np.float64)

    OUT = np.zeros_like(IN, dtype=np.float32)
    for i in range(ch):
        tin = IN[:, :, i].ravel(order="F").astype(np.float64)
        tout = spsolve(A, tin)
        OUT[:, :, i] = tout.reshape((r, c), order="F").astype(np.float32)

    return OUT


def tsmooth(
    I: np.ndarray,
    lambda_: float = 0.02,
    sigma: float = 2.0,
    sharpness: float = 0.02,
    maxIter: int = 4,
    dec: float = 2.0
) -> np.ndarray:
    """
    Texture smoothing (RTV-like). Accepts 1ch or 3ch input in [0..255] or [0..1].
    Returns float32 in [0..1].
    Defaults are tuned slightly more aggressively for text extraction vs the original.
    """
    I = _as_float01(I)
    if I.ndim == 2:
        I = I[..., None]
    x = I.copy()
    sigma_iter = float(sigma)
    lam = float(lambda_) * 0.5  # mild
    for _ in range(maxIter):
        wx, wy = computeTextureWeights(_to_3ch(x), sigma_iter, sharpness)
        x = solveLinearEquation(I, wx, wy, lam)
        sigma_iter = max(sigma_iter / dec, 0.5)
    return np.clip(x.squeeze(), 0.0, 1.0)


# ==========================================================================================
#                         Text-aware feature construction
# ==========================================================================================

def _normalize01(arr: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    arr = arr.astype(np.float32)
    mn, mx = float(np.min(arr)), float(np.max(arr))
    if mx - mn < eps:
        return np.zeros_like(arr, dtype=np.float32)
    return (arr - mn) / (mx - mn)


def _scharr_mag(img_f01: np.ndarray) -> np.ndarray:
    g = (img_f01 * 255.0).astype(np.uint8)
    g = cv2.GaussianBlur(g, (0, 0), 0.8)
    gx = cv2.Scharr(g, cv2.CV_32F, 1, 0)
    gy = cv2.Scharr(g, cv2.CV_32F, 0, 1)
    mag = np.sqrt(gx * gx + gy * gy)
    return _normalize01(mag)


def _blackhat_multiscale(img_f01: np.ndarray, base: int = 3, steps: int = 4) -> np.ndarray:
    """Enhance dark strokes on light background using morphological black-hat at multiple scales."""
    g = (img_f01 * 255.0).astype(np.uint8)
    agg = np.zeros_like(g, dtype=np.float32)
    for i in range(steps):
        k = base + 2 * i  # 3,5,7,9...
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
        bh = cv2.morphologyEx(g, cv2.MORPH_BLACKHAT, kernel)
        agg = np.maximum(agg, bh.astype(np.float32))
    return _normalize01(agg)


def _whitehat_multiscale(img_f01: np.ndarray, base: int = 3, steps: int = 4) -> np.ndarray:
    """Enhance light strokes on dark background using white-hat (top-hat)."""
    g = (img_f01 * 255.0).astype(np.uint8)
    agg = np.zeros_like(g, dtype=np.float32)
    for i in range(steps):
        k = base + 2 * i
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
        wh = cv2.morphologyEx(g, cv2.MORPH_TOPHAT, kernel)
        agg = np.maximum(agg, wh.astype(np.float32))
    return _normalize01(agg)


def _min_area_for_mser(shape_hw: Tuple[int, int]) -> int:
    h, w = shape_hw[:2]
    return max(30, (h * w) // 5000)  # adaptive minimum


def _mser_mask(img_f01: np.ndarray) -> np.ndarray:
    """Detect extremal regions (both polarities) to produce a text-likelihood mask."""
    g = (img_f01 * 255.0).astype(np.uint8)
    mser = cv2.MSER_create(_min_area_for_mser(g.shape), int(0.25 * g.size), int(0.8 * g.size))
    mask = np.zeros_like(g, dtype=np.uint8)

    # Dark-on-light
    regions, _ = mser.detectRegions(g)
    for pts in regions:
        cv2.fillPoly(mask, [pts], 255)

    # Light-on-dark
    regions, _ = mser.detectRegions(255 - g)
    for pts in regions:
        cv2.fillPoly(mask, [pts], 255)

    # Clean-up
    mask = cv2.medianBlur(mask, 3)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
    return mask.astype(np.float32) / 255.0


def _clahe01(gray01: np.ndarray) -> np.ndarray:
    tile = 8
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(tile, tile))
    g = (gray01 * 255.0).astype(np.uint8)
    e = clahe.apply(g)
    return e.astype(np.float32) / 255.0


def structure_texture_decompose(bgr: np.ndarray, rtv_lambda: float = 0.02) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns (L, S, T): luminance, structure, absolute texture; all in [0..1].
    """
    # Convert to L channel (perceptual)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    L = lab[:, :, 0].astype(np.float32) / 255.0
    L_eq = _clahe01(L)  # improve contrast for weak strokes
    S = tsmooth(L_eq, lambda_=rtv_lambda, sigma=2.0, sharpness=0.02, maxIter=3)
    T = np.abs(L_eq - S)
    return L_eq, S, T


def build_text_feature_stack(bgr: np.ndarray, rtv_lambda: float = 0.02) -> Dict[str, np.ndarray]:
    """
    Compute a set of machine-oriented channels for OCR models.
    Returns a dict of float32 arrays in [0..1].
    Keys:
      - L   : CLAHE-equalized luminance
      - S   : structure (texture-suppressed)
      - T   : |texture| residual
      - E   : Scharr gradient magnitude (on S)
      - BH  : black-hat (dark text saliency)
      - WH  : white-hat (light text saliency)
      - MSER: extremal-region mask
      - OCRPrime3 : 3-ch pseudo-RGB = [S, BH, E]
      - OCRStack4 : 4-ch stack = [S, T, E, MSER]
    """
    L, S, T = structure_texture_decompose(bgr, rtv_lambda=rtv_lambda)
    E = _scharr_mag(S)
    BH = _blackhat_multiscale(S)
    WH = _whitehat_multiscale(S)
    MSER = _mser_mask(L)

    # 3-channel "prime" representation that many RGB models accept
    OCRPrime3 = np.dstack([S, BH, E]).astype(np.float32)
    # 4-channel stack if your model can ingest >3 channels
    OCRStack4 = np.dstack([S, T, E, MSER]).astype(np.float32)

    return {
        "L": L.astype(np.float32),
        "S": S.astype(np.float32),
        "T": T.astype(np.float32),
        "E": E.astype(np.float32),
        "BH": BH.astype(np.float32),
        "WH": WH.astype(np.float32),
        "MSER": MSER.astype(np.float32),
        "OCRPrime3": OCRPrime3,
        "OCRStack4": OCRStack4,
    }


def save_feature_previews(prefix: str, feats: Dict[str, np.ndarray]) -> None:
    """
    Save PNG previews (uint8) for quick inspection.
    Multichannel tensors are saved as PNG with channel mixing (OCRStack4 -> first 3 channels).
    """
    def to_u8(x: np.ndarray) -> np.ndarray:
        x = np.clip(x, 0.0, 1.0)
        if x.ndim == 2:
            return (x * 255.0).astype(np.uint8)
        if x.ndim == 3 and x.shape[2] == 3:
            return (x * 255.0).astype(np.uint8)
        if x.ndim == 3 and x.shape[2] == 4:
            # visualize first 3 channels
            return (x[:, :, :3] * 255.0).astype(np.uint8)
        return (x * 255.0).astype(np.uint8)

    for k, v in feats.items():
        out = to_u8(v)
        cv2.imwrite(f"{prefix}_{k}.png", out)


# ==========================================================================================
#                                      CLI
# ==========================================================================================

def _read_image(path: str) -> np.ndarray:
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(path)
    return bgr


def run_cli(in_path: str, out_prefix: str, rtv_lambda: float = 0.02) -> None:
    bgr = _read_image(in_path)
    feats = build_text_feature_stack(bgr, rtv_lambda=rtv_lambda)
    save_feature_previews(out_prefix, feats)


if __name__ == "__main__":
    import argparse
    from pathlib import Path as _Path
    ap = argparse.ArgumentParser(description="Text-aware structure/texture decomposition for OCR preproc")
    ap.add_argument("image", help="Path to input image")
    ap.add_argument("--out", default="out/ocr", help="Output prefix (folders will be created as needed)")
    ap.add_argument("--lambda", dest="rtv_lambda", type=float, default=0.02, help="RTV lambda (higher -> smoother)")

    args = ap.parse_args()

    out_dir = _Path(args.out).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    run_cli(args.image, args.out, rtv_lambda=args.rtv_lambda)
