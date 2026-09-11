"""Fixed 41-column appearance/context features for one instance mask.

Masks are binary and independent: ring pixels are the disk(5) dilation minus
the mask, without excluding other instances. Perimeter is skimage regionprops'
standard perimeter estimate on the binary mask; all degenerate values use
finite zero/one defaults. RGB is float in [0,1].
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage
from skimage.feature import local_binary_pattern
from skimage.filters import sobel
from skimage.measure import regionprops


class PatchAppearanceContext:
    """Patch-level arrays shared by all instance masks in one RGB patch."""

    def __init__(self, rgb: np.ndarray) -> None:
        im = np.asarray(rgb, dtype=np.float32)
        if im.ndim != 3 or im.shape[2] != 3:
            raise ValueError("RGB shape")
        self.rgb = np.nan_to_num(im, nan=0.0, posinf=1.0, neginf=0.0).clip(0, 1)
        self.gray = self.rgb.mean(2)
        self.sobel = sobel(self.gray)
        self.lbp = local_binary_pattern(
            np.round(self.gray * 255).astype(np.uint8), 8, 1, method="uniform"
        )
        yy, xx = np.ogrid[-5:6, -5:6]
        self.disk5 = (xx * xx + yy * yy) <= 25

    def features(self, mask: np.ndarray) -> np.ndarray:
        return _features_from_context(np.asarray(mask, dtype=bool), self)


def _features_from_context(m: np.ndarray, ctx: PatchAppearanceContext) -> np.ndarray:
    im, gray, grad = ctx.rgb, ctx.gray, ctx.sobel
    if im.shape[:2] != m.shape:
        raise ValueError("mask/RGB shape")
    pix = im[m]
    graypix = gray[m]
    if m.any():
        rp = regionprops(m.astype(np.uint8))[0]; area=float(rp.area); per=float(rp.perimeter)
        minr,minc,maxr,maxc=rp.bbox; h=maxr-minr; w=maxc-minc
        ecc=float(rp.eccentricity); sol=float(rp.solidity); ext=area/max(1,h*w)
        major=float(rp.axis_major_length)/256; minor=float(rp.axis_minor_length)/256
    else: area=per=0.; h=w=1; ecc=sol=ext=major=minor=0.
    shape=[per/np.sqrt(max(area,1.0)), 4*np.pi*area/max(per*per,1.0), np.log(max(h/w,1e-6)), ext,ecc,sol,major,minor]
    ch=[]
    for c in range(3): ch += [float(np.percentile(pix[:,c],q)) if len(pix) else 0.0 for q in (10,50,90)]
    ch += [float(np.percentile(graypix,q)) if len(pix) else 0.0 for q in (10,50,90)] + [float(np.percentile(graypix,75)-np.percentile(graypix,25)) if len(pix) else 0.0]
    gp=grad[m]; ch += [float(np.mean(gp)) if len(gp) else 0.,float(np.std(gp)) if len(gp) else 0.,float(np.percentile(gp,90)) if len(gp) else 0.]
    hist=np.bincount(ctx.lbp[m].astype(int),minlength=10)[:10].astype(np.float64) if m.any() else np.zeros(10); hist/=max(hist.sum(),1.)
    ring=ndimage.binary_dilation(m,structure=ctx.disk5)&~m
    rpix=im[ring]; rg=gray[ring]; nm=im[m].mean(0) if m.any() else np.zeros(3); rm=rpix.mean(0) if len(rpix) else nm
    base_gray = float(graypix.mean()) if len(graypix) else 0.0
    base_std = float(graypix.std()) if len(graypix) else 0.0
    ringf=list((rm-nm).astype(float))+[float(rg.mean()-base_gray) if len(rg) else 0.,float(rg.std()-base_std) if len(rg) else 0.,float((grad[ring].mean() if len(rpix) else 0)-(gp.mean() if len(gp) else 0))]
    pad=6; pm=np.pad(m,pad); expected=float((ndimage.binary_dilation(pm,structure=ctx.disk5)&~pm).sum()); expected=max(expected,1.)
    ringf.append(min(max(len(rpix)/expected,0.),1.))
    out=np.asarray(shape+ch+hist.tolist()+ringf,dtype=np.float32)
    if out.shape!=(41,) or not np.isfinite(out).all(): raise ValueError("nonfinite/width")
    return out

def appearance_context_features(mask: np.ndarray, rgb: np.ndarray) -> np.ndarray:
    return PatchAppearanceContext(rgb).features(mask)
