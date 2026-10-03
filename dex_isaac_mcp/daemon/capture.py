"""Offscreen frames from a dedicated camera: works headless, no viewport needed.

A Replicator render product on our own camera prim, read back through the
"rgb" annotator. The daemon always renders (simd.py forces enable_cameras), so
this works with or without --gui, and the frame is independent of whatever
the GUI viewport is looking at.

Recording appends a frame every N physics steps to an in-memory list, each
tagged with the caption current at the time, and writes an animated GIF.

Import only after AppLauncher has started Kit.
"""

from __future__ import annotations

import base64
import io
import math
from pathlib import Path
from typing import Any

import numpy as np
import omni.kit.app
import omni.replicator.core as rep
import omni.usd
from pxr import Gf, Usd, UsdGeom

CAM_PATH = "/World/CaptureCam"
BACKDROP_PATH = "/World/Backdrop"


class Capturer:
    def __init__(self) -> None:
        self._rp = None
        self._annot = None
        self._size: tuple[int, int] | None = None
        self.frames: list[tuple[np.ndarray, str]] = []
        self.recording = False
        self.every = 2
        self.caption = ""
        self._count = 0

    # ---- camera --------------------------------------------------------

    def _camera(self) -> UsdGeom.Camera:
        stage = omni.usd.get_context().get_stage()
        cam = UsdGeom.Camera(stage.GetPrimAtPath(CAM_PATH))
        if not cam:
            cam = UsdGeom.Camera.Define(stage, CAM_PATH)
            cam.CreateFocalLengthAttr(35.0)
            cam.CreateHorizontalApertureAttr(20.955)
            cam.CreateClippingRangeAttr(Gf.Vec2f(0.005, 100.0))
        return cam

    def look_at(self, eye: list[float], target: list[float]) -> None:
        cam = self._camera()
        # USD cameras look down -Z with +Y up; build that frame with world Z up.
        view = Gf.Matrix4d().SetLookAt(Gf.Vec3d(*eye), Gf.Vec3d(*target), Gf.Vec3d(0, 0, 1))
        xf = UsdGeom.Xformable(cam.GetPrim())
        xf.ClearXformOpOrder()
        xf.AddTransformOp().Set(view.GetInverse())

    def fit(self, prim_path: str, direction: list[float], margin: float = 1.15,
            width: int = 640, height: int = 480, raise_frac: float = 0.0) -> dict[str, Any]:
        """Aim the camera along `direction` (target -> eye) so the prim's bounds fill the frame.

        raise_frac lifts the subject in the frame by that fraction of its radius,
        e.g. to keep it clear of a caption bar along the bottom.
        """
        stage = omni.usd.get_context().get_stage()
        cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(),
                                  [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
        box = cache.ComputeWorldBound(stage.GetPrimAtPath(prim_path)).ComputeAlignedRange()
        if box.IsEmpty():
            raise RuntimeError(f"{prim_path} has no renderable bounds")
        center = box.GetMidpoint()
        radius = box.GetSize().GetLength() / 2
        cam = self._camera()
        focal = cam.GetFocalLengthAttr().Get()
        h_ap = cam.GetHorizontalApertureAttr().Get()
        # The narrower field of view decides the distance at this aspect.
        fov_h = 2 * math.atan(h_ap / (2 * focal))
        fov_v = 2 * math.atan(h_ap * height / width / (2 * focal))
        dist = radius * margin / math.sin(min(fov_h, fov_v) / 2)
        d = Gf.Vec3d(*direction).GetNormalized()
        up = Gf.Vec3d(0, 0, 1) - d * d[2]
        up = up.GetNormalized() if up.GetLength() > 1e-6 else Gf.Vec3d(0, 1, 0)
        center = center - up * (radius * raise_frac)
        eye = center + d * dist
        target = [center[0], center[1], center[2]]
        self.look_at([eye[0], eye[1], eye[2]], target)
        return {"eye": [eye[0], eye[1], eye[2]], "target": target, "radius": radius}

    # ---- backdrop ------------------------------------------------------

    def backdrop(self, color: list[float] | None, center: list[float], direction: list[float],
                 size: float = 4.0) -> dict[str, Any]:
        """A matte panel behind the subject, facing the camera; color=None removes it."""
        stage = omni.usd.get_context().get_stage()
        if stage.GetPrimAtPath(BACKDROP_PATH):
            stage.RemovePrim(BACKDROP_PATH)
        if color is None:
            return {"backdrop": None}
        d = Gf.Vec3d(*direction).GetNormalized()
        # Far enough behind that it never intersects the subject.
        pos = Gf.Vec3d(*center) - d * (size / 2)
        plane = UsdGeom.Cube.Define(stage, BACKDROP_PATH)
        plane.CreateSizeAttr(1.0)
        plane.CreateDisplayColorAttr([Gf.Vec3f(*color)])
        xf = UsdGeom.Xformable(plane.GetPrim())
        rot = Gf.Rotation(Gf.Vec3d(0, 0, 1), d)  # panel normal toward the camera
        xf.AddTranslateOp().Set(pos)
        xf.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(rot.GetQuat()))
        xf.AddScaleOp().Set(Gf.Vec3d(size, size, 0.01))
        return {"backdrop": list(color)}

    # ---- frames --------------------------------------------------------

    def _ensure_product(self, width: int, height: int) -> None:
        if self._size == (width, height) and self._annot is not None:
            return
        if self._rp is not None:
            self._annot.detach()  # takes render-product PATHS, or nothing for all
            self._rp.destroy()
        self._camera()
        self._rp = rep.create.render_product(CAM_PATH, (width, height))
        self._annot = rep.AnnotatorRegistry.get_annotator("rgb")
        self._annot.attach([self._rp])
        self._size = (width, height)
        # A new render product takes a few frames before it holds an image.
        for _ in range(4):
            omni.kit.app.get_app().update()

    def grab(self, width: int = 640, height: int = 480) -> np.ndarray:
        self._ensure_product(width, height)
        for _ in range(60):
            data = self._annot.get_data()
            if data is not None and getattr(data, "size", 0):
                arr = np.asarray(data)[..., :3]
                if arr.any():
                    return arr.copy()
            omni.kit.app.get_app().update()
        raise RuntimeError("capture camera produced no image")

    def png_base64(self, width: int, height: int) -> str:
        from PIL import Image

        buf = io.BytesIO()
        Image.fromarray(self.grab(width, height)).save(buf, "PNG")
        return base64.b64encode(buf.getvalue()).decode()

    # ---- recording -------------------------------------------------------

    def start(self, every: int, width: int, height: int) -> None:
        self._ensure_product(width, height)
        self.frames = []
        self.every = max(1, int(every))
        self._count = 0
        self.recording = True

    def on_step(self) -> None:
        if not self.recording:
            return
        self._count += 1
        if self._count % self.every == 0:
            self.frames.append((self.grab(*self._size), self.caption))

    def stop(self, path: Path, fps: float, hold_last: float = 1.0) -> dict[str, Any]:
        from PIL import Image, ImageDraw, ImageFont

        self.recording = False
        if not self.frames:
            raise RuntimeError("no frames recorded; step the sim while recording")
        font = ImageFont.load_default(size=max(14, self._size[1] // 22))
        images = []
        for arr, caption in self.frames:
            img = Image.fromarray(arr)
            if caption:
                draw = ImageDraw.Draw(img, "RGBA")
                pad = img.height // 40
                tb = draw.textbbox((0, 0), caption, font=font)
                h = tb[3] - tb[1] + 2 * pad
                draw.rectangle([0, img.height - h - pad, img.width, img.height], fill=(0, 0, 0, 150))
                draw.text((2 * pad, img.height - h - pad + pad - tb[1]), caption,
                          font=font, fill=(255, 255, 255, 255))
            images.append(img)
        images = _stabilize(images)
        # One palette for the whole clip, no dithering: per-frame palettes and
        # dither both re-randomise pixels that did not move, which defeats the
        # GIF's frame-difference compression.
        palette = images[len(images) // 2].quantize(colors=128, dither=Image.Dither.NONE)
        images = [im.quantize(palette=palette, dither=Image.Dither.NONE) for im in images]
        frame_ms = int(1000 / fps)
        durations = [frame_ms] * len(images)
        durations[-1] += int(hold_last * 1000)
        path.parent.mkdir(parents=True, exist_ok=True)
        images[0].save(path, save_all=True, append_images=images[1:], duration=durations,
                       loop=0, optimize=True)
        n = len(images)
        self.frames = []
        return {"path": str(path), "frames": n, "bytes": path.stat().st_size,
                "seconds": round(sum(durations) / 1000, 2)}


def _stabilize(images: list, threshold: int = 6) -> list:
    """Hold pixels that changed by less than `threshold` at their previous value.

    The renderer's denoiser shimmers: between two frames of a motionless scene
    about 9% of background pixels change slightly, and a GIF re-encodes every
    one of them (a 9 s clip came to 15 MB). Real motion is far above the
    threshold, so it passes through untouched.
    """
    from PIL import Image

    out = [images[0]]
    prev = np.asarray(images[0]).astype(np.int16)
    for im in images[1:]:
        cur = np.asarray(im).astype(np.int16)
        still = (np.abs(cur - prev).max(axis=2) < threshold)[..., None]
        prev = np.where(still, prev, cur)
        out.append(Image.fromarray(prev.astype(np.uint8)))
    return out
