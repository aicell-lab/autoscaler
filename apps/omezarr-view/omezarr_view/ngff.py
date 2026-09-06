"""Synthesise NGFF 0.4 group metadata for a view over a non-Zarr source file.

tifffile's ``write_fsspec`` emits ``multiscales`` version 0.1 with only
``datasets[].path`` — no axes, no coordinateTransformations, no omero block —
which NGFF 0.4 consumers (Vizarr, ome-zarr-py) reject. This module builds the
0.4 metadata from whatever the source file actually declares, and records what
it could not fill in so the mapping stays honest.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple
from xml.etree import ElementTree

OME_NS = "{http://www.openmicroscopy.org/Schemas/OME/2016-06}"

# OME PhysicalSize*Unit values -> UDUNITS-2 names, which NGFF 0.4 requires.
_UNITS = {
    "m": "meter", "dm": "decimeter", "cm": "centimeter", "mm": "millimeter",
    "µm": "micrometer", "um": "micrometer", "micron": "micrometer",
    "nm": "nanometer", "pm": "picometer", "Å": "angstrom",
}

_AXIS_TYPE = {"t": "time", "c": "channel", "z": "space", "y": "space", "x": "space"}

# Fallback channel colours, in omero hex (no leading '#').
_DEFAULT_COLORS = ["00FF00", "FF0000", "0000FF", "FFFF00", "FF00FF", "00FFFF", "FFFFFF"]


@dataclass
class SourceMetadata:
    """What a reader could extract from the source file, before NGFF shaping.

    Fields left as None mean the source did not declare them; that distinction
    is what the mapping report is built from, so do not default them to
    plausible values here.
    """

    axes: str
    level_shapes: Sequence[Sequence[int]]
    dtype: str
    name: str
    # axis letter -> (size, unit string as the source spelled it)
    physical_sizes: Dict[str, Tuple[float, Optional[str]]] = field(default_factory=dict)
    channel_names: Optional[List[Optional[str]]] = None
    acquisition_date: Optional[str] = None


class MetadataMapping:
    """What was mapped into the view, and what the source did not provide.

    Carried alongside the attrs so the served view reports its own provenance,
    rather than the claim being made in prose somewhere else.
    """

    def __init__(self) -> None:
        self.mapped: Dict[str, Any] = {}
        self.unmapped: List[str] = []

    def add(self, field_name: str, value: Any) -> None:
        self.mapped[field_name] = value

    def miss(self, field_name: str, reason: str) -> None:
        self.unmapped.append(f"{field_name}: {reason}")

    def as_dict(self) -> Dict[str, Any]:
        return {"mapped": self.mapped, "not_mapped": self.unmapped}

    def as_sentences(self) -> Dict[str, str]:
        """Render the mapping as prose meant to be quoted verbatim.

        Figure captions quote this rather than paraphrasing it, so it has to
        read as English and it has to stay generated — a paraphrase drifts from
        what the code actually mapped, which is the whole failure this guards.
        """
        mapped_parts: List[str] = []
        dims = self.mapped.get("dimensions")
        if dims:
            inner = ", ".join(f"{a} {n}" for a, n in dims.items())
            mapped_parts.append(f"dimensions ({inner})")
        sizes = self.mapped.get("physical_pixel_sizes")
        if sizes:
            inner = ", ".join(
                f"{a} {v['size']:g} {'µm' if v['unit'] == 'micrometer' else (v['unit'] or 'units')}"
                for a, v in sizes.items())
            mapped_parts.append(f"physical pixel sizes ({inner})")
        names = [n for n in (self.mapped.get("channel_names") or []) if n]
        if names:
            mapped_parts.append(
                f"{len(names)} channel name{'s' if len(names) != 1 else ''} "
                f"({', '.join(names)})")
        dtype = self.mapped.get("dtype")
        if dtype:
            import numpy as _np
            try:
                dtype = _np.dtype(dtype).name
            except TypeError:
                pass
            mapped_parts.append(f"data type {dtype}")
        levels = self.mapped.get("pyramid_levels")
        if levels:
            mapped_parts.append(
                f"{levels} pyramid level{'s' if levels != 1 else ''} with their "
                "true scale factors" if levels > 1 else "a single resolution level")
        if self.mapped.get("acquisition_date"):
            mapped_parts.append(f"acquisition date {self.mapped['acquisition_date']}")

        # Drop the parenthetical reason; the caption states the fact, and the
        # reason lives in the dataset's own record.
        unmapped_parts = [m.split(":", 1)[0] for m in self.unmapped]

        mapped = ("Mapped from the source: " + "; ".join(mapped_parts) + "."
                  if mapped_parts else "Nothing could be mapped from the source.")
        not_mapped = ("Not mapped: " + "; ".join(unmapped_parts) + "."
                      if unmapped_parts else "Everything the source declares is mapped.")
        return {"mapped": mapped, "not_mapped": not_mapped}


def from_ome_xml(
    ome_xml: Optional[str],
    axes: str,
    level_shapes: Sequence[Sequence[int]],
    dtype: str,
    name: str,
) -> SourceMetadata:
    """Extract source metadata from a file's OME-XML."""
    md = SourceMetadata(axes=axes, level_shapes=level_shapes, dtype=dtype, name=name)
    if not ome_xml:
        return md
    image = ElementTree.fromstring(ome_xml).find(f"{OME_NS}Image")
    if image is None:
        return md
    acquired = image.find(f"{OME_NS}AcquisitionDate")
    if acquired is not None and acquired.text:
        md.acquisition_date = acquired.text.strip()
    pixels = image.find(f"{OME_NS}Pixels")
    if pixels is None:
        return md
    for axis in ("X", "Y", "Z"):
        raw = pixels.get(f"PhysicalSize{axis}")
        if raw is not None:
            md.physical_sizes[axis.lower()] = (
                float(raw), pixels.get(f"PhysicalSize{axis}Unit", "µm")
            )
    channels = pixels.findall(f"{OME_NS}Channel")
    if channels:
        md.channel_names = [c.get("Name") for c in channels]
    return md


def _dtype_window(dtype_str: str) -> Dict[str, float]:
    m = re.search(r"([iuf])(\d+)", dtype_str)
    if not m or m.group(1) == "f":
        return {"start": 0, "end": 1, "min": 0, "max": 1}
    bits = int(m.group(2)) * 8
    hi = (2 ** (bits - 1) - 1) if m.group(1) == "i" else (2**bits - 1)
    lo = -(2 ** (bits - 1)) if m.group(1) == "i" else 0
    return {"start": lo, "end": hi, "min": lo, "max": hi}


def build_ngff_attrs(md: SourceMetadata) -> Tuple[Dict[str, Any], MetadataMapping]:
    """Build NGFF 0.4 group ``.zattrs`` for a multiscale view.

    Returns ``(attrs, mapping)`` — the attrs dict, and the record of which
    source metadata made it in and which did not.
    """
    mapping = MetadataMapping()
    axes = md.axes.lower()
    base = list(md.level_shapes[0])

    sizes: Dict[str, Tuple[float, Optional[str]]] = {}
    for axis in ("x", "y", "z"):
        if axis not in axes:
            continue
        if axis not in md.physical_sizes:
            mapping.miss(f"physical pixel size {axis.upper()}",
                         "not declared in the source")
            continue
        size, unit_raw = md.physical_sizes[axis]
        unit = _UNITS.get(unit_raw) if unit_raw else None
        if unit_raw and unit is None:
            mapping.miss(f"physical pixel size {axis.upper()} unit",
                         f"source unit {unit_raw!r} has no UDUNITS-2 equivalent; "
                         "size mapped, unit dropped")
        sizes[axis] = (size, unit)
    if sizes:
        mapping.add("physical_pixel_sizes",
                    {k: {"size": v[0], "unit": v[1]} for k, v in sizes.items()})

    ngff_axes = []
    for a in axes:
        entry: Dict[str, Any] = {"name": a, "type": _AXIS_TYPE.get(a, "space")}
        if entry["type"] == "space" and a in sizes and sizes[a][1]:
            entry["unit"] = sizes[a][1]
        ngff_axes.append(entry)

    # Scale per level is the ACTUAL shape ratio, not an assumed factor of 2 —
    # the last levels of a pyramid are commonly rounded rather than exact.
    datasets = []
    for lvl, shape in enumerate(md.level_shapes):
        scale = [sizes.get(a, (1.0, None))[0] * (base[i] / shape[i])
                 for i, a in enumerate(axes)]
        datasets.append({
            "path": str(lvl),
            "coordinateTransformations": [{"type": "scale", "scale": scale}],
        })

    mapping.add("dimensions", {a: base[i] for i, a in enumerate(axes)})
    mapping.add("pyramid_levels", len(md.level_shapes))
    mapping.add("dtype", md.dtype)
    if len(md.level_shapes) == 1:
        mapping.miss("pyramid levels",
                     "source is single-scale; view is single-scale, not downsampled")

    attrs: Dict[str, Any] = {
        "multiscales": [{
            "version": "0.4",
            "name": md.name,
            "axes": ngff_axes,
            "datasets": datasets,
        }]
    }

    if "c" in axes:
        n_c = base[axes.index("c")]
        names = list(md.channel_names or [None] * n_c)[:n_c]
        names += [None] * (n_c - len(names))
        if any(names):
            mapping.add("channel_names", names)
        else:
            mapping.miss("channel names", "not declared in the source")
        # Colours and display windows are absent from most acquisition files;
        # a synthesised default is a rendering hint, not source metadata.
        mapping.miss("channel colours",
                     "not declared in the source; view assigns defaults")
        mapping.miss("display windows (contrast limits)",
                     "not declared in the source; view assigns full dtype range")
        window = _dtype_window(md.dtype)
        attrs["omero"] = {
            "version": "0.4",
            "name": md.name,
            "channels": [{
                "label": names[i] or f"Channel {i}",
                "color": _DEFAULT_COLORS[i % len(_DEFAULT_COLORS)],
                "window": dict(window),
                "active": i < 3,
            } for i in range(n_c)],
            "rdefs": {"model": "color"},
        }

    if md.acquisition_date:
        # Provenance only; NGFF 0.4 has nowhere standard to put it.
        mapping.add("acquisition_date", md.acquisition_date)
    else:
        mapping.miss("acquisition date", "not declared in the source")

    # Fields no reader in either recipe currently surfaces, stated once so the
    # view never implies a completeness it does not have.
    for absent in ("objective / instrument metadata", "stage position",
                   "plate / well context", "ROIs and annotations"):
        mapping.miss(absent, "not mapped by this recipe")

    return attrs, mapping
