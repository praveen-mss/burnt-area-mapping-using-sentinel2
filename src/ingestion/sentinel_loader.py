# src/ingestion/sentinel_loader.py

import os
import numpy as np
import rasterio
import xml.etree.ElementTree as ET

from pathlib import Path
from rasterio.enums import Resampling


S2_SCALE_FACTOR = 10000.0


BAND_PATHS = {
    "B02": ("R10m", "B02"),
    "B03": ("R10m", "B03"),
    "B04": ("R10m", "B04"),
    "B08": ("R10m", "B08"),
    "B11": ("R20m", "B11"),
    "B12": ("R20m", "B12"),
}


# SCL classes to mask
SCL_MASK_VALUES = {3, 6, 8, 9, 10}


class Sentinel2L2ALoader:

    def __init__(self, safe_path):

        self.safe_path = Path(safe_path)
        self.granule = list((self.safe_path / "GRANULE").glob("*"))[0]
        self.img_data = self.granule / "IMG_DATA"

    # -----------------------------------------------------
    # Fast band path discovery
    # -----------------------------------------------------

    def _band_file(self, band):

        folder, code = BAND_PATHS[band]

        search_dir = self.img_data / folder

        files = list(search_dir.glob(f"*_{code}_*.jp2"))

        if len(files) == 0:
            raise RuntimeError(f"{band} not found")

        return files[0]

    # -----------------------------------------------------
    # Read CRS from XML
    # -----------------------------------------------------

    def _crs_from_xml(self):

        xml_file = self.granule / "MTD_TL.xml"

        tree = ET.parse(xml_file)
        root = tree.getroot()

        for elem in root.iter():

            if "HORIZONTAL_CS_CODE" in elem.tag:

                epsg = int(elem.text.split(":")[1])

                zone = epsg % 100

                north = epsg < 32700

                if north:
                    hemi = "N"
                else:
                    hemi = "S"

                # WKT CRS definition (no PROJ lookup required)
                wkt = f'''
    PROJCS["WGS 84 / UTM zone {zone}{hemi}",
    GEOGCS["WGS 84",
    DATUM["WGS_1984",
    SPHEROID["WGS 84",6378137,298.257223563]],
    PRIMEM["Greenwich",0],
    UNIT["degree",0.0174532925199433]],
    PROJECTION["Transverse_Mercator"],
    PARAMETER["latitude_of_origin",0],
    PARAMETER["central_meridian",{(zone-1)*6-180+3}],
    PARAMETER["scale_factor",0.9996],
    PARAMETER["false_easting",500000],
    PARAMETER["false_northing",0],
    UNIT["metre",1]]
    '''

                return wkt

        return None

    # -----------------------------------------------------
    # Reference grid (B11 20m)
    # -----------------------------------------------------

    def _reference(self):

        b11_path = self._band_file("B11")

        ref = rasterio.open(b11_path)

        return ref

    # -----------------------------------------------------
    # Load band and resample
    # -----------------------------------------------------

    def _load_band(self, band, ref):

        band_file = self._band_file(band)

        with rasterio.open(band_file) as src:

            if src.res[0] == 20:

                data = src.read(1).astype(np.float32)

            else:

                data = src.read(
                    1,
                    out_shape=(ref.height, ref.width),
                    resampling=Resampling.average
                ).astype(np.float32)

        data = data / S2_SCALE_FACTOR

        return data

    # -----------------------------------------------------
    # Scene classification layer
    # -----------------------------------------------------

    def load_scl(self):

        scl_dir = self.img_data / "R20m"

        scl_file = list(scl_dir.glob("*_SCL_*.jp2"))[0]

        with rasterio.open(scl_file) as src:

            scl = src.read(1)

        return scl

    # -----------------------------------------------------
    # Generate cloud / water mask
    # -----------------------------------------------------

    def _generate_scl_mask(self, scl):

        mask = np.isin(scl, list(SCL_MASK_VALUES))

        return mask

    # -----------------------------------------------------
    # Load requested bands and apply mask
    # -----------------------------------------------------

    def load_bands(self, bands):

        ref = self._reference()

        band_arrays = []

        for band in bands:

            band_arrays.append(
                self._load_band(band, ref)
            )

        stack = np.stack(band_arrays)

        # Load SCL and generate mask
        scl = self.load_scl()

        mask = self._generate_scl_mask(scl)

        # Apply mask to all bands
        stack[:, mask] = np.nan

        meta = {
            "driver": "GTiff",
            "dtype": "float32",
            "width": ref.width,
            "height": ref.height,
            "count": 1,
            "transform": ref.transform,
            "crs": self._crs_from_xml(),
        }

        return stack, meta, mask
