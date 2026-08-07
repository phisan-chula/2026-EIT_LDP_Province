# -*- coding: utf-8 -*-
"""
PROGRAM: QAQC_LDP_TestLine.py
Description: Quality Control module to validate custom LDP parameters vs Ground and UTM.
"""

import math
import warnings
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, LineString
import pyproj
import rasterio
import pygeodesy as pgd
from pathlib import Path


class LDPValidator:
    def __init__(self, gadm_path, dem_path, out_dir):
        """Initialize the validator with paths and geodesic models."""
        self.gadm_path = Path(gadm_path)
        self.dem_path = Path(dem_path)
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        
        self.geod = pyproj.Geod(ellps='WGS84')
        self.ellps = pgd.datums.Ellipsoids.WGS84
        self.geoid = self._init_geoid()
        
    def _init_geoid(self):
        """Initialize the TGM2017 Geoid model."""
        print("Loading TGM2017 Geoid model...")
        tgm_2017 = "/usr/share/GeographicLib/geoids/tgm2017-1.pgm"
        base_dir = Path(__file__).resolve().parent
        tgm_path = base_dir / "tgm2017-1.pgm"
        
        if Path(tgm_2017).is_file():
            return pgd.geoids.GeoidKarney(tgm_2017)
        return pgd.geoids.GeoidKarney(str(tgm_path))

    def _get_elevations(self, src, coords):
        """Extract orthometric heights from the DEM raster."""
        try:
            elevs = list(src.sample(coords))
            return float(elevs[0][0]), float(elevs[1][0])
        except Exception as e:
            raise ValueError(f"DEM sampling failed: {e}")

    def _calc_utm_parameters(self, lon1, lat1, lon2, lat2):
        """Calculate UTM grid distance and average Point Scale Factor (PSF)."""
        utm_crs_info = pyproj.database.query_utm_crs_info(
            datum_name="WGS 84", 
            area_of_interest=pyproj.aoi.AreaOfInterest(lon1, lat1, lon1, lat1)
        )[0]
        utm_proj = pyproj.CRS.from_epsg(utm_crs_info.code)
        transformer_utm = pyproj.Transformer.from_crs("EPSG:4326", utm_proj, always_xy=True)
        
        utm1_e, utm1_n = transformer_utm.transform(lon1, lat1)
        utm2_e, utm2_n = transformer_utm.transform(lon2, lat2)
        L3_UTM = math.hypot(utm2_e - utm1_e, utm2_n - utm1_n)
        
        utm_proj_obj = pyproj.Proj(utm_proj)
        psf1 = utm_proj_obj.get_factors(lon1, lat1).meridional_scale
        psf2 = utm_proj_obj.get_factors(lon2, lat2).meridional_scale
        psf_avg = (psf1 + psf2) / 2.0
        
        return L3_UTM, psf_avg

    def _calc_ldp_distance(self, hasc_safe, lon1, lat1, lon2, lat2):
        """Transform coordinates to LDP, calculate grid distance, and return PROJ string."""
        pj4_path = self.out_dir / hasc_safe / f"{hasc_safe}_LDP_CRS.PJ4"
        
        if not pj4_path.exists():
            print(f"  -> WARNING: {pj4_path} not found. L6 will be NaN.")
            return np.nan, "Not found"
            
        with open(pj4_path, 'r') as f:
            ldp_proj_str = f.read().strip()
            
        ldp_crs = pyproj.CRS(ldp_proj_str)
        transformer_ldp = pyproj.Transformer.from_crs("EPSG:4326", ldp_crs, always_xy=True)
        
        ldp1_e, ldp1_n = transformer_ldp.transform(lon1, lat1)
        ldp2_e, ldp2_n = transformer_ldp.transform(lon2, lat2)
        
        return math.hypot(ldp2_e - ldp1_e, ldp2_n - ldp1_n), ldp_proj_str

    def process_province(self, row, src):
        """Process a single province boundary to compute all test distances."""
        hasc = row.get('HASC_1')
        if not hasc or pd.isna(hasc): 
            return None
            
        hasc_safe = str(hasc).replace('.', '_').upper()
        name_1 = row.get('NAME_1', 'Unknown')
        
        # 1. P1: Centroid
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            p1_geom = row.geometry.centroid
        lon1, lat1 = p1_geom.x, p1_geom.y
        
        # 2. P2: 1000m North
        L1 = 1000.000 
        lon2, lat2, _ = self.geod.fwd(lon1, lat1, 0, L1)
        
        # 3. Heights and Geoid
        try:
            H1, H2 = self._get_elevations(src, [(lon1, lat1), (lon2, lat2)])
        except ValueError as e:
            print(f"  -> Skipping {hasc_safe}: {e}")
            return None
            
        H_avg = (H1 + H2) / 2.0
        N_avg = (self.geoid.height(lat1, lon1) + self.geoid.height(lat2, lon2)) / 2.0
        h_avg = H_avg + N_avg
        
        # 4. Height Scale Factor & Ground Distance
        avg_lat = (lat1 + lat2) / 2.0
        RG = self.ellps.rocGauss(avg_lat)
        HSF = RG / (RG + h_avg)
        L2 = L1 / HSF
        
        # 5. UTM Distance & Point Scale Factor
        L3, PSF_avg = self._calc_utm_parameters(lon1, lat1, lon2, lat2)
        
        # 6. Combined Scale Factor (CSF)
        L4 = L3 / PSF_avg
        CSF_avg = PSF_avg * HSF
        L5 = L3 / CSF_avg
        
        # 7. LDP Grid Distance and Definition
        L6, ldp_def = self._calc_ldp_distance(hasc_safe, lon1, lat1, lon2, lat2)
        
        return {
            'HASC_1': hasc,
            'NAME_1': name_1,
            'Province_Code': hasc_safe,
            'lat1': lat1, 'lon1': lon1,
            'lat2': lat2, 'lon2': lon2,
            'H_Orthometric': H_avg,
            'N_Undulation': N_avg,
            'h_Ellipsoidal': h_avg,
            'HSF': HSF,
            'PSF_utm': PSF_avg,
            'CSF_utm': CSF_avg,
            'L1_Ellps': L1,
            'L2_Grnd': L2,
            'L3_UTM': L3,
            'L4_UTM2Ellps': L4,
            'L5_UTM2Grnd': L5,
            'L6_LDP': L6,
            'LDP_Def': ldp_def,
            'diff_L1': L1 - L2,
            'diff_L2': 0.000,
            'diff_L3': L3 - L2,
            'diff_L4': L4 - L2,
            'diff_L5': L5 - L2,
            'diff_L6': L6 - L2,
            'geometry': LineString([Point(lon1, lat1), Point(lon2, lat2)])
        }

    def write_geopackages(self, df_results):
        """Export individual province Geopackages."""
        print(f"\nSaving generated test lines to Geopackages...")
        gdf_results = gpd.GeoDataFrame(df_results, geometry='geometry', crs="EPSG:4326")
        
        for hasc_safe, group in gdf_results.groupby('Province_Code'):
            prov_dir = self.out_dir / hasc_safe
            prov_dir.mkdir(parents=True, exist_ok=True)
            out_gpkg = prov_dir / f"{hasc_safe}_TestLine.gpkg"
            group.to_file(out_gpkg, driver='GPKG', layer='QAQC_Line')

    def write_markdown_report(self, df_results, df_line_descr):
        """Generate the final README.md summary report."""
        print("Generating Summary in OUTPUT_LDP/README.md ...")
        md_path = self.out_dir / 'README.md'
        
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write("# LDP Quality Control & Validation Summary\n\n")
            f.write("This report provides a comparative analysis of distances calculated across various map projections and surfaces. By establishing the **Ground Distance (L2)** as the true baseline (0.000m), we measure the linear distortion introduced by standard UTM grids against the custom Low Distortion Projections (LDP).\n\n")
            
            f.write("## --- Distance Definitions ---\n\n")
            f.write(df_line_descr.to_markdown(index=False) + "\n\n")
            
            for _, res in df_results.iterrows():
                p1_str = f"({res['lat1']:.9f}, {res['lon1']:.9f})"
                p2_str = f"({res['lat2']:.9f}, {res['lon2']:.9f})"
                msl_val = f"{res['H_Orthometric']:.3f}"
                hae_val = f"{res['h_Ellipsoidal']:.3f}"
                diff_l6_str = "NaN" if pd.isna(res['diff_L6']) else f"{res['diff_L6']:+.3f}"
                
                f.write("---\n\n") 
                f.write(f"### 📍 Province: {res['NAME_1']} ({res['HASC_1']})\n\n")
                
                f.write(f"| HASC_1 | {res['HASC_1']} | NameTH | {res['NAME_1']} |\n")
                f.write("|:---|:---|:---|:---|\n")
                f.write(f"| P1 | {p1_str} | P2 | {p2_str} |\n")
                f.write(f"| MSL | {msl_val} | HAE | {hae_val} |\n\n")
                
                f.write(f"> **LDP Definition:**\n> `{res['LDP_Def']}`\n\n")
                
                f.write("| ΔL1 | ΔL2 | ΔL3 | ΔL4 | ΔL5 | ΔL6 |\n")
                f.write("|:---:|:---:|:---:|:---:|:---:|:---:|\n")
                f.write(f"| {res['diff_L1']:+.3f} | *0.000 | {res['diff_L3']:+.3f} | {res['diff_L4']:+.3f} | {res['diff_L5']:+.3f} | {diff_l6_str} |\n\n")


def generate_line_descriptions(out_dir):
    """Generate and save the line descriptions lookup table."""
    line_descriptions = {
        'Line': ['L1', 'L2', 'L3', 'L4', 'L5', 'L6'],
        'LineDescr': [
            'on ellipsoid surface',
            'on ellipsoid surface , HSF applied (Ground)', 
            'on UTM grid',
            'on UTM grid , PSF applied',
            'on UTM grid , PSF&HSF (CSF) applied',
            'on LDP grid'
        ]
    }
    df = pd.DataFrame(line_descriptions)
    df.to_csv(Path(out_dir) / 'Line_Descriptions.csv', index=False)
    return df


def main():
    GADM_PATH = 'OUTPUT_PROV/gadm41_THA.gpkg'
    DEM_PATH = 'DATA/FABDEM_Thailand.vrt'
    OUT_DIR = 'OUTPUT_LDP'
    
    # 1. Initialization
    df_line_descr = generate_line_descriptions(OUT_DIR)
    validator = LDPValidator(GADM_PATH, DEM_PATH, OUT_DIR)
    
    # 2. Read Province Geometries
    if not Path(GADM_PATH).exists():
        raise FileNotFoundError(f"Cannot find province boundaries at {GADM_PATH}")
        
    print(f"Reading {GADM_PATH} (layer: ADM_ADM_1) ...")
    gdf_prov = gpd.read_file(GADM_PATH, layer='ADM_ADM_1')
    if gdf_prov.crs.to_epsg() != 4326:
        gdf_prov = gdf_prov.to_crs("EPSG:4326")

    # 3. Processing Loop
    results = []
    print(f"Processing {len(gdf_prov)} provinces ...")
    
    with rasterio.open(DEM_PATH) as src:
        for _, row in gdf_prov.iterrows():
            result = validator.process_province(row, src)
            if result:
                results.append(result)

    # 4. Export Data and Reports
    df_results = pd.DataFrame(results)
    if not df_results.empty:
        # Sort the DataFrame by HASC_1 ascendingly before writing outputs
        df_results = df_results.sort_values(by='HASC_1', ascending=True).reset_index(drop=True)
        
        validator.write_geopackages(df_results)
        validator.write_markdown_report(df_results, df_line_descr)
        
    print("QA/QC module execution complete.")


if __name__ == "__main__":
    main()
