import geopandas as gpd
import pandas as pd
import rasterio
from rasterstats import zonal_stats
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

class PopulatedAreaAnalyzer:
    """
    Modular geospatial analyzer that loads a single GeoPackage, extracts raster 
    population, filters noise, and computes centroid distances.
    """
    
    def __init__(self, gpkg_path: str, raster_path: str, layer_name: str = "ADM_ADM_1"):
        self.gpkg_path = gpkg_path
        self.raster_path = raster_path
        self.layer_name = layer_name
        
        # State variables
        self.gdf = None
        self.raster_crs = None
        self.metric_crs = None

    def load_data(self) -> None:
        """Loads the GPKG layer into memory once and reads the raster CRS."""
        # Load the specific layer from the GPKG
        self.gdf = gpd.read_file(self.gpkg_path, layer=self.layer_name)
        
        # Open raster to read metadata without loading the array into memory
        with rasterio.open(self.raster_path) as src:
            self.raster_crs = src.crs

        # Dynamically estimate the best UTM projection based on the bounds
        self.metric_crs = self.gdf.estimate_utm_crs()

    def generate_report(self, min_area_sqm: float = 5.0, noise_cutoff: float = 0.80) -> pd.DataFrame:
        """
        Executes the analysis pipeline and returns a Pandas DataFrame sorted by
        population, EW_km, NS_km, and centroid distance (descending).
        """
        if self.gdf is None:
            self.load_data()

        # 1. Reproject to metric CRS
        gdf_metric = self.gdf.to_crs(self.metric_crs)

        # 2. Filter target areas by physical size
        gdf_metric["area_sqm"] = gdf_metric.geometry.area
        filtered_gdf = gdf_metric[gdf_metric["area_sqm"] >= min_area_sqm].copy()
        
        if filtered_gdf.empty:
            print(f"Warning: No areas found >= {min_area_sqm} sqm.")
            return pd.DataFrame()

        # 3. Sample raster population for the remaining areas
        filtered_raster_crs = filtered_gdf.to_crs(self.raster_crs)
        stats = zonal_stats(
            filtered_raster_crs, 
            self.raster_path, 
            stats="sum", 
            all_touched=True,
            nodata=0
        )
        filtered_gdf["pop_est"] = [stat["sum"] if stat["sum"] else 0 for stat in stats]

        # 4. Cut off noise (drop the bottom X% based on population)
        if 0 < noise_cutoff < 1:
            threshold = filtered_gdf["pop_est"].quantile(noise_cutoff)
            denoised_gdf = filtered_gdf[filtered_gdf["pop_est"] >= threshold].copy()
        else:
            denoised_gdf = filtered_gdf.copy()

        if denoised_gdf.empty:
            print("Warning: No areas remaining after noise cutoff.")
            return pd.DataFrame()

        # 5. Calculate distance to the overall boundary centroid
        # We merge all features to find the global center of the loaded GPKG layer
        overall_polygon = gdf_metric.geometry.unary_union
        overall_centroid = overall_polygon.centroid
        
        denoised_gdf["centr_dist_km"] = denoised_gdf.geometry.centroid.distance(overall_centroid) / 1000.0

        # 6. Convert to standard Pandas DataFrame (drop the heavy geometry column)
        df_report = pd.DataFrame(denoised_gdf.drop(columns=["geometry"]))

        # 7. Apply descending sort requirements
        sort_columns = ["pop_est"]
        for col in ["EW_km", "NS_km", "centr_dist_km"]:
            if col in df_report.columns:
                sort_columns.append(col)
                
        # Sort all identified columns in descending order
        df_report = df_report.sort_values(
            by=sort_columns, 
            ascending=[False] * len(sort_columns)
        )

        return df_report

# ==========================================
# Execution Block
# ==========================================
if __name__ == "__main__":
    
    # Merged input file
    INPUT_GPKG = "OUTPUT_PROV/gadm41_THA.gpkg"
    RASTER_FILE = "DATA/tha_pop_2025_CN_100m_R2025A_v1.tif"

    print("[*] Initializing Analyzer...")
    analyzer = PopulatedAreaAnalyzer(
        gpkg_path=INPUT_GPKG,
        raster_path=RASTER_FILE,
        layer_name="ADM_ADM_1"
    )
    
    print("[*] Generating Report (Min Area: 5 sqm, Noise Cutoff: 80%)...")
    report_df = analyzer.generate_report(min_area_sqm=1.0, noise_cutoff=0.80)
    
    if not report_df.empty:
        print("\n" + "="*80)
        print("TOP POPULATED AREAS REPORT")
        print("="*80)
        
        # Display the specific columns of interest if they exist
        display_cols = ["NAME_1", "pop_est", "EW_km", "NS_km", "centr_dist_km"]
        existing_cols = [col for col in display_cols if col in report_df.columns]
        
        print(report_df[existing_cols].head(20).to_string(index=False))
        print("="*80)
