# -*- coding: utf-8 -*-
import os
import argparse
import io
import json
import textwrap
import numpy as np
import rasterio
import matplotlib
import matplotlib.image as mpimg
import cairosvg
import concurrent.futures

# Force headless backend for thread-safe parallel rendering
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class ProvinceReport:
    """Generates a one-page A4 summary report for a given province (HASC_1)."""

    def __init__(self, hasc: str):
        self.hasc = hasc
        self.paths = self._generate_paths()
        self.fig = None
        self.axes = {}

    def _generate_paths(self) -> dict:
        """Constructs input/output file paths based on the HASC_1 code."""
        return {
            "fabdem": f"OUTPUT_FABDEM/{self.hasc}/{self.hasc}_fabdem.tif",
            "pop2025": f"OUTPUT_POP2025/{self.hasc}/{self.hasc}_population_2025_100m.tif",
            "svg_pp": f"OUTPUT_LDP/{self.hasc}/{self.hasc}_Plot_PP_Popu.svg",
            "svg_ldp": f"OUTPUT_LDP/{self.hasc}/{self.hasc}_LDP.svg",
            "log": f"OUTPUT_SAMPL/{self.hasc}/{self.hasc}_pipeline.log",
            "out_pdf": f"OUTPUT_LDP/{self.hasc}/{self.hasc}_OnePage.pdf"
        }

    def _setup_canvas(self):
        """Initializes the strictly A4 Matplotlib canvas and custom GridSpec layout."""
        self.fig = plt.figure(figsize=(8.27, 11.69))
        
        # Maximize space usage by explicitly setting margins and spacing
        gs = self.fig.add_gridspec(
            nrows=3, 
            ncols=2, 
            height_ratios=[1.8, 1.8, 1], 
            left=0.08,    
            right=0.95,   
            top=0.95,     
            bottom=0.05,  
            wspace=0.3,   
            hspace=0.2    
        )

        self.axes = {
            "dem": self.fig.add_subplot(gs[0, 0]),
            "pop": self.fig.add_subplot(gs[0, 1]),
            "svg_pp": self.fig.add_subplot(gs[1, 0]),
            "svg_ldp": self.fig.add_subplot(gs[1, 1]),
            "log_summary": self.fig.add_subplot(gs[2, 0]), # Bottom Left (TM + LDP Def)
            "log_json": self.fig.add_subplot(gs[2, 1])     # Bottom Right (Coverage Analysis)
        }

    def _render_dem(self):
        """Plots the FABDEM elevation raster."""
        with rasterio.open(self.paths["fabdem"]) as src:
            dem = src.read(1).astype(float)
            
            # Mask nodata so it prints as pure paper-white
            dem = np.where(dem == src.nodata, np.nan, dem)
            extent = [src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top]
            
        ax = self.axes["dem"]
        cmap = plt.cm.terrain.copy()
        cmap.set_bad(color='white')
        
        im = ax.imshow(dem, cmap=cmap, extent=extent)
        ax.set_title("FABDEM (MSL)", fontsize=10)
        self.fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Elevation (m)")

    def _render_population(self):
        """Plots the Population raster with log scaling, styled like the reference image."""
        with rasterio.open(self.paths["pop2025"]) as src:
            pop = src.read(1).astype(float)
            
            # Mask out nodata completely using NaN so it prints as pure white
            pop = np.where(pop == src.nodata, np.nan, pop)
            
            # Apply log1p (numpy automatically passes NaN through without error)
            pop_log = np.log1p(pop)
            extent = [src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top]

        ax = self.axes["pop"]
        
        # Switch to 'viridis' to match reference image, keeping nodata white
        cmap = plt.cm.viridis.copy()
        cmap.set_bad(color='white') 
        
        im = ax.imshow(pop_log, cmap=cmap, extent=extent)
        ax.set_title("Population 2025", fontsize=10)
        
        # Add gridlines to match reference style
        ax.grid(True, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
        
        # Update the colorbar label
        self.fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="log(1 + population)")

    def _render_svg(self, ax_key: str, path_key: str, title: str, zoom: float = 1.0):
        """Rasterizes and plots an SVG into the specified axis at 600 DPI, with optional zoom."""
        svg_path = self.paths[path_key]
        
        # Rasterize at 600 DPI to match output PDF quality
        png_data = cairosvg.svg2png(url=svg_path, dpi=600)
        img = mpimg.imread(io.BytesIO(png_data))
        
        ax = self.axes[ax_key]
        ax.imshow(img)
        
        # Apply zoom by shrinking the visible axis limits towards the center of the image
        if zoom != 1.0:
            h, w = img.shape[:2]
            center_x, center_y = w / 2, h / 2
            
            new_w = w / zoom
            new_h = h / zoom
            
            # Set new limits. Note: Y-axis is inverted in imshow (0 is at the top)
            ax.set_xlim(center_x - new_w / 2, center_x + new_w / 2)
            ax.set_ylim(center_y + new_h / 2, center_y - new_h / 2)
            
        ax.axis('off')
        ax.set_title(title, fontsize=10)

    def _render_log_text(self):
        """Scans pipeline logs, extracting the LDP TM block and safely wrapping JSONL outputs into subplots."""
        summary_extracted = []
        json_extracted = []
        
        try:
            with open(self.paths["log"], 'r', encoding='utf-8') as f:
                lines = f.readlines()
                
            i = 0
            while i < len(lines):
                raw_line = lines[i].rstrip('\n')
                
                # Scan for LDP TM or LDP LCC blocks (using space padding to catch exactly)
                if " LDP TM " in raw_line or " LDP LCC " in raw_line:
                    summary_extracted.append(raw_line)
                    lines_read = 0
                    
                    # Read ahead until we hit the row containing "| diff" or max 12 lines
                    while i + 1 < len(lines) and lines_read < 12:
                        i += 1
                        lines_read += 1
                        nxt_line = lines[i].rstrip('\n')
                        summary_extracted.append(nxt_line)
                        if "diff" in nxt_line:
                            break
                            
                    summary_extracted.append("") # Spacer
                    i += 1
                    continue
                            
                # Scan for JSONL containing LDP_Definition and put it in the LEFT column (summary_extracted)
                elif raw_line.startswith('{"meta": "LDP_Definition"'):
                    try:
                        json_data = json.loads(raw_line)
                        pretty_json = json.dumps(json_data, indent=2, ensure_ascii=False)
                        
                        summary_extracted.append("================ LDP Definition ================")
                        # Wrap long JSON lines (like PROJ Strings) at 65 characters so they don't bleed
                        for line in pretty_json.split('\n'):
                            summary_extracted.append(textwrap.fill(line, width=65, subsequent_indent='    '))
                        summary_extracted.append("")
                    except json.JSONDecodeError:
                        summary_extracted.append(textwrap.fill(raw_line, width=65))
                        
                # Scan for JSONL containing Coverage_Analysis and put it in the RIGHT column (json_extracted)
                elif raw_line.startswith('{"meta": "Coverage_Analysis"'):
                    try:
                        json_data = json.loads(raw_line)
                        pretty_json = json.dumps(json_data, indent=2, ensure_ascii=False)
                        
                        json_extracted.append("========= LDP Population Coverage Analysis =========")
                        for line in pretty_json.split('\n'):
                            json_extracted.append(textwrap.fill(line, width=65, subsequent_indent='    '))
                        json_extracted.append("")
                    except json.JSONDecodeError:
                        json_extracted.append(textwrap.fill(raw_line, width=65))
                        
                i += 1
                            
        except FileNotFoundError:
            summary_extracted.append(f"Log file not found: {self.paths['log']}")
            json_extracted.append(f"Log file not found: {self.paths['log']}")

        # Render Summary text block (Bottom Left)
        ax_sum = self.axes["log_summary"]
        ax_sum.axis('off')
        ax_sum.text(
            x=0.0, y=1.0, 
            s="\n".join(summary_extracted), 
            fontsize=7, 
            family='monospace', 
            verticalalignment='top', 
            horizontalalignment='left',
            transform=ax_sum.transAxes 
        )

        # Render JSON config text block (Bottom Right)
        ax_json = self.axes["log_json"]
        ax_json.axis('off')
        ax_json.text(
            x=0.0, y=1.0, 
            s="\n".join(json_extracted), 
            fontsize=7, 
            family='monospace', 
            verticalalignment='top', 
            horizontalalignment='left',
            transform=ax_json.transAxes 
        )

    def generate(self):
        """Executes the full rendering pipeline and saves the strictly A4 PDF."""
        os.makedirs(os.path.dirname(self.paths["out_pdf"]), exist_ok=True)
        self._setup_canvas()

        try:
            self._render_dem()
            self._render_population()
            
            # Apply requested zoom factors
            self._render_svg("svg_pp", "svg_pp", "Plot PP Popu", zoom=1.05)
            self._render_svg("svg_ldp", "svg_ldp", "LDP Map", zoom=1.1)
            
            self._render_log_text()
        except FileNotFoundError as e:
            print(f"[-] Error: Missing dependency for the province -> {e}")
            plt.close(self.fig)
            return
        
        # Save explicitly at 600 DPI, without bbox_inches='tight' to lock page size
        self.fig.savefig(self.paths["out_pdf"], dpi=600)
        plt.close(self.fig)
        print(f"[+] {self.hasc}: One-Page Report generated successfully.")


def worker_process(hasc: str):
    """Worker function for the ProcessPoolExecutor scanning log files if present."""
    log_path = f"OUTPUT_SAMPL/{hasc}/{hasc}_pipeline.log"
    if os.path.exists(log_path):
        print(f"Processing province: {hasc}")
        report = ProvinceReport(hasc)
        report.generate()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate OOP A4 One-Page Report (Ink Optimized, 600 DPI, Multiprocessing)")
    
    # Use --hasc argument that can accept a specific HASC code or 'ALL'
    parser.add_argument("--hasc", required=True, help="Specific HASC_1 code to process (e.g., TH_AC) or 'ALL' to process all provinces")
    
    args = parser.parse_args()
    
    base_dir = "OUTPUT_FABDEM"
    if not os.path.exists(base_dir):
        print(f"Error: Base directory '{base_dir}' not found.")
        exit(1)
        
    hasc_codes = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    
    if not hasc_codes:
        print(f"No provinces found in '{base_dir}'.")
        exit(0)
        
    if args.hasc.upper() == "ALL":
        print(f"Found {len(hasc_codes)} provinces. Starting parallel generation...")
        with concurrent.futures.ProcessPoolExecutor() as executor:
            executor.map(worker_process, hasc_codes)
        print("All reports generated successfully.")
    else:
        # Normalize dots to underscores (e.g., TH.BK -> TH_BK) to match folder structure
        target_hasc = args.hasc.replace('.', '_')
        
        if target_hasc not in hasc_codes:
            print(f"Warning: Province '{target_hasc}' (from '{args.hasc}') not found in '{base_dir}', attempting execution anyway...")
            
        worker_process(target_hasc)
