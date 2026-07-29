import argparse
import sys
import copy
import pandas as pd
import importlib
import matplotlib.pyplot as plt
import io
import concurrent.futures

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

# Dynamically import the script since it starts with a number
analyse_ldp = importlib.import_module("4_analyse_ldp")
LDP_Design = analyse_ldp.LDP_Design

def worker_process(offset, args_dict, province, prov_data):
    """
    Standalone worker function to be executed in a separate process.
    Handles the heavy lifting of LDP instantiation and PyProj transforms.
    """
    # Reconstruct the argparse Namespace from the dictionary
    args = argparse.Namespace(**args_dict)
    args.OFFSET_PP = offset
    
    current_prov_data = copy.deepcopy(prov_data)
    
    # Trap the stdout to suppress LDP_Design's internal print statements
    trap = io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = trap
    try:
        # Instantiate LDP_Design per iteration
        ldp = LDP_Design(args, province, current_prov_data)
    finally:
        # Restore standard output
        sys.stdout = old_stdout
        
    # Replicate Print_Coverage logic to extract DataFrame values
    if 'POP' in ldp.dfPP.columns:
        df_valid = ldp.dfPP[(ldp.dfPP.CSF_ppm >= -20) & (ldp.dfPP.CSF_ppm <= 20)]
        valid_points = len(df_valid)
        valid_pop = df_valid['POP'].sum()
        
        pt_pct = (valid_points / ldp.INITIAL_POINTS * 100) if ldp.INITIAL_POINTS > 0 else 0.0
        pop_pct = (valid_pop / ldp.INITIAL_POP * 100) if ldp.INITIAL_POP > 0 else 0.0
        
        if valid_points > 0:
            csf_min = df_valid['CSF_ppm'].min()
            csf_mean = df_valid['CSF_ppm'].mean()
            csf_max = df_valid['CSF_ppm'].max()
        else:
            csf_min = csf_mean = csf_max = 0.0
    else:
        valid_points = valid_pop = 0
        pt_pct = pop_pct = 0.0
        csf_min = csf_mean = csf_max = 0.0
        
    # Return both the row data for the DataFrame and metadata for the plot
    return {
        # DataFrame metrics
        'MSL_PP': ldp.MSL_PP,
        'HAE_PP': ldp.HAE_PP,
        'Province': ldp.PROV_CODE,
        'Total_Pts': ldp.INITIAL_POINTS,
        'Valid_Pts': valid_points,
        'Pt_Coverage(%)': pt_pct,
        'Total_POP': ldp.INITIAL_POP,
        'Valid_POP': valid_pop,
        'POP_Coverage(%)': pop_pct,
        'CSF_Min_ppm': csf_min,
        'CSF_Mean_ppm': csf_mean,
        'CSF_Max_ppm': csf_max,
        
        # Metadata required for plotting later
        '_OFFSET': offset,
        '_TRUE_MEAN_MSL': ldp.dfPP.MSL.mean(),
        '_RESULT_PATH': ldp.RESULT,
        '_FILE_CODE': ldp.FILE_CODE
    }

def process_single_province(province_code, args, config_data):
    """Processes pipeline analysis and plotting for a single province."""
    if province_code not in config_data:
        print(f"ERROR: Configuration for '{province_code}' not found in {args.toml}.")
        return
        
    prov_data = config_data[province_code]
    
    # Preliminary load to inspect dataset and retrieve the true mean MSL
    temp_args = copy.deepcopy(args)
    temp_args.OFFSET_PP = 0.0
    
    trap = io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = trap
    try:
        temp_ldp = LDP_Design(temp_args, province_code, prov_data)
    finally:
        sys.stdout = old_stdout
        
    true_mean_msl = temp_ldp.dfPP.MSL.mean()
    
    # Apply dynamic defaults if not provided via CLI
    upper_msl = args.upper_msl if args.upper_msl is not None else true_mean_msl + 100.0
    lower_msl = args.lower_msl if args.lower_msl is not None else true_mean_msl - 200.0
        
    # Error checking assertions for MSL boundaries
    if upper_msl <= true_mean_msl:
        print(f"ERROR for {province_code}: --upper_msl ({upper_msl}) must be greater than mean MSL ({true_mean_msl:.2f} m).")
        return
        
    if lower_msl >= true_mean_msl:
        print(f"ERROR for {province_code}: --lower_msl ({lower_msl}) must be less than mean MSL ({true_mean_msl:.2f} m).")
        return
        
    if lower_msl >= upper_msl:
        print(f"ERROR for {province_code}: --lower_msl ({lower_msl}) must be lower than --upper_msl ({upper_msl}).")
        return

    # List to store the row dictionaries for our final DataFrame
    records = []

    # Translate absolute MSL targets into relative offsets for the worker process
    upper_offset = upper_msl - true_mean_msl
    lower_offset = lower_msl - true_mean_msl

    offsets = []
    current_offset = lower_offset
    while current_offset <= upper_offset + (args.step * 0.01):  # 0.01 tolerance for floating point math
        offsets.append(current_offset)
        current_offset += args.step

    # Temporarily override args bounds for the dictionary packaging
    run_args = copy.deepcopy(args)
    run_args.upper_msl = upper_msl
    run_args.lower_msl = lower_msl
    args_dict = vars(run_args)
    
    # Process Pool for parallel execution of offsets
    print(f"[{province_code}] Starting ProcessPoolExecutor with {len(offsets)} offsets to compute (Mean MSL: {true_mean_msl:.2f} m)...")
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = executor.map(
            worker_process, 
            offsets, 
            [args_dict] * len(offsets), 
            [province_code] * len(offsets), 
            [prov_data] * len(offsets)
        )
        
        for res in results:
            print(f"[{province_code}] OFFSET_PP [{res['_OFFSET']}, defined by CLI args]")
            records.append(res)

    # Filter out the internal metadata keys ('_') before creating the DataFrame
    df_records = [{k: v for k, v in rec.items() if not k.startswith('_')} for rec in records]
    
    # Create the DataFrame
    df_report = pd.DataFrame(df_records)

    # Format setup mapping the Markdown columns strictly
    formats = [
        '+.2f', '+.2f', None, '.0f', '.0f', '.2f', ',.0f', ',.0f', '.2f', '+.1f', '+.1f', '+.1f'
    ]

    print(f"\n==================================== LDP Population Coverage Analysis: {province_code} ====================================")
    print(df_report.to_markdown(index=False, floatfmt=formats))
    print("==========================================================================================================================")

    # ================= PLOTTING SECTION ================= 
    if not df_report.empty:
        fig, ax1 = plt.subplots(figsize=(10, 6))

        # Main Plot: Population Coverage vs MSL
        ax1.plot(df_report['MSL_PP'], df_report['POP_Coverage(%)'], marker='o', color='b', linestyle='-', linewidth=2, label='Population Coverage')
        
        # Configure primary Y-axis (Population %)
        ax1.set_ylim(0, 100)
        ax1.set_ylabel('Population Coverage (%)', color='b', fontweight='bold')
        ax1.tick_params(axis='y', labelcolor='b')
        ax1.grid(True, linestyle='--', alpha=0.7)

        # Configure primary X-axis (MSL)
        ax1.set_xlabel('Project Plane MSL (m)', fontweight='bold')
        
        # Extract metadata from the first record for plotting attributes
        meta = records[0]
        
        # Extract the TRUE Mean MSL of the dataset directly from the returned metadata
        true_mean_msl_plot = meta['_TRUE_MEAN_MSL']
        ax1.axvline(x=true_mean_msl_plot, color='r', linestyle='--', 
                    linewidth=2, label=f'Topo MSL ({true_mean_msl_plot:.0f} m)')
        
        # Add Vertical Line based on user-defined MSL (if present) in Green
        if 'PP_MSL' in prov_data and str(prov_data['PP_MSL']).upper() != 'AUTO':
            user_msl = float(prov_data['PP_MSL'])
            ax1.axvline(x=user_msl, color='g', linestyle='--', 
                        linewidth=2, label=f'LDP PP_MSL ({user_msl:.0f} m)')
            
        ax1.legend(loc='lower right')

        # Setup Twin X-axis for HAE
        ax2 = ax1.twiny()
        
        # HAE = MSL + UNDUL. Calculate the constant undulation shift
        undul_shift = df_report['HAE_PP'].iloc[0] - df_report['MSL_PP'].iloc[0]
        
        # Sync ax2 limits with ax1 limits, shifted by undul_shift
        x1_limits = ax1.get_xlim()
        ax2.set_xlim(x1_limits[0] + undul_shift, x1_limits[1] + undul_shift)
        ax2.set_xlabel('Project Plane HAE (m)', fontweight='bold')

        plt.title(f"{meta['Province']}: Population Coverage vs Project Plane Offsets", pad=20, fontsize=14)
        plt.tight_layout()
        
        # Construct path and save: OUTPUT_LDP/TH_xx/TH_xx_Plot_PP_Popu.svg
        out_svg = meta['_RESULT_PATH'] / f"{meta['_FILE_CODE']}_Plot_PP_Popu.svg"
        plt.savefig(out_svg, format='svg')
        plt.close(fig)
        print(f"\n -> Plot saved successfully to: {out_svg}\n")

def main():
    parser = argparse.ArgumentParser(
        prog='41_plot_PP_PctPopu',
        description='Loop over PP MSL values to analyze population coverage and output as a dataframe and plot.'
    )
    # Positional Argument
    parser.add_argument('province', help="HASC_1 province code (e.g., TH.BR) or 'ALL'")
    
    # Existing Optional Arguments
    parser.add_argument('-t', '--toml', default='PROV_LDP.toml', help="TOML file containing province configuration data")
    parser.add_argument('-b', '--bypass', action='store_true', help="Bypass MSL outliers filtering")
    
    # New Optional Arguments for absolute MSL looping
    parser.add_argument('--upper_msl', type=float, default=None, help='Upper Project Plane MSL (default: mean MSL + 100 meter)')
    parser.add_argument('--lower_msl', type=float, default=None, help='Lower Project Plane MSL (default: mean MSL - 200 meter)')
    parser.add_argument('--step', type=float, default=20.0, help='PP step, default 20 meter')
    
    args = parser.parse_args()

    # Load configuration
    with open(args.toml, "rb") as f:
        config_data = tomllib.load(f)
        
    if args.province.upper() == 'ALL':
        provinces = list(config_data.keys())
        print(f"Found {len(provinces)} provinces in {args.toml}. Processing sequentially...")
        for prov in provinces:
            print(f"\n>>> Processing province: {prov}")
            process_single_province(prov, args, config_data)
        print("All provinces processed successfully.")
    else:
        process_single_province(args.province, args, config_data)

if __name__ == "__main__":
    main()
