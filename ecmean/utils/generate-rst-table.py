#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generate reStructuredText table from climatology/reference YAML files.
"""

__author__ = "GitHub Copilot, Feb 2026"

import argparse
import sys
from pathlib import Path

from ecmean.libs.files import load_yaml


def format_rst_table(data):
    """
    Create a reStructuredText list-table from YAML data.
    
    Parameters
    ----------
    data : dict
        Dictionary with variable names as keys and metadata as values
        
    Returns
    -------
    str
        Formatted RST list-table
    """
    # Check if any variable has CMIP6 model count
    has_cmip6_models = any(
        isinstance(info, dict) and info.get('cmip6', {}).get('nmodels')
        for info in data.values()
    )

    lines = []
    lines.append('.. list-table::')
    lines.append('   :header-rows: 1')

    # Adjust widths based on CMIP6 models column
    if has_cmip6_models:
        lines.append('   :widths: 30 20 12 12 10')
    else:
        lines.append('   :widths: 30 20 12 12')

    lines.append('')
    lines.append('   * - Long Name')
    lines.append('     - Dataset')
    lines.append('     - Mask')
    lines.append('     - Period')

    if has_cmip6_models:
        lines.append('     - # Models & Period')

    for _, info in data.items():
        if not isinstance(info, dict):
            continue

        longname = info.get('longname', '-')
        dataset = info.get('dataset', '-')
        version = info.get('version', '')
        mask = info.get('mask', 'global')

        # Merge version into dataset if available
        if version:
            dataset_full = f"{dataset} (v{version})"
        else:
            dataset_full = dataset

        # Format period
        year1 = info.get('year1', '')
        year2 = info.get('year2', '')
        if year1 and year2:
            period = f"{year1}-{year2}"
        else:
            period = '-'

        # Get CMIP6 model count if present
        nmodels = info.get('cmip6', {}).get('nmodels', '')
        cmip6_year1 = info.get('cmip6', {}).get('year1', '')
        cmip6_year2 = info.get('cmip6', {}).get('year2', '')

        lines.append(f'   * - {longname}')
        lines.append(f'     - {dataset_full}')
        lines.append(f'     - {mask}')
        lines.append(f'     - {period}')

        if has_cmip6_models:
            cmip6_period = f"{cmip6_year1}-{cmip6_year2}"
            lines.append(f'     - {nmodels} ({cmip6_period})')
    
    return '\n'.join(lines)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Generate RST table from climatology/reference YAML file.'
    )
    parser.add_argument(
        'yaml_file',
        type=str,
        help='Path to the climatology/reference YAML file'
    )
    parser.add_argument(
        '--output',
        type=str,
        help='Output file (default: print to terminal)'
    )
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()

    # Check if file exists
    yaml_path = Path(args.yaml_file)
    if not yaml_path.exists():
        print(f"Error: File '{args.yaml_file}' not found.", file=sys.stderr)
        sys.exit(1)

    # Load YAML data
    try:
        data = load_yaml(args.yaml_file)
    except Exception as e:
        print(f"Error reading YAML file: {e}", file=sys.stderr)
        sys.exit(1)

    if not data:
        print("Warning: No data found in YAML file.", file=sys.stderr)
        sys.exit(1)

    # Generate table
    table = format_rst_table(data)

    # Output
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(table)
            f.write('\n')
        print(f"Table written to {args.output}")
    else:
        print(table)


if __name__ == '__main__':
    main()
