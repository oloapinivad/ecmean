#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    Comparison script for CDO vs ESMF interpolation methods
    Tests both approaches on a range of CMIP6 models and produces a comparison report
    
    Usage:
        python cdo-esmf-comparison.py [--test-single] [--config CONFIG_FILE] [--debug]
        
    Options:
        --test-single       Test with single model only (EC-Earth3)
        --config FILE       Use specific config file (default: config_create_clim.yml)
        --debug             Enable debug mode with verbose output
        --test-setup-only   Only test setup, don't run comparison
        
    Examples:
        # Quick test with one model
        python cdo-esmf-comparison.py --test-single
        
        # Test setup only (useful for troubleshooting)
        python cdo-esmf-comparison.py --test-setup-only
        
        # Full comparison with debug output
        python cdo-esmf-comparison.py --debug
        
        # Use custom config
        python cdo-esmf-comparison.py --config my_config.yml
"""

__author__ = "Paolo Davini (p.davini@isac.cnr.it), Aug 2025."

import os
import warnings
import glob
import copy
import yaml
import argparse
import numpy as np
import time
import gc
from datetime import datetime
from ecmean.performance_indices import performance_indices
from ecmean.libs.files import load_yaml

warnings.simplefilter("ignore")

def sanitize_for_yaml(obj):
    """
    Recursively sanitize data structures to make them YAML-serializable.
    Converts numpy types to native Python types.
    """
    if isinstance(obj, dict):
        return {key: sanitize_for_yaml(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [sanitize_for_yaml(item) for item in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif hasattr(obj, 'item'):  # numpy scalar
        return obj.item()
    else:
        return obj

def test_performance_indices_setup(config_dict, model='EC-Earth3', ensemble='r1i1p1f1'):
    """Test if performance_indices can be imported and basic setup works"""
    
    print("Testing performance indices setup...")
    
    try:
        # Test import
        from ecmean.performance_indices import PerformanceIndices
        print("✓ PerformanceIndices imported successfully")
        
        # Test basic initialization
        pi = PerformanceIndices(
            expname, year1, year1,  # Just one year for quick test
            config=config_dict,
            model=model,
            ensemble=ensemble,
            climatology=refclim,
            loglevel='debug',
            tool='ESMF'  # Start with ESMF as it's more established
        )
        print("✓ PerformanceIndices initialized successfully")
        
        # Test preparation step
        try:
            pi.prepare()
            print("✓ PerformanceIndices preparation successful")
            return True
        except Exception as e:
            print(f"✗ PerformanceIndices preparation failed: {e}")
            import traceback
            traceback.print_exc()
            return False
            
    except Exception as e:
        print(f"✗ PerformanceIndices setup failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def check_configuration(config_dict):
    """Check if configuration has required fields"""
    required_fields = ['dirs']
    missing = []
    
    for field in required_fields:
        if field not in config_dict:
            missing.append(field)
    
    if 'dirs' in config_dict:
        required_dirs = ['exp', 'tab', 'fig']
        for dir_field in required_dirs:
            if dir_field not in config_dict['dirs']:
                missing.append(f"dirs.{dir_field}")
    
    if missing:
        print(f"✗ Missing required configuration fields: {missing}")
        return False
    
    # Check if directories exist
    base_dir = config_dict['dirs']['exp']
    if not os.path.exists(base_dir):
        print(f"✗ Experiment directory does not exist: {base_dir}")
        return False
    
    print("✓ Configuration appears valid")
    return True

def check_dependencies():
    """Check if required packages are available"""
    missing = []
    
    try:
        import ecmean
        print("✓ ecmean available")
    except ImportError:
        missing.append("ecmean")
    
    try:
        import xesmf
        print("✓ xesmf available")
    except ImportError:
        missing.append("xesmf")
    
    try:
        import smmregrid
        print("✓ smmregrid available")
    except ImportError:
        missing.append("smmregrid")
    
    # Check CDO
    import shutil
    if shutil.which('cdo'):
        print("✓ cdo available")
    else:
        print("⚠ cdo not found in PATH (CDO method may fail)")
    
    if missing:
        print("✗ Missing required packages:")
        for pkg in missing:
            print(f"  - {pkg}")
        print("\nPlease install missing packages before running the comparison.")
        return False
    
    return True

# Configuration parameters
year1 = 1990
year2 = 1990
expname = 'historical'
refclim = 'EC24'
nprocs = 1
config_file = 'config_create_clim.yml'

# Tolerance for significant differences (as fraction, not percentage)
TOLERANCE = 0.1  # 10%

# Output directory for comparison results
comparison_dir = './comparison_results'
os.makedirs(comparison_dir, exist_ok=True)

# Models to test (subset for faster testing, can be expanded)
test_models = ['EC-Earth3', 'IPSL-CM6A-LR', 'FGOALS-g3', 'CanESM5', 'CESM2']
# Full model list (uncomment for complete test)
#test_models = ['EC-Earth3', 'IPSL-CM6A-LR', 'FGOALS-g3', 'CanESM5', 'CESM2', 'CNRM-CM6-1',
#               'GISS-E2-1-G', 'ACCESS-CM2', 'SAM0-UNICON', 'UKESM1-0-LL',
#               'MIROC6', 'MPI-ESM1-2-HR', 'AWI-CM-1-1-MR', 'NorESM2-MM', 'GFDL-CM4']

def modify_config_for_tool(config_dict, tool):
    """Modify configuration to use specified interpolation tool"""
    modified_config = copy.deepcopy(config_dict)
    
    # Adjust output directory to avoid conflicts
    if 'dirs' in modified_config:
        original_tab = modified_config['dirs']['tab']
        modified_config['dirs']['tab'] = os.path.join(original_tab, tool.lower())
        os.makedirs(modified_config['dirs']['tab'], exist_ok=True)
    
    return modified_config

def run_performance_indices_with_tool(model, ensemble, config_dict, tool, log_level='info'):
    """Run performance indices with specified tool and handle errors"""
    
    print(f"  → Running {tool} interpolation...")
    start_time = time.time()
    
    try:
        # Modify config for this tool
        tool_config = modify_config_for_tool(config_dict, tool)
        
        print(f"    Output directory: {tool_config['dirs']['tab']}")
    
        
        # Add small delay between attempts to reduce resource conflicts
        if tool == 'CDO':
            print(f"    Adding delay to reduce resource conflicts...")
            time.sleep(2.0)
        
        # Run performance indices with timing
        execution_start = time.time()
        performance_indices(
            expname, year1, year2, 
            config=tool_config, 
            model=model,
            ensemble=ensemble, 
            numproc=nprocs,  # Use adjusted process count
            climatology=refclim,
            loglevel=log_level,  # Use configurable log level
            tool=tool  # Pass the interpolation tool
        )
        execution_time = time.time() - execution_start
        
        # Look for the results file
        result_pattern = os.path.join(
            tool_config['dirs']['tab'], 
            f'PI4_{refclim}_{expname}_{model}_{ensemble}_{year1}_{year2}.yml'
        )
        result_files = glob.glob(result_pattern)
        
        print(f"    Looking for result file: {result_pattern}")
        print(f"    Found {len(result_files)} matching files")
        
        if result_files:
            print(f"    Loading results from: {result_files[0]}")
            results = load_yaml(result_files[0])
            total_time = time.time() - start_time
            
            print(f"    ✓ Results loaded successfully ({len(results)} variables)")
            print(f"    ⏱️  Execution time: {execution_time:.2f}s, Total time: {total_time:.2f}s")
            
            return {
                'status': 'success', 
                'results': results, 
                'error': None,
                'performance': {
                    'execution_time': round(execution_time, 2),
                    'total_time': round(total_time, 2),
                    'variables_processed': len(results),
                    'time_per_variable': round(execution_time / len(results), 2) if results else 0,
                    'memory_efficient': execution_time < 300,  # Flag for fast execution
                }
            }
        else:
            # List what files are actually in the directory
            dir_files = glob.glob(os.path.join(tool_config['dirs']['tab'], "*.yml"))
            error_msg = f"Result file not found. Expected: {os.path.basename(result_pattern)}"
            if dir_files:
                error_msg += f"\nAvailable files: {[os.path.basename(f) for f in dir_files]}"
            else:
                error_msg += "\nNo .yml files found in output directory"
            
            total_time = time.time() - start_time
            print(f"    ✗ {error_msg}")
            print(f"    ⏱️  Failed after: {total_time:.2f}s")
            
            return {
                'status': 'error', 
                'results': None, 
                'error': error_msg,
                'performance': {
                    'execution_time': round(total_time, 2),
                    'total_time': round(total_time, 2),
                    'variables_processed': 0,
                    'time_per_variable': 0,
                    'memory_efficient': False,
                }
            }
            
    except Exception as e:
        import traceback
        total_time = time.time() - start_time
        error_msg = f"Exception during {tool} execution: {str(e)}"
        traceback_str = traceback.format_exc()
        print(f"    ✗ {error_msg}")
        print(f"    ⏱️  Failed after: {total_time:.2f}s")
        print(f"    Full traceback:\n{traceback_str}")
        
        return {
            'status': 'error', 
            'results': None, 
            'error': f"{error_msg}\n{traceback_str}",
            'performance': {
                'execution_time': round(total_time, 2),
                'total_time': round(total_time, 2),
                'variables_processed': 0,
                'time_per_variable': 0,
                'memory_efficient': False,
            }
        }

def compare_results(esmf_results, cdo_results, model, esmf_perf=None, cdo_perf=None, tolerance=TOLERANCE):
    """Compare results between ESMF and CDO approaches, including performance metrics"""
    
    comparison = {
        'model': model,
        'variables': {},
        'summary': {
            'total_comparisons': 0,
            'significant_differences': 0,
            'max_relative_difference': 0.0,
            'mean_relative_difference': 0.0,
            'variables_compared': 0,
            'missing_variables': []
        },
        'performance': {}
    }
    
    # Add performance comparison
    if esmf_perf and cdo_perf:
        comparison['performance'] = {
            'esmf': esmf_perf,
            'cdo': cdo_perf,
            'speed_comparison': {
                'cdo_faster': bool(cdo_perf['execution_time'] < esmf_perf['execution_time']),
                'speedup_factor': round(float(esmf_perf['execution_time']) / float(cdo_perf['execution_time']), 2) 
                                if cdo_perf['execution_time'] > 0 else 0.0,
                'time_difference': round(float(abs(esmf_perf['execution_time'] - cdo_perf['execution_time'])), 2),
                'relative_difference': round(float(abs(esmf_perf['execution_time'] - cdo_perf['execution_time'])) 
                                           / float(max(esmf_perf['execution_time'], cdo_perf['execution_time'])), 3)
                                     if max(esmf_perf['execution_time'], cdo_perf['execution_time']) > 0 else 0.0
            }
        }
    
    if not esmf_results or not cdo_results:
        comparison['error'] = 'Missing results from one or both methods'
        return comparison
    
    all_diffs = []
    
    # Compare variables that exist in both results
    common_vars = set(esmf_results.keys()) & set(cdo_results.keys())
    missing_vars = (set(esmf_results.keys()) | set(cdo_results.keys())) - common_vars
    
    comparison['summary']['variables_compared'] = len(common_vars)
    comparison['summary']['missing_variables'] = list(missing_vars)
    
    for var in common_vars:
        comparison['variables'][var] = {}
        
        for season in esmf_results[var]:
            if season in cdo_results[var]:
                comparison['variables'][var][season] = {}
                
                for region in esmf_results[var][season]:
                    if region in cdo_results[var][season]:
                        esmf_val = esmf_results[var][season][region]
                        cdo_val = cdo_results[var][season][region]
                        
                        if isinstance(esmf_val, (int, float)) and isinstance(cdo_val, (int, float)):
                            # Skip NaN values
                            if np.isnan(esmf_val) or np.isnan(cdo_val):
                                continue
                                
                            # Calculate relative difference
                            if abs(esmf_val) > 1e-10:  # Avoid division by very small numbers
                                rel_diff = abs(cdo_val - esmf_val) / abs(esmf_val)
                            else:
                                rel_diff = abs(cdo_val - esmf_val)
                            
                            all_diffs.append(rel_diff)
                            comparison['summary']['total_comparisons'] += 1
                            
                            comparison['variables'][var][season][region] = {
                                'esmf_value': float(esmf_val),
                                'cdo_value': float(cdo_val),
                                'absolute_difference': float(cdo_val - esmf_val),
                                'relative_difference': float(rel_diff),
                                'significant': bool(rel_diff > tolerance)
                            }
                            
                            if rel_diff > tolerance:
                                comparison['summary']['significant_differences'] += 1
    
    if all_diffs:
        comparison['summary']['max_relative_difference'] = float(max(all_diffs))
        comparison['summary']['mean_relative_difference'] = float(np.mean(all_diffs))
    
    return comparison

def main():
    """Main comparison function"""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Compare CDO vs ESMF interpolation methods')
    parser.add_argument('--test-single', action='store_true',
                       help='Test with single model only (EC-Earth3)')
    parser.add_argument('--config', default='../config_CMIP6_esgpull.yml',
                       help='Configuration file to use')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug mode with more verbose output')
    parser.add_argument('--test-setup-only', action='store_true',
                       help='Only test the setup, do not run full comparison')
    args = parser.parse_args()
    
    # Set log level based on debug flag
    log_level = 'debug' if args.debug else 'info'
    
    # Select models based on argument
    models_to_test = ['EC-Earth3'] if args.test_single else test_models
    
    print("=== CDO vs ESMF Interpolation Comparison ===")
    
    # Check dependencies first
    if not check_dependencies():
        return
        
    print(f"Testing {len(models_to_test)} models from {year1} to {year2}")
    print(f"Using climatology: {refclim}")
    print(f"Config file: {args.config}")
    if args.test_single:
        print("*** SINGLE MODEL TEST MODE ***")
    if args.debug:
        print("*** DEBUG MODE ENABLED ***")
    if args.test_setup_only:
        print("*** SETUP TEST ONLY ***")
    print(f"Results will be saved to: {comparison_dir}")
    print()
    
    # Load base configuration
    try:
        base_config = load_yaml(args.config)
        print("✓ Configuration file loaded successfully")
    except FileNotFoundError:
        print(f"✗ Error: Configuration file {args.config} not found!")
        print("Make sure the config file exists in the current directory")
        return
    except Exception as e:
        print(f"✗ Error loading configuration: {e}")
        return
    
    # Validate configuration
    if not check_configuration(base_config):
        return
    
    print(f"Configuration details:")
    print(f"  Experiment dir: {base_config['dirs']['exp']}")
    print(f"  Output dir: {base_config['dirs']['tab']}")
    print()
    
    # Test performance indices setup
    if not test_performance_indices_setup(base_config):
        print("✗ Performance indices setup test failed. Please check your configuration and data availability.")
        return
    
    print("All checks passed. Starting comparison...")
    print()
    
    # If only testing setup, stop here
    if args.test_setup_only:
        print("Setup test completed successfully!")
        return
    
    # Initialize comparison report
    comparison_report = {
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'experiment': expname,
            'years': f'{year1}-{year2}',
            'climatology': refclim,
            'models_tested': models_to_test,
            'config_file': args.config,
            'test_mode': 'single' if args.test_single else 'multiple'
        },
        'model_comparisons': {},
        'overall_summary': {
            'models_successful': 0,
            'models_failed': 0,
            'esmf_only_success': 0,
            'cdo_only_success': 0,
            'both_methods_success': 0
        },
        'benchmark_summary': {
            'total_models_benchmarked': 0,
            'cdo_faster_count': 0,
            'esmf_faster_count': 0,
            'avg_cdo_time': 0,
            'avg_esmf_time': 0,
            'max_speedup_cdo': 0,
            'max_speedup_esmf': 0
        }
    }
    
    # Test each model
    for i, model in enumerate(models_to_test, 1):
        print(f"[{i}/{len(models_to_test)}] Testing model: {model}")
        
        # Determine ensemble
        if model in ['CNRM-CM6-1', 'UKESM1-0-LL']:
            ensemble = "r1i1p1f2"
        else:
            ensemble = "r1i1p1f1"
        
        print(f"  Using ensemble: {ensemble}")
        
        # Run with ESMF first (more stable)
        print(f"  → Running ESMF interpolation...")
        esmf_result = run_performance_indices_with_tool(model, ensemble, base_config, 'ESMF', log_level)
        
        # Add delay between methods to avoid resource conflicts
        print(f"  → Waiting 5 seconds before CDO test to avoid resource conflicts...")
        time.sleep(5)
        
        # Run with CDO (more prone to resource issues)
        print(f"  → Running CDO interpolation...")
        cdo_result = run_performance_indices_with_tool(model, ensemble, base_config, 'CDO', log_level)
        
        # Compare results
        if esmf_result['status'] == 'success' and cdo_result['status'] == 'success':
            comparison = compare_results(
                esmf_result['results'], 
                cdo_result['results'], 
                model,
                esmf_result.get('performance'),
                cdo_result.get('performance')
            )
            comparison_report['overall_summary']['both_methods_success'] += 1
            comparison_report['overall_summary']['models_successful'] += 1
            
            # Update benchmark summary
            if 'performance' in comparison and 'speed_comparison' in comparison['performance']:
                comparison_report['benchmark_summary']['total_models_benchmarked'] += 1
                speed_comp = comparison['performance']['speed_comparison']
                
                if speed_comp['cdo_faster']:
                    comparison_report['benchmark_summary']['cdo_faster_count'] += 1
                    if speed_comp['speedup_factor'] > comparison_report['benchmark_summary']['max_speedup_cdo']:
                        comparison_report['benchmark_summary']['max_speedup_cdo'] = speed_comp['speedup_factor']
                else:
                    comparison_report['benchmark_summary']['esmf_faster_count'] += 1
                    if speed_comp['speedup_factor'] > 0:
                        inverse_speedup = 1.0 / speed_comp['speedup_factor']
                        if inverse_speedup > comparison_report['benchmark_summary']['max_speedup_esmf']:
                            comparison_report['benchmark_summary']['max_speedup_esmf'] = inverse_speedup
            
        elif esmf_result['status'] == 'success' and cdo_result['status'] == 'error':
            comparison = {
                'model': model,
                'status': 'esmf_only',
                'esmf_error': None,
                'cdo_error': cdo_result['error'],
                'performance': {
                    'esmf': esmf_result.get('performance'),
                    'cdo': cdo_result.get('performance')
                }
            }
            comparison_report['overall_summary']['esmf_only_success'] += 1
            
        elif esmf_result['status'] == 'error' and cdo_result['status'] == 'success':
            comparison = {
                'model': model,
                'status': 'cdo_only',
                'esmf_error': esmf_result['error'],
                'cdo_error': None,
                'performance': {
                    'esmf': esmf_result.get('performance'),
                    'cdo': cdo_result.get('performance')
                }
            }
            comparison_report['overall_summary']['cdo_only_success'] += 1
            
        else:
            comparison = {
                'model': model,
                'status': 'both_failed',
                'esmf_error': esmf_result['error'],
                'cdo_error': cdo_result['error'],
                'performance': {
                    'esmf': esmf_result.get('performance'),
                    'cdo': cdo_result.get('performance')
                }
            }
            comparison_report['overall_summary']['models_failed'] += 1
        
        comparison_report['model_comparisons'][model] = comparison
        
        # Clean up memory after each model test
        gc.collect()
        print(f"  → Memory cleaned up after {model}")
        
        # Show status
        if 'status' in comparison:
            print(f"  → Status: {comparison['status']}")
            # Show performance info if available
            if 'performance' in comparison:
                perf = comparison['performance']
                if 'esmf' in perf and 'cdo' in perf:
                    if perf['esmf'] and perf['cdo']:
                        esmf_time = perf['esmf']['execution_time']
                        cdo_time = perf['cdo']['execution_time']
                        if 'speed_comparison' in perf:
                            faster_method = 'CDO' if perf['speed_comparison']['cdo_faster'] else 'ESMF'
                            speedup = perf['speed_comparison']['speedup_factor']
                            print(f"  ⏱️  Performance: {faster_method} faster by {speedup}x "
                                  f"(ESMF: {esmf_time}s, CDO: {cdo_time}s)")
        elif 'summary' in comparison:
            total = comparison['summary']['total_comparisons']
            significant = comparison['summary']['significant_differences']
            print(f"  → Completed: {total} comparisons, {significant} significant differences")
            # Show performance info
            if 'performance' in comparison and 'speed_comparison' in comparison['performance']:
                perf = comparison['performance']['speed_comparison']
                faster_method = 'CDO' if perf['cdo_faster'] else 'ESMF'
                speedup = perf['speedup_factor']
                print(f"  ⏱️  Performance: {faster_method} faster by {speedup}x")
        print()
    
    # Save comparison report
    suffix = '_single' if args.test_single else ''
    report_file = os.path.join(comparison_dir, f'cdo_vs_esmf_comparison_{expname}_{year1}_{year2}{suffix}.yml')
    
    # Sanitize the report before saving
    sanitized_report = sanitize_for_yaml(comparison_report)
    
    with open(report_file, 'w', encoding='utf8') as file:
        yaml.safe_dump(sanitized_report, file, sort_keys=False, default_flow_style=False)
    
    # Calculate benchmark averages
    successful_comparisons = []
    esmf_times = []
    cdo_times = []
    
    for model_data in comparison_report['model_comparisons'].values():
        if 'performance' in model_data:
            perf = model_data['performance']
            if 'esmf' in perf and 'cdo' in perf and perf['esmf'] and perf['cdo']:
                esmf_times.append(perf['esmf']['execution_time'])
                cdo_times.append(perf['cdo']['execution_time'])
    
    if esmf_times and cdo_times:
        comparison_report['benchmark_summary']['avg_esmf_time'] = round(float(np.mean(esmf_times)), 2)
        comparison_report['benchmark_summary']['avg_cdo_time'] = round(float(np.mean(cdo_times)), 2)
        
        # Sanitize the entire report before saving
        sanitized_report = sanitize_for_yaml(comparison_report)
        
        # Re-save with updated averages
        with open(report_file, 'w', encoding='utf8') as file:
            yaml.safe_dump(sanitized_report, file, sort_keys=False, default_flow_style=False)
    
    # Print summary
    print("=" * 50)
    print("COMPARISON COMPLETE")
    print("=" * 50)
    print(f"Report saved to: {report_file}")
    print()
    print("SUMMARY:")
    print(f"  Models tested: {len(models_to_test)}")
    print(f"  Both methods successful: {comparison_report['overall_summary']['both_methods_success']}")
    print(f"  ESMF only successful: {comparison_report['overall_summary']['esmf_only_success']}")
    print(f"  CDO only successful: {comparison_report['overall_summary']['cdo_only_success']}")
    print(f"  Both methods failed: {comparison_report['overall_summary']['models_failed']}")
    
    # Print significant differences summary
    significant_models = []
    for model_data in comparison_report['model_comparisons'].values():
        if 'summary' in model_data and model_data['summary']['significant_differences'] > 0:
            significant_models.append({
                'model': model_data['model'],
                'significant_count': model_data['summary']['significant_differences'],
                'total_count': model_data['summary']['total_comparisons'],
                'max_diff': model_data['summary']['max_relative_difference']
            })
    
    if significant_models:
        print(f"\nMODELS WITH SIGNIFICANT DIFFERENCES (>10%):")
        for model in significant_models:
            print(f"  {model['model']}: {model['significant_count']}/{model['total_count']} "
                  f"comparisons, max diff: {model['max_diff']:.2%}")
    else:
        print(f"\n✓ No significant differences found between methods")
    
    # Performance summary
    benchmark = comparison_report['benchmark_summary']
    if benchmark['total_models_benchmarked'] > 0:
        print(f"\nPERFORMANCE BENCHMARK:")
        print(f"  Models benchmarked: {benchmark['total_models_benchmarked']}")
        print(f"  CDO faster: {benchmark['cdo_faster_count']} models")
        print(f"  ESMF faster: {benchmark['esmf_faster_count']} models")
        
        if benchmark['avg_esmf_time'] > 0 and benchmark['avg_cdo_time'] > 0:
            avg_ratio = benchmark['avg_esmf_time'] / benchmark['avg_cdo_time']
            faster_method = 'CDO' if avg_ratio > 1 else 'ESMF'
            avg_speedup = max(avg_ratio, 1/avg_ratio)
            
            print(f"  Average execution time:")
            print(f"    ESMF: {benchmark['avg_esmf_time']}s")
            print(f"    CDO:  {benchmark['avg_cdo_time']}s")
            print(f"  Overall winner: {faster_method} ({avg_speedup:.2f}x faster on average)")
            
            if benchmark['max_speedup_cdo'] > 0:
                print(f"  Max CDO speedup: {benchmark['max_speedup_cdo']:.2f}x")
            if benchmark['max_speedup_esmf'] > 0:
                print(f"  Max ESMF speedup: {benchmark['max_speedup_esmf']:.2f}x")
    
    print(f"\nNext steps:")
    print(f"  1. Analyze results: python analyze_cdo_esmf_results.py {report_file}")
    if args.test_single:
        print(f"  2. Run full test: python cdo-esmf-comparison.py")
    print(f"  3. Check the detailed report and plots in the analysis output")
    print()
    print("Troubleshooting tips:")
    print("  - Use --debug for more verbose output")
    print("  - Use --test-setup-only to check configuration")
    print("  - Check that data paths in config file are correct")
    print("  - Ensure CDO and required Python packages are installed")

if __name__ == "__main__":
    main()
