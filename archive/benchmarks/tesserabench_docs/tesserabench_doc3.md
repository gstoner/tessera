# TesseraBench - Document 3: Command Line Interface and Automation

This document details the command-line interface, automation features, and CI/CD integration for TesseraBench, enabling comprehensive benchmarking workflows from development to production.

## Command Line Interface

### Core CLI Design

```python
import click
import json
import yaml
from pathlib import Path
from typing import Optional, List, Dict, Any
import sys
import os

@click.group()
@click.version_option()
@click.option('--config', '-c', type=click.Path(exists=True), 
              help='Path to configuration file')
@click.option('--verbose', '-v', count=True, help='Increase verbosity')
@click.option('--quiet', '-q', is_flag=True, help='Suppress output except errors')
@click.pass_context
def cli(ctx: click.Context, config: Optional[str], verbose: int, quiet: bool):
    """
    TesseraBench - Comprehensive benchmarking suite for Tessera GPU kernels.
    
    TesseraBench provides systematic performance evaluation across the complete
    Tessera compilation pipeline, from Graph IR to Target IR execution.
    """
    
    # Initialize context
    ctx.ensure_object(dict)
    ctx.obj['verbose'] = verbose
    ctx.obj['quiet'] = quiet
    
    # Load configuration
    if config:
        ctx.obj['config'] = load_config_file(config)
    else:
        ctx.obj['config'] = load_default_config()
        
    # Setup logging
    setup_logging(verbose, quiet)

def load_config_file(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML or JSON file"""
    
    config_path = Path(config_path)
    
    if config_path.suffix.lower() in ['.yaml', '.yml']:
        with open(config_path) as f:
            return yaml.safe_load(f)
    elif config_path.suffix.lower() == '.json':
        with open(config_path) as f:
            return json.load(f)
    else:
        raise click.BadParameter(f"Unsupported config format: {config_path.suffix}")

def load_default_config() -> Dict[str, Any]:
    """Load default configuration"""
    return {
        'hardware': {
            'gpu_arch': 'auto',  # Auto-detect
            'gpu_count': 'auto',
            'memory_size_gb': 'auto',
            'memory_bandwidth_gbps': 'auto'
        },
        'benchmarking': {
            'warmup_iterations': 5,
            'timing_iterations': 100,
            'statistical_significance': 0.95,
            'max_variance_percent': 5.0
        },
        'precision_policies': ['bf16@accum(fp32)'],
        'optimization_levels': ['O3'],
        'output_formats': ['json', 'csv'],
        'autotuning_enabled': True
    }

### Benchmark Execution Commands

@cli.command()
@click.option('--benchmark', '-b', multiple=True, 
              help='Specific benchmarks to run (can be used multiple times)')
@click.option('--precision', '-p', multiple=True,
              help='Precision policies to test')
@click.option('--problem-size', multiple=True,
              help='Specific problem sizes (format: key=value)')
@click.option('--output', '-o', type=click.Path(), default='results.json',
              help='Output file for results')
@click.option('--format', 'output_format', 
              type=click.Choice(['json', 'csv', 'html', 'markdown']),
              default='json', help='Output format')
@click.option('--compare-baseline', type=click.Path(exists=True),
              help='Compare against baseline results file')
@click.option('--regression-threshold', type=float, default=5.0,
              help='Regression detection threshold (percent)')
@click.option('--timeout', type=int, default=3600,
              help='Timeout for entire benchmark suite (seconds)')
@click.option('--continue-on-error', is_flag=True,
              help='Continue running other benchmarks if one fails')
@click.pass_context
def run(ctx: click.Context, benchmark: tuple, precision: tuple, problem_size: tuple,
        output: str, output_format: str, compare_baseline: Optional[str],
        regression_threshold: float, timeout: int, continue_on_error: bool):
    """
    Run TesseraBench benchmark suite.
    
    Examples:
        tesserabench run                                    # Run all benchmarks
        tesserabench run -b gemm -b flash_attention         # Run specific benchmarks
        tesserabench run -p fp16 -p bf16                   # Test specific precisions
        tesserabench run --compare-baseline baseline.json   # Compare against baseline
    """
    
    config = ctx.obj['config']
    verbose = ctx.obj['verbose']
    
    try:
        # Initialize TesseraBench
        tessera_runtime = initialize_tessera_runtime(config)
        bench_core = TesseraBenchCore(tessera_runtime)
        
        # Register benchmarks
        benchmark_suite = get_benchmark_suite(list(benchmark) if benchmark else None)
        for bench in benchmark_suite:
            bench_core.register_benchmark(bench)
            
        # Create benchmark configuration
        benchmark_config = create_benchmark_config(config, precision, problem_size)
        
        # Run benchmarks with timeout
        click.echo(f"Running {len(benchmark_suite)} benchmarks...")
        
        with click.progressbar(benchmark_suite, label='Benchmarking') as progress:
            results = run_benchmarks_with_timeout(
                bench_core, [b.get_name() for b in progress], 
                benchmark_config, timeout, continue_on_error
            )
            
        # Save results
        save_results(results, output, output_format)
        
        # Compare against baseline if provided
        if compare_baseline:
            comparison_results = compare_against_baseline(results, compare_baseline, regression_threshold)
            display_comparison_results(comparison_results)
            
            # Exit with error code if regressions detected
            if any(r.get('is_regression', False) for r in comparison_results.values()):
                click.echo("⚠️  Performance regressions detected!", err=True)
                sys.exit(1)
                
        click.echo(f"✅ Benchmarking completed. Results saved to {output}")
        
    except Exception as e:
        click.echo(f"❌ Benchmarking failed: {str(e)}", err=True)
        if verbose > 0:
            import traceback
            traceback.print_exc()
        sys.exit(1)

@cli.command()
@click.option('--benchmark', '-b', required=True,
              help='Benchmark name to profile')
@click.option('--problem-size', required=True,
              help='Problem size specification (JSON format)')
@click.option('--precision', default='bf16@accum(fp32)',
              help='Precision policy')
@click.option('--profiler', type=click.Choice(['nsys', 'ncu', 'nvprof']),
              default='nsys', help='Profiler to use')
@click.option('--output-dir', type=click.Path(), default='./profiling_results',
              help='Directory for profiling output')
@click.pass_context
def profile(ctx: click.Context, benchmark: str, problem_size: str, precision: str,
           profiler: str, output_dir: str):
    """
    Profile a specific benchmark with detailed hardware metrics.
    
    Examples:
        tesserabench profile -b gemm --problem-size '{"M":1024,"N":1024,"K":1024}'
        tesserabench profile -b flash_attention --profiler ncu --problem-size '{"batch_size":8}'
    """
    
    config = ctx.obj['config']
    
    try:
        # Parse problem size
        import json
        problem_size_dict = json.loads(problem_size)
        
        # Setup profiling environment
        profiling_config = setup_profiling_environment(profiler, output_dir)
        
        # Initialize benchmark
        tessera_runtime = initialize_tessera_runtime(config)
        bench_core = TesseraBenchCore(tessera_runtime)
        
        # Get specific benchmark
        benchmark_instance = get_benchmark_by_name(benchmark)
        bench_core.register_benchmark(benchmark_instance)
        
        # Create configuration
        benchmark_config = BenchmarkConfig(
            name=f"profile_{benchmark}",
            description=f"Profiling run for {benchmark}",
            mode=BenchmarkMode.PROFILING,
            measurements=[MeasurementType.LATENCY, MeasurementType.COMPUTE_UTILIZATION],
            hardware=detect_hardware_config(),
            precision_policies=[precision],
            warmup_iterations=2,  # Fewer iterations for profiling
            timing_iterations=5
        )
        
        click.echo(f"🔍 Profiling {benchmark} with {profiler}...")
        
        # Run with profiling
        with profiling_context(profiling_config):
            result = bench_core.run_benchmark(benchmark, benchmark_config)
            
        # Generate profiling report
        profiling_report = generate_profiling_report(result, profiling_config)
        
        click.echo(f"✅ Profiling completed. Results in {output_dir}")
        click.echo(f"📊 Report: {profiling_report}")
        
    except Exception as e:
        click.echo(f"❌ Profiling failed: {str(e)}", err=True)
        sys.exit(1)

### Analysis and Comparison Commands

@cli.command()
@click.argument('results_files', nargs=-1, type=click.Path(exists=True), required=True)
@click.option('--output', '-o', type=click.Path(), default='comparison.html',
              help='Output file for comparison report')
@click.option('--format', 'output_format',
              type=click.Choice(['html', 'json', 'csv', 'markdown']),
              default='html', help='Output format')
@click.option('--metric', type=click.Choice(['latency', 'throughput', 'efficiency']),
              default='throughput', help='Primary metric for comparison')
@click.option('--statistical-test', 
              type=click.Choice(['auto', 'ttest', 'mannwhitney']),
              default='auto', help='Statistical test for significance')
@click.pass_context  
def compare(ctx: click.Context, results_files: tuple, output: str, output_format: str,
           metric: str, statistical_test: str):
    """
    Compare multiple benchmark result files.
    
    Examples:
        tesserabench compare results1.json results2.json
        tesserabench compare *.json --format html --output comparison_report.html
        tesserabench compare baseline.json current.json --metric efficiency
    """
    
    try:
        # Load all result files
        all_results = {}
        for file_path in results_files:
            file_name = Path(file_path).stem
            results = load_results_file(file_path)
            all_results[file_name] = results
            
        click.echo(f"📊 Comparing {len(results_files)} result files...")
        
        # Generate comparison analysis
        comparison_engine = ComparisonEngine()
        comparison_results = comparison_engine.compare_multiple_results(
            all_results, primary_metric=metric, statistical_test=statistical_test
        )
        
        # Generate report
        report_generator = ReportGenerator()
        report_generator.generate_comparison_report(
            comparison_results, output, output_format
        )
        
        # Display summary
        display_comparison_summary(comparison_results)
        
        click.echo(f"✅ Comparison completed. Report saved to {output}")
        
    except Exception as e:
        click.echo(f"❌ Comparison failed: {str(e)}", err=True)
        sys.exit(1)

@cli.command()
@click.argument('results_file', type=click.Path(exists=True))
@click.option('--output', '-o', type=click.Path(), default='analysis.html',
              help='Output file for analysis report')
@click.option('--format', 'output_format',
              type=click.Choice(['html', 'json', 'markdown']),
              default='html', help='Output format')
@click.option('--include-plots', is_flag=True,
              help='Include performance plots in analysis')
@click.option('--statistical-analysis', is_flag=True,
              help='Include detailed statistical analysis')
@click.pass_context
def analyze(ctx: click.Context, results_file: str, output: str, output_format: str,
           include_plots: bool, statistical_analysis: bool):
    """
    Analyze benchmark results with detailed statistical analysis.
    
    Examples:
        tesserabench analyze results.json
        tesserabench analyze results.json --include-plots --statistical-analysis
    """
    
    try:
        # Load results
        results = load_results_file(results_file)
        
        click.echo("📈 Analyzing benchmark results...")
        
        # Generate analysis
        analysis_engine = AnalysisEngine()
        analysis_results = analysis_engine.comprehensive_analysis(
            results, include_plots=include_plots, 
            statistical_analysis=statistical_analysis
        )
        
        # Generate report
        report_generator = ReportGenerator()
        report_generator.generate_analysis_report(
            analysis_results, output, output_format
        )
        
        # Display key insights
        display_analysis_insights(analysis_results)
        
        click.echo(f"✅ Analysis completed. Report saved to {output}")
        
    except Exception as e:
        click.echo(f"❌ Analysis failed: {str(e)}", err=True)
        sys.exit(1)

### Configuration and Setup Commands

@cli.command()
@click.option('--output', '-o', type=click.Path(), default='tesserabench_config.yaml',
              help='Output configuration file')
@click.option('--interactive', '-i', is_flag=True,
              help='Interactive configuration wizard')
@click.pass_context
def init_config(ctx: click.Context, output: str, interactive: bool):
    """
    Initialize TesseraBench configuration file.
    
    Examples:
        tesserabench init-config                          # Create default config
        tesserabench init-config --interactive            # Interactive wizard
        tesserabench init-config --output my_config.yaml # Custom output path
    """
    
    try:
        if interactive:
            config = interactive_config_wizard()
        else:
            config = generate_default_config()
            
        # Auto-detect hardware if needed
        if config['hardware']['gpu_arch'] == 'auto':
            config['hardware'] = detect_hardware_config().__dict__
            
        # Save configuration
        with open(output, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, indent=2)
            
        click.echo(f"✅ Configuration saved to {output}")
        
        # Validate configuration
        validation_result = validate_config(config)
        if not validation_result.valid:
            click.echo("⚠️  Configuration validation warnings:")
            for warning in validation_result.warnings:
                click.echo(f"  - {warning}")
                
    except Exception as e:
        click.echo(f"❌ Configuration initialization failed: {str(e)}", err=True)
        sys.exit(1)

def interactive_config_wizard() -> Dict[str, Any]:
    """Interactive configuration wizard"""
    
    config = {}
    
    click.echo("🚀 TesseraBench Configuration Wizard")
    click.echo("="*40)
    
    # Hardware detection
    click.echo("\n📡 Hardware Detection")
    if click.confirm("Auto-detect hardware configuration?", default=True):
        hardware = detect_hardware_config()
        config['hardware'] = hardware.__dict__
        click.echo(f"  Detected: {hardware.gpu_arch} with {hardware.gpu_count} GPUs")
    else:
        config['hardware'] = {}
        config['hardware']['gpu_arch'] = click.prompt("GPU Architecture (e.g., sm_90)", 
                                                     default="sm_80")
        config['hardware']['gpu_count'] = click.prompt("Number of GPUs", type=int, default=1)
        config['hardware']['memory_size_gb'] = click.prompt("Memory per GPU (GB)", 
                                                           type=int, default=80)
                                                           
    # Benchmark selection
    click.echo("\n🎯 Benchmark Selection")
    all_benchmarks = [b().get_name() for b in get_default_benchmark_suite()]
    
    if click.confirm("Include all benchmarks?", default=True):
        config['benchmarks'] = all_benchmarks
    else:
        config['benchmarks'] = []
        for benchmark in all_benchmarks:
            if click.confirm(f"  Include {benchmark}?", default=True):
                config['benchmarks'].append(benchmark)
                
    # Precision policies
    click.echo("\n🎛️  Precision Policies")
    available_precisions = [
        "fp32", "fp16", "bf16", 
        "fp8_e4m3@accum(fp32)", "fp8_e5m2@accum(fp32)"
    ]
    
    config['precision_policies'] = []
    for precision in available_precisions:
        if click.confirm(f"  Test {precision}?", 
                        default=(precision == "bf16@accum(fp32)")):
            config['precision_policies'].append(precision)
            
    # Performance settings
    click.echo("\n⚡ Performance Settings")
    config['benchmarking'] = {
        'warmup_iterations': click.prompt("Warmup iterations", type=int, default=5),
        'timing_iterations': click.prompt("Timing iterations", type=int, default=100),
        'autotuning_enabled': click.confirm("Enable autotuning?", default=True),
        'timeout_seconds': click.prompt("Benchmark timeout (seconds)", 
                                       type=int, default=3600)
    }
    
    # Output settings  
    click.echo("\n📄 Output Settings")
    config['output'] = {
        'default_format': click.prompt("Default output format", 
                                     default="json",
                                     type=click.Choice(['json', 'csv', 'html'])),
        'include_raw_data': click.confirm("Include raw measurement data?", default=False),
        'generate_plots': click.confirm("Generate performance plots?", default=True)
    }
    
    click.echo("\n✅ Configuration wizard completed!")
    
    return config

@cli.command()
@click.pass_context
def system_info(ctx: click.Context):
    """
    Display system information and TesseraBench environment.
    """
    
    try:
        # Detect hardware
        hardware = detect_hardware_config()
        
        # Get Tessera version info
        tessera_info = get_tessera_version_info()
        
        # Get system info
        system_info_dict = get_system_info()
        
        click.echo("🖥️  TesseraBench System Information")
        click.echo("="*40)
        
        click.echo(f"\n📊 TesseraBench Version: {get_tesserabench_version()}")
        click.echo(f"🔧 Tessera Version: {tessera_info.version}")
        click.echo(f"🐍 Python Version: {system_info_dict.python_version}")
        
        click.echo(f"\n🖱️  Hardware Configuration:")
        click.echo(f"  GPU Architecture: {hardware.gpu_arch}")
        click.echo(f"  GPU Count: {hardware.gpu_count}")
        click.echo(f"  Memory per GPU: {hardware.memory_size_gb} GB")
        click.echo(f"  Memory Bandwidth: {hardware.memory_bandwidth_gbps} GB/s")
        click.echo(f"  Compute Capability: {hardware.compute_capability}")
        click.echo(f"  Tensor Cores: {'Yes' if hardware.tensor_cores else 'No'}")
        
        if hardware.gpu_count > 1:
            click.echo(f"  NVLink Bandwidth: {hardware.nvlink_bandwidth_gbps} GB/s")
            if hardware.nvswitch_bandwidth_gbps > 0:
                click.echo(f"  NVSwitch Bandwidth: {hardware.nvswitch_bandwidth_gbps} GB/s")
                
        click.echo(f"\n🔧 Environment:")
        click.echo(f"  CUDA Version: {system_info_dict.cuda_version}")
        click.echo(f"  Driver Version: {system_info_dict.driver_version}")
        click.echo(f"  NCCL Version: {system_info_dict.nccl_version}")
        
        click.echo(f"\n📁 Paths:")
        click.echo(f"  Config Directory: {get_config_directory()}")
        click.echo(f"  Cache Directory: {get_cache_directory()}")
        click.echo(f"  Results Directory: {get_results_directory()}")
        
        # Validate environment
        validation_result = validate_environment()
        if validation_result.valid:
            click.echo("\n✅ Environment validation passed")
        else:
            click.echo("\n⚠️  Environment validation warnings:")
            for warning in validation_result.warnings:
                click.echo(f"  - {warning}")
                
            if validation_result.errors:
                click.echo("\n❌ Environment validation errors:")
                for error in validation_result.errors:
                    click.echo(f"  - {error}")
                    
    except Exception as e:
        click.echo(f"❌ System info failed: {str(e)}", err=True)
        sys.exit(1)

### Utility Commands

@cli.command()
@click.option('--cache-dir', type=click.Path(), 
              help='Specific cache directory to clean (default: all)')
@click.option('--older-than', type=int, default=30,
              help='Remove cache files older than N days')
@click.option('--dry-run', is_flag=True,
              help='Show what would be deleted without actually deleting')
@click.pass_context
def clean_cache(ctx: click.Context, cache_dir: Optional[str], older_than: int, dry_run: bool):
    """
    Clean TesseraBench cache files (autotuning, compilation artifacts, etc.).
    
    Examples:
        tesserabench clean-cache                    # Clean all cache
        tesserabench clean-cache --older-than 7    # Clean files older than 7 days
        tesserabench clean-cache --dry-run          # Preview what would be deleted
    """
    
    try:
        if cache_dir:
            cache_dirs = [Path(cache_dir)]
        else:
            cache_dirs = [
                Path.home() / '.tesserabench' / 'autotune',
                Path.home() / '.tesserabench' / 'compilation_cache',
                Path.home() / '.tessera' / 'kernel_cache'
            ]
            
        total_files = 0
        total_size = 0
        
        for cache_path in cache_dirs:
            if not cache_path.exists():
                continue
                
            files_to_delete = []
            cutoff_time = time.time() - (older_than * 24 * 3600)
            
            for file_path in cache_path.rglob('*'):
                if file_path.is_file() and file_path.stat().st_mtime < cutoff_time:
                    files_to_delete.append(file_path)
                    total_size += file_path.stat().st_size
                    
            total_files += len(files_to_delete)
            
            if dry_run:
                click.echo(f"Would delete {len(files_to_delete)} files from {cache_path}")
                for file_path in files_to_delete[:5]:  # Show first 5
                    click.echo(f"  - {file_path}")
                if len(files_to_delete) > 5:
                    click.echo(f"  ... and {len(files_to_delete) - 5} more")
            else:
                for file_path in files_to_delete:
                    file_path.unlink()
                click.echo(f"Deleted {len(files_to_delete)} files from {cache_path}")
                
        size_mb = total_size / (1024 * 1024)
        
        if dry_run:
            click.echo(f"\n📊 Summary: Would delete {total_files} files ({size_mb:.1f} MB)")
        else:
            click.echo(f"\n✅ Cleaned {total_files} files ({size_mb:.1f} MB)")
            
    except Exception as e:
        click.echo(f"❌ Cache cleaning failed: {str(e)}", err=True)
        sys.exit(1)

@cli.command()
@click.option('--format', 'list_format', 
              type=click.Choice(['table', 'json', 'names-only']),
              default='table', help='Output format')
@click.option('--filter', 'benchmark_filter',
              help='Filter benchmarks by name pattern')
@click.pass_context
def list_benchmarks(ctx: click.Context, list_format: str, benchmark_filter: Optional[str]):
    """
    List all available benchmarks with descriptions.
    
    Examples:
        tesserabench list-benchmarks                    # Table format
        tesserabench list-benchmarks --format json     # JSON format  
        tesserabench list-benchmarks --filter gemm     # Filter by pattern
    """
    
    try:
        # Get all available benchmarks
        benchmark_suite = get_default_benchmark_suite()
        
        # Apply filter if specified
        if benchmark_filter:
            benchmark_suite = [b for b in benchmark_suite 
                             if benchmark_filter.lower() in b.get_name().lower()]
            
        if list_format == 'names-only':
            for benchmark in benchmark_suite:
                click.echo(benchmark.get_name())
                
        elif list_format == 'json':
            benchmark_info = []
            for benchmark in benchmark_suite:
                info = {
                    'name': benchmark.get_name(),
                    'description': benchmark.get_description(),
                    'supported_precisions': benchmark.get_supported_precisions(),
                    'problem_size_count': len(benchmark.get_problem_sizes())
                }
                benchmark_info.append(info)
                
            click.echo(json.dumps(benchmark_info, indent=2))
            
        else:  # table format
            click.echo("📋 Available Benchmarks")
            click.echo("=" * 80)
            click.echo(f"{'Name':<20} {'Description':<40} {'Precisions':<15}")
            click.echo("-" * 80)
            
            for benchmark in benchmark_suite:
                name = benchmark.get_name()
                desc = benchmark.get_description()[:38] + "..." if len(benchmark.get_description()) > 38 else benchmark.get_description()
                precisions = f"{len(benchmark.get_supported_precisions())} types"
                
                click.echo(f"{name:<20} {desc:<40} {precisions:<15}")
                
            click.echo(f"\nTotal: {len(benchmark_suite)} benchmarks")
            
    except Exception as e:
        click.echo(f"❌ Listing benchmarks failed: {str(e)}", err=True)
        sys.exit(1)

## Automation and CI/CD Integration

### GitHub Actions Integration

def create_github_actions_workflow() -> str:
    """Generate GitHub Actions workflow for automated benchmarking"""
    
    return """
name: TesseraBench Performance Testing

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]
  schedule:
    # Run daily performance regression tests
    - cron: '0 2 * * *'

env:
  TESSERA_CACHE_DIR: ~/.tessera_cache
  TESSERABENCH_CONFIG: .github/tesserabench_config.yaml

jobs:
  performance-test:
    runs-on: [self-hosted, gpu, tesla-v100]  # Use self-hosted GPU runners
    
    strategy:
      matrix:
        precision: [fp16, bf16, fp32]
        gpu_count: [1, 2, 4]
        
    steps:
    - uses: actions/checkout@v4
    
    - name: Setup Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'
        
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install tesserabench tessera-gpu
        
    - name: Cache TesseraBench artifacts
      uses: actions/cache@v3
      with:
        path: |
          ~/.tesserabench
          ~/.tessera
        key: tesserabench-${{ runner.os }}-${{ hashFiles('**/tesserabench_config.yaml') }}
        
    - name: System info
      run: tesserabench system-info
      
    - name: Run core benchmarks
      run: |
        tesserabench run \\
          --benchmark gemm \\
          --benchmark flash_attention \\
          --benchmark layer_norm \\
          --precision ${{ matrix.precision }} \\
          --output results-${{ matrix.precision }}-${{ matrix.gpu_count }}gpu.json \\
          --timeout 1800
          
    - name: Compare against baseline
      if: github.event_name == 'pull_request'
      run: |
        # Download baseline results from main branch
        gh api /repos/${{ github.repository }}/contents/baseline_results.json \\
          --jq '.content' | base64 -d > baseline.json
          
        tesserabench compare \\
          baseline.json \\
          results-${{ matrix.precision }}-${{ matrix.gpu_count }}gpu.json \\
          --format json \\
          --output comparison.json
          
        # Check for regressions
        python .github/scripts/check_regression.py comparison.json
        
    - name: Generate performance report
      run: |
        tesserabench analyze \\
          results-${{ matrix.precision }}-${{ matrix.gpu_count }}gpu.json \\
          --format html \\
          --include-plots \\
          --output performance-report.html
          
    - name: Upload results
      uses: actions/upload-artifact@v3
      with:
        name: benchmark-results-${{ matrix.precision }}-${{ matrix.gpu_count }}gpu
        path: |
          results-*.json
          performance-report.html
          comparison.json
        retention-days: 30
        
    - name: Update baseline (main branch only)
      if: github.ref == 'refs/heads/main' && github.event_name == 'push'
      run: |
        # Update baseline results
        cp results-${{ matrix.precision }}-${{ matrix.gpu_count }}gpu.json \\
           baseline_results-${{ matrix.precision }}-${{ matrix.gpu_count }}gpu.json
           
        git config --local user.email "action@github