# TesseraBench - Document 7: Production Deployment and CI/CD Integration

This document covers TesseraBench's production deployment capabilities, including CI/CD pipeline integration, automated performance regression detection, cloud deployment strategies, and enterprise monitoring integration.

## Overview

TesseraBench is designed to seamlessly integrate into production software development workflows, providing automated performance validation, regression detection, and continuous performance monitoring. This document explores how to deploy TesseraBench in enterprise environments and integrate it with modern CI/CD pipelines.

### Production Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                     TesseraBench Production Stack                  │
├─────────────────────────────────────────────────────────────────────┤
│   CI/CD        │  Regression   │  Cloud        │  Enterprise      │
│ Integration    │   Detection   │ Deployment    │  Monitoring      │
├────────────────┼───────────────┼───────────────┼──────────────────┤
│              Performance Data Pipeline & Analytics                  │
├─────────────────────────────────────────────────────────────────────┤
│  GitHub Actions │  Jenkins     │   Docker      │   Kubernetes     │
│   Integration   │ Integration  │ Containers    │  Orchestration   │
├─────────────────────────────────────────────────────────────────────┤
│          Database & Storage Infrastructure                          │
├─────────────────────────────────────────────────────────────────────┤
│  PostgreSQL  │  InfluxDB   │  S3/GCS     │  Prometheus │  Grafana │
│  Benchmark   │ Time Series │ Artifact    │  Metrics    │ Dashboards│
│   Results    │   Data      │  Storage    │ Collection  │           │
└─────────────────────────────────────────────────────────────────────┘
```

## CI/CD Pipeline Integration

### GitHub Actions Integration

```yaml
# .github/workflows/tesserabench-performance.yml
name: TesseraBench Performance Validation

on:
  pull_request:
    branches: [main, develop]
    paths: ['src/kernels/**', 'src/tessera/**']
  push:
    branches: [main]
  schedule:
    # Run nightly performance benchmarks
    - cron: '0 2 * * *'

jobs:
  performance-validation:
    runs-on: [self-hosted, gpu, nvl72]
    
    strategy:
      matrix:
        gpu-config:
          - single-h100
          - multi-gpu-8x
          - nvl72-full
        test-suite:
          - smoke-tests
          - full-benchmarks
          - regression-tests
    
    steps:
    - name: Checkout code
      uses: actions/checkout@v4
      with:
        fetch-depth: 0  # Need full history for regression analysis
    
    - name: Setup TesseraBench Environment
      uses: ./.github/actions/setup-tesserabench
      with:
        gpu-config: ${{ matrix.gpu-config }}
        install-dependencies: true
        setup-monitoring: true
    
    - name: Run Performance Benchmarks
      id: benchmarks
      run: |
        python -m tesserabench.ci \
          --config configs/ci/${{ matrix.gpu-config }}.yaml \
          --test-suite ${{ matrix.test-suite }} \
          --output-format github-actions \
          --baseline-branch main \
          --regression-threshold 0.05 \
          --timeout 3600
    
    - name: Performance Regression Analysis
      uses: ./.github/actions/performance-regression
      with:
        benchmark-results: ${{ steps.benchmarks.outputs.results-file }}
        baseline-branch: main
        regression-threshold: 5%
        fail-on-regression: ${{ github.event_name == 'pull_request' }}
    
    - name: Upload Benchmark Results
      uses: actions/upload-artifact@v4
      with:
        name: benchmark-results-${{ matrix.gpu-config }}-${{ github.sha }}
        path: |
          benchmark-results/
          performance-reports/
          profiling-data/
        retention-days: 30
    
    - name: Update Performance Database
      if: github.event_name == 'push' && github.ref == 'refs/heads/main'
      run: |
        python -m tesserabench.database.upload \
          --results benchmark-results/ \
          --commit-sha ${{ github.sha }} \
          --branch ${{ github.ref_name }} \
          --database-url ${{ secrets.TESSERABENCH_DB_URL }}
    
    - name: Generate Performance Report
      uses: ./.github/actions/generate-performance-report
      with:
        results-path: benchmark-results/
        output-path: performance-report.md
        include-plots: true
        compare-with-baseline: true
    
    - name: Comment Performance Results
      if: github.event_name == 'pull_request'
      uses: actions/github-script@v7
      with:
        script: |
          const fs = require('fs');
          const report = fs.readFileSync('performance-report.md', 'utf8');
          
          github.rest.issues.createComment({
            issue_number: context.issue.number,
            owner: context.repo.owner,
            repo: context.repo.repo,
            body: `## 🚀 Performance Benchmark Results\n\n${report}`
          });

  performance-alerts:
    needs: performance-validation
    if: failure()
    runs-on: ubuntu-latest
    
    steps:
    - name: Send Slack Alert
      uses: 8398a7/action-slack@v3
      with:
        status: failure
        channel: '#gpu-performance'
        message: |
          🚨 Performance regression detected in ${{ github.repository }}
          Commit: ${{ github.sha }}
          Branch: ${{ github.ref_name }}
          View details: ${{ github.server_url }}/${{ github.repository }}/actions/runs/${{ github.run_id }}
      env:
        SLACK_WEBHOOK_URL: ${{ secrets.SLACK_WEBHOOK_URL }}
```

### Jenkins Pipeline Integration

```groovy
// Jenkinsfile for TesseraBench integration
pipeline {
    agent {
        label 'gpu-nodes && nvl72'
    }
    
    parameters {
        choice(
            name: 'BENCHMARK_SUITE',
            choices: ['smoke', 'regression', 'full', 'nightly'],
            description: 'Benchmark suite to run'
        )
        booleanParam(
            name: 'FAIL_ON_REGRESSION',
            defaultValue: true,
            description: 'Fail build on performance regression'
        )
        string(
            name: 'BASELINE_COMMIT',
            defaultValue: 'main',
            description: 'Baseline commit for comparison'
        )
    }
    
    environment {
        TESSERABENCH_CONFIG = "${WORKSPACE}/configs/jenkins/nvl72-config.yaml"
        CUDA_VISIBLE_DEVICES = "0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71"
        TESSERA_CACHE_DIR = "${WORKSPACE}/.tessera_cache"
    }
    
    stages {
        stage('Environment Setup') {
            steps {
                script {
                    // Setup GPU environment
                    sh '''
                        nvidia-smi
                        python -c "import torch; print(f'PyTorch CUDA: {torch.cuda.is_available()}')"
                        python -c "import cupy; print(f'CuPy version: {cupy.__version__}')"
                    '''
                }
                
                // Install TesseraBench and dependencies
                sh '''
                    pip install -e .
                    pip install tesserabench[all]
                    tesserabench --version
                '''
            }
        }
        
        stage('Baseline Performance Data') {
            when {
                not { params.BASELINE_COMMIT == 'skip' }
            }
            steps {
                script {
                    // Fetch baseline performance data
                    sh '''
                        python -m tesserabench.baseline \
                            --commit ${BASELINE_COMMIT} \
                            --output baseline-results.json \
                            --cache-duration 24h
                    '''
                }
            }
        }
        
        stage('Performance Benchmarks') {
            parallel {
                stage('Flash Attention Benchmarks') {
                    steps {
                        sh '''
                            python -m tesserabench.suites.attention \
                                --config ${TESSERABENCH_CONFIG} \
                                --suite ${BENCHMARK_SUITE} \
                                --output flash-attention-results.json \
                                --profiling-enabled
                        '''
                    }
                }
                
                stage('GEMM Benchmarks') {
                    steps {
                        sh '''
                            python -m tesserabench.suites.linear_algebra \
                                --config ${TESSERABENCH_CONFIG} \
                                --suite ${BENCHMARK_SUITE} \
                                --output gemm-results.json \
                                --include-mixed-precision
                        '''
                    }
                }
                
                stage('Distributed Benchmarks') {
                    steps {
                        sh '''
                            python -m tesserabench.suites.distributed \
                                --config ${TESSERABENCH_CONFIG} \
                                --suite ${BENCHMARK_SUITE} \
                                --output distributed-results.json \
                                --mesh-configs nvl72
                        '''
                    }
                }
            }
        }
        
        stage('Regression Analysis') {
            steps {
                script {
                    def regressionResults = sh(
                        script: '''
                            python -m tesserabench.regression \
                                --current-results flash-attention-results.json,gemm-results.json,distributed-results.json \
                                --baseline-results baseline-results.json \
                                --threshold 0.05 \
                                --output regression-report.json \
                                --format json
                        ''',
                        returnStdout: true
                    ).trim()
                    
                    def regression = readJSON text: regressionResults
                    
                    if (regression.has_regressions && params.FAIL_ON_REGRESSION) {
                        error("Performance regression detected: ${regression.summary}")
                    }
                }
            }
        }
        
        stage('Generate Reports') {
            steps {
                sh '''
                    python -m tesserabench.reports \
                        --results flash-attention-results.json,gemm-results.json,distributed-results.json \
                        --baseline baseline-results.json \
                        --output-dir performance-reports/ \
                        --formats html,pdf,json \
                        --include-plots \
                        --include-profiling-data
                '''
            }
        }
        
        stage('Archive Results') {
            steps {
                archiveArtifacts artifacts: '''
                    *-results.json,
                    regression-report.json,
                    performance-reports/**,
                    profiling-data/**
                ''', fingerprint: true
                
                publishHTML([
                    allowMissing: false,
                    alwaysLinkToLastBuild: true,
                    keepAll: true,
                    reportDir: 'performance-reports',
                    reportFiles: 'index.html',
                    reportName: 'Performance Report'
                ])
            }
        }
        
        stage('Database Upload') {
            when {
                branch 'main'
            }
            steps {
                sh '''
                    python -m tesserabench.database.upload \
                        --results *-results.json \
                        --commit-sha ${GIT_COMMIT} \
                        --branch ${BRANCH_NAME} \
                        --build-number ${BUILD_NUMBER} \
                        --database-url ${TESSERABENCH_DB_URL}
                '''
            }
        }
    }
    
    post {
        always {
            script {
                // Cleanup GPU memory
                sh 'nvidia-smi --gpu-reset || true'
            }
        }
        
        failure {
            emailext (
                subject: "Performance Build Failed: ${env.JOB_NAME} - ${env.BUILD_NUMBER}",
                body: '''
                    Performance benchmarks failed for ${env.JOB_NAME} build ${env.BUILD_NUMBER}.
                    
                    Build URL: ${env.BUILD_URL}
                    Git Commit: ${env.GIT_COMMIT}
                    
                    Please check the performance reports for details.
                ''',
                to: "${env.CHANGE_AUTHOR_EMAIL},gpu-performance-team@company.com"
            )
        }
    }
}
```

## Automated Performance Regression Detection

### Regression Detection Engine

```python
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import json
import logging
from dataclasses import dataclass
from enum import Enum

class RegressionSeverity(Enum):
    NONE = "none"
    MINOR = "minor"
    MAJOR = "major"
    CRITICAL = "critical"

@dataclass
class RegressionResult:
    metric_name: str
    baseline_value: float
    current_value: float
    percentage_change: float
    severity: RegressionSeverity
    confidence: float
    statistical_significance: bool

class PerformanceRegressionDetector:
    """Advanced regression detection for TesseraBench results."""
    
    def __init__(self, config: Dict = None):
        self.config = config or self._get_default_config()
        self.logger = logging.getLogger(__name__)
        
    def _get_default_config(self) -> Dict:
        return {
            'thresholds': {
                'minor_regression': 0.02,    # 2%
                'major_regression': 0.05,    # 5%
                'critical_regression': 0.10, # 10%
            },
            'statistical': {
                'min_samples': 5,
                'confidence_level': 0.95,
                'use_statistical_tests': True
            },
            'metrics': {
                'primary': ['execution_time_ms', 'tflops_achieved', 'memory_bandwidth_gbps'],
                'secondary': ['occupancy_achieved', 'register_usage', 'shared_memory_usage']
            }
        }
    
    def detect_regressions(self, current_results: Dict, baseline_results: Dict) -> List[RegressionResult]:
        """Detect performance regressions between current and baseline results."""
        
        regressions = []
        
        # Analyze primary metrics
        for metric in self.config['metrics']['primary']:
            regression = self._analyze_metric_regression(
                metric, current_results, baseline_results, primary=True
            )
            if regression and regression.severity != RegressionSeverity.NONE:
                regressions.append(regression)
        
        # Analyze secondary metrics with relaxed thresholds
        for metric in self.config['metrics']['secondary']:
            regression = self._analyze_metric_regression(
                metric, current_results, baseline_results, primary=False
            )
            if regression and regression.severity != RegressionSeverity.NONE:
                regressions.append(regression)
        
        return regressions
    
    def _analyze_metric_regression(self, metric_name: str, current: Dict, baseline: Dict, 
                                  primary: bool = True) -> Optional[RegressionResult]:
        """Analyze regression for a specific metric."""
        
        try:
            # Extract metric values
            current_values = self._extract_metric_values(current, metric_name)
            baseline_values = self._extract_metric_values(baseline, metric_name)
            
            if not current_values or not baseline_values:
                self.logger.warning(f"Missing data for metric: {metric_name}")
                return None
            
            # Calculate statistics
            current_mean = np.mean(current_values)
            baseline_mean = np.mean(baseline_values)
            
            if baseline_mean == 0:
                return None
            
            percentage_change = (current_mean - baseline_mean) / baseline_mean
            
            # Determine severity
            severity = self._classify_regression_severity(percentage_change, primary)
            
            # Calculate confidence and statistical significance
            confidence, is_significant = self._calculate_statistical_significance(
                current_values, baseline_values
            )
            
            return RegressionResult(
                metric_name=metric_name,
                baseline_value=baseline_mean,
                current_value=current_mean,
                percentage_change=percentage_change,
                severity=severity,
                confidence=confidence,
                statistical_significance=is_significant
            )
            
        except Exception as e:
            self.logger.error(f"Error analyzing metric {metric_name}: {e}")
            return None
    
    def _extract_metric_values(self, results: Dict, metric_name: str) -> List[float]:
        """Extract metric values from nested results dictionary."""
        
        values = []
        
        def extract_from_dict(d, path=""):
            if isinstance(d, dict):
                for key, value in d.items():
                    if key == metric_name and isinstance(value, (int, float)):
                        values.append(float(value))
                    elif isinstance(value, (dict, list)):
                        extract_from_dict(value, f"{path}.{key}" if path else key)
            elif isinstance(d, list):
                for i, item in enumerate(d):
                    extract_from_dict(item, f"{path}[{i}]")
        
        extract_from_dict(results)
        return values
    
    def _classify_regression_severity(self, percentage_change: float, primary: bool) -> RegressionSeverity:
        """Classify regression severity based on percentage change."""
        
        # For performance metrics, negative changes are regressions
        abs_change = abs(percentage_change)
        
        # Adjust thresholds for secondary metrics
        thresholds = self.config['thresholds'].copy()
        if not primary:
            for key in thresholds:
                thresholds[key] *= 1.5  # More lenient for secondary metrics
        
        if abs_change >= thresholds['critical_regression']:
            return RegressionSeverity.CRITICAL
        elif abs_change >= thresholds['major_regression']:
            return RegressionSeverity.MAJOR
        elif abs_change >= thresholds['minor_regression']:
            return RegressionSeverity.MINOR
        else:
            return RegressionSeverity.NONE
    
    def _calculate_statistical_significance(self, current_values: List[float], 
                                           baseline_values: List[float]) -> Tuple[float, bool]:
        """Calculate statistical significance using t-test."""
        
        if not self.config['statistical']['use_statistical_tests']:
            return 0.5, False
        
        if (len(current_values) < self.config['statistical']['min_samples'] or 
            len(baseline_values) < self.config['statistical']['min_samples']):
            return 0.0, False
        
        try:
            from scipy import stats
            
            # Perform two-sample t-test
            t_statistic, p_value = stats.ttest_ind(current_values, baseline_values)
            
            confidence_level = self.config['statistical']['confidence_level']
            is_significant = p_value < (1 - confidence_level)
            confidence = 1 - p_value if p_value > 0 else 0.99
            
            return confidence, is_significant
            
        except ImportError:
            self.logger.warning("SciPy not available for statistical tests")
            return 0.5, False
        except Exception as e:
            self.logger.error(f"Error in statistical analysis: {e}")
            return 0.0, False
    
    def generate_regression_report(self, regressions: List[RegressionResult]) -> Dict:
        """Generate comprehensive regression report."""
        
        if not regressions:
            return {
                'has_regressions': False,
                'summary': 'No performance regressions detected',
                'total_regressions': 0,
                'by_severity': {},
                'recommendations': []
            }
        
        # Categorize by severity
        by_severity = {}
        for severity in RegressionSeverity:
            by_severity[severity.value] = [r for r in regressions if r.severity == severity]
        
        # Generate summary
        critical_count = len(by_severity[RegressionSeverity.CRITICAL.value])
        major_count = len(by_severity[RegressionSeverity.MAJOR.value])
        minor_count = len(by_severity[RegressionSeverity.MINOR.value])
        
        if critical_count > 0:
            summary = f"{critical_count} critical regression(s) detected"
        elif major_count > 0:
            summary = f"{major_count} major regression(s) detected"
        else:
            summary = f"{minor_count} minor regression(s) detected"
        
        # Generate recommendations
        recommendations = self._generate_recommendations(regressions)
        
        return {
            'has_regressions': True,
            'summary': summary,
            'total_regressions': len(regressions),
            'by_severity': {
                'critical': len(by_severity[RegressionSeverity.CRITICAL.value]),
                'major': len(by_severity[RegressionSeverity.MAJOR.value]),
                'minor': len(by_severity[RegressionSeverity.MINOR.value])
            },
            'regressions': [
                {
                    'metric': r.metric_name,
                    'baseline_value': r.baseline_value,
                    'current_value': r.current_value,
                    'percentage_change': r.percentage_change * 100,
                    'severity': r.severity.value,
                    'confidence': r.confidence,
                    'statistically_significant': r.statistical_significance
                }
                for r in regressions
            ],
            'recommendations': recommendations
        }
    
    def _generate_recommendations(self, regressions: List[RegressionResult]) -> List[str]:
        """Generate actionable recommendations based on regressions."""
        
        recommendations = []
        
        # Analyze regression patterns
        critical_regressions = [r for r in regressions if r.severity == RegressionSeverity.CRITICAL]
        performance_regressions = [r for r in regressions if 'time' in r.metric_name.lower()]
        throughput_regressions = [r for r in regressions if 'tflops' in r.metric_name.lower() or 'bandwidth' in r.metric_name.lower()]
        resource_regressions = [r for r in regressions if 'register' in r.metric_name.lower() or 'memory' in r.metric_name.lower()]
        
        if critical_regressions:
            recommendations.append(
                f"⚠️  CRITICAL: {len(critical_regressions)} critical regression(s) require immediate attention"
            )
        
        if performance_regressions:
            recommendations.append(
                f"🐌 Performance: {len(performance_regressions)} execution time regression(s) detected. "
                "Check for algorithmic changes, compiler optimizations, or resource contention."
            )
        
        if throughput_regressions:
            recommendations.append(
                f"📉 Throughput: {len(throughput_regressions)} throughput regression(s) detected. "
                "Analyze memory access patterns, tensor core utilization, and parallelization efficiency."
            )
        
        if resource_regressions:
            recommendations.append(
                f"🔧 Resources: {len(resource_regressions)} resource usage regression(s) detected. "
                "Review register allocation, shared memory usage, and occupancy characteristics."
            )
        
        # General recommendations
        if len(regressions) > 5:
            recommendations.append(
                "🔍 Multiple regressions detected across different metrics. "
                "Consider comprehensive profiling to identify root cause."
            )
        
        return recommendations

# CLI interface for regression detection
class RegressionCLI:
    """Command-line interface for regression detection."""
    
    def __init__(self):
        self.detector = PerformanceRegressionDetector()
    
    def run_regression_analysis(self, current_file: str, baseline_file: str, 
                              output_file: str = None, threshold: float = 0.05):
        """Run regression analysis from command line."""
        
        try:
            # Load results
            with open(current_file, 'r') as f:
                current_results = json.load(f)
            
            with open(baseline_file, 'r') as f:
                baseline_results = json.load(f)
            
            # Update threshold if provided
            if threshold != 0.05:
                self.detector.config['thresholds']['major_regression'] = threshold
            
            # Detect regressions
            regressions = self.detector.detect_regressions(current_results, baseline_results)
            
            # Generate report
            report = self.detector.generate_regression_report(regressions)
            
            # Output results
            if output_file:
                with open(output_file, 'w') as f:
                    json.dump(report, f, indent=2)
                print(f"Regression report written to: {output_file}")
            else:
                print(json.dumps(report, indent=2))
            
            # Exit with appropriate code
            if report['has_regressions']:
                critical_count = report['by_severity']['critical']
                major_count = report['by_severity']['major']
                
                if critical_count > 0:
                    return 3  # Critical regressions
                elif major_count > 0:
                    return 2  # Major regressions
                else:
                    return 1  # Minor regressions
            else:
                return 0  # No regressions
                
        except Exception as e:
            print(f"Error in regression analysis: {e}")
            return 4  # Error

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="TesseraBench Regression Detection")
    parser.add_argument("--current", required=True, help="Current benchmark results file")
    parser.add_argument("--baseline", required=True, help="Baseline benchmark results file")
    parser.add_argument("--output", help="Output file for regression report")
    parser.add_argument("--threshold", type=float, default=0.05, help="Regression threshold")
    
    args = parser.parse_args()
    
    cli = RegressionCLI()
    exit_code = cli.run_regression_analysis(
        args.current, args.baseline, args.output, args.threshold
    )
    exit(exit_code)
```

## Cloud Deployment Architecture

### Docker Configuration

```dockerfile
# Dockerfile for TesseraBench production deployment
FROM nvidia/cuda:12.1-devel-ubuntu22.04 as base

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3-pip \
    git \
    wget \
    curl \
    build-essential \
    cmake \
    ninja-build \
    libssl-dev \
    libffi-dev \
    libnccl2 \
    libnccl-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt /tmp/requirements.txt
RUN pip3 install --no-cache-dir -r /tmp/requirements.txt

# Install TesseraBench and Tessera
COPY . /workspace/tesserabench
WORKDIR /workspace/tesserabench

RUN pip3 install -e .
RUN pip3 install tesserabench[all]

# Create non-root user
RUN useradd -m -u 1000 benchmark && \
    chown -R benchmark:benchmark /workspace
USER benchmark

# Set up environment
ENV PATH="/home/benchmark/.local/bin:${PATH}"
ENV TESSERA_CACHE_DIR="/workspace/.tessera_cache"
ENV TESSERABENCH_CONFIG_DIR="/workspace/configs"

# Create required directories
RUN mkdir -p /workspace/results /workspace/reports /workspace/logs

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python3 -c "import tesserabench; tesserabench.health_check()" || exit 1

# Default command
CMD ["python3", "-m", "tesserabench.server", "--host", "0.0.0.0", "--port", "8080"]

# Multi-stage build for production
FROM base as production

# Copy only necessary files
COPY --chown=benchmark:benchmark configs/ /workspace/configs/
COPY --chown=benchmark:benchmark scripts/ /workspace/scripts/

# Remove development dependencies
RUN pip3 uninstall -y pytest pytest-cov black isort mypy

# Set production environment
ENV TESSERABENCH_ENV=production
ENV PYTHONOPTIMIZE=1

EXPOSE 8080

# Production entrypoint
ENTRYPOINT ["python3", "-m", "tesserabench.production"]
```

### Kubernetes Deployment

```yaml
# kubernetes/tesserabench-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: tesserabench
  labels:
    app: tesserabench
    version: v1.0.0
spec:
  replicas: 2
  selector:
    matchLabels:
      app: tesserabench
  template:
    metadata:
      labels:
        app: tesserabench
    spec:
      nodeSelector:
        gpu-type: "h100"
        gpu-count: "8"
      
      tolerations:
      - key: "gpu-node"
        operator: "Equal"
        value: "true"
        effect: "NoSchedule"
      
      containers:
      - name: tesserabench
        image: tesserabench:latest
        imagePullPolicy: Always
        
        resources:
          requests:
            nvidia.com/gpu: 8
            cpu: "16"
            memory: "64Gi"
          limits:
            nvidia.com/gpu: 8
            cpu: "32"
            memory: "128Gi"
        
        env:
        - name: CUDA_VISIBLE_DEVICES
          value: "0,1,2,3,4,5,6,7"
        - name: TESSERABENCH_CONFIG
          value: "/config/production.yaml"
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: tesserabench-secrets
              key: database-url
        - name: MONITORING_ENDPOINT
          value: "http://prometheus:9090"
        
        volumeMounts:
        - name: config-volume
          mountPath: /config
          readOnly: true
        - name: results-volume
          mountPath: /workspace/results
        - name: cache-volume
          mountPath: /workspace/.tessera_cache
        
        ports:
        - containerPort: 8080
          name: http
        - containerPort: 9090
          name: metrics
        
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 10
          
        readinessProbe:
          httpGet:
            path: /ready
            port: 8080
          initialDelaySeconds: 15
          periodSeconds: 5
      
      volumes:
      - name: config-volume
        configMap:
          name: tesserabench-config
      - name: results-volume
        persistentVolumeClaim:
          claimName: tesserabench-results
      - name: cache-volume
        persistentVolumeClaim:
          claimName: tesserabench-cache

---
apiVersion: v1
kind: Service
metadata:
  name: tesserabench-service
  labels:
    app: tesserabench
spec:
  type: LoadBalancer
  ports:
  - port: 80
    targetPort: 8080
    protocol: TCP
    name: http
  - port: 9090
    targetPort: 9090
    protocol: TCP
    name: metrics
  selector:
    app: tesserabench

---
apiVersion: v1
kind: ConfigMap
metadata:
  name: tesserabench-config
data:
  production.yaml: |
    environment: production
    
    benchmarks:
      default_warmup_runs: 10
      default_timing_runs: 100
      timeout_seconds: 3600
      
    gpu:
      devices: "0,1,2,3,4,5,6,7"
      enable_profiling: true
      memory_fraction: 0.9
      
    database:
      connection_pool_size: 10
      statement_timeout: 30s
      
    monitoring:
      enable_metrics: true
      metrics_port: 9090
      log_level: INFO
      
    reporting:
      formats: ["json", "html"]
      include_plots: true
      retention_days: 90

---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: tesserabench-results
spec:
  accessModes:
    - ReadWriteOnce
  storageClassName: fast-ssd
  resources:
    requests:
      storage: 500Gi

---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:  
  name: tesserabench-cache
spec:
  accessModes:
    - ReadWriteOnce
  storageClassName: fast-ssd
  resources:
    requests:
      storage: 100Gi
```

### Helm Chart Configuration

```yaml
# charts/tesserabench/values.yaml
replicaCount: 2

image:
  repository: tesserabench
  pullPolicy: Always
  tag: "latest"

service:
  type: LoadBalancer
  port: 80
  targetPort: 8080

ingress:
  enabled: true
  className: "nginx"
  annotations:
    nginx.ingress.kubernetes.io/proxy-body-size: "100m"
    cert-manager.io/cluster-issuer: "letsencrypt-prod"
  hosts:
    - host: tesserabench.company.com
      paths:
        - path: /
          pathType: Prefix
  tls:
    - secretName: tesserabench-tls
      hosts:
        - tesserabench.company.com

resources:
  requests:
    nvidia.com/gpu: 8
    cpu: "16"
    memory: "64Gi"
  limits:
    nvidia.com/gpu: 8
    cpu: "32" 
    memory: "128Gi"

nodeSelector:
  gpu-type: "h100"
  gpu-count: "8"

tolerations:
  - key: "gpu-node"
    operator: "Equal"
    value: "true"
    effect: "NoSchedule"

persistence:
  results:
    enabled: true
    storageClass: "fast-ssd"
    size: 500Gi
  cache:
    enabled: true
    storageClass: "fast-ssd" 
    size: 100Gi

config:
  environment: production
  benchmarks:
    default_warmup_runs: 10
    default_timing_runs: 100
    timeout_seconds: 3600
  gpu:
    devices: "0,1,2,3,4,5,6,7"
    enable_profiling: true
    memory_fraction: 0.9

monitoring:
  enabled: true
  serviceMonitor:
    enabled: true
    namespace: monitoring
    labels:
      app: tesserabench

secrets:
  database:
    url: ""  # Set via CI/CD or external secrets
  monitoring:
    token: ""  # Set via CI/CD or external secrets

autoscaling:
  enabled: false  # GPU workloads typically don't autoscale
  minReplicas: 1
  maxReplicas: 3
  targetCPUUtilizationPercentage: 80
```

This completes Document 7 covering Production Deployment and CI/CD Integration. The document provides comprehensive coverage of:

1. **CI/CD Integration**: GitHub Actions and Jenkins pipeline configurations with automated performance validation
2. **Regression Detection**: Advanced statistical regression detection with configurable thresholds and reporting
3. **Cloud Deployment**: Docker containers, Kubernetes deployments, and Helm charts for scalable cloud deployment
4. **Production Infrastructure**: Complete production-ready configurations with monitoring, persistence, and security

The document demonstrates how TesseraBench can be deployed at scale in enterprise environments with automated performance monitoring and regression detection integrated into the software development lifecycle.