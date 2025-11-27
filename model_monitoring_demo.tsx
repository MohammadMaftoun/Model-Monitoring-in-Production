// ============================================================================
// MODEL MONITORING SYSTEM - MODULAR TENSORFLOW.JS IMPLEMENTATION
// ============================================================================

// ----------------------------------------------------------------------------
// 1. DATA DRIFT DETECTION MODULE
// ----------------------------------------------------------------------------

class DriftDetector {
  constructor() {
    this.referenceData = null;
    this.thresholds = {
      psi: 0.2,
      kl: 0.1,
      ks: 0.05
    };
  }

  /**
   * Set reference (training) data distribution
   */
  setReferenceData(data) {
    this.referenceData = data;
  }

  /**
   * Calculate Population Stability Index (PSI)
   * PSI = Σ (actual% - expected%) * ln(actual% / expected%)
   */
  calculatePSI(expected, actual, bins = 10) {
    const expHist = this._histogram(expected, bins);
    const actHist = this._histogram(actual, bins);
    
    let psi = 0;
    for (let i = 0; i < bins; i++) {
      const expPct = expHist[i] / expected.length;
      const actPct = actHist[i] / actual.length;
      
      if (expPct > 0 && actPct > 0) {
        psi += (actPct - expPct) * Math.log(actPct / expPct);
      }
    }
    
    return {
      psi: psi,
      isDrifted: psi > this.thresholds.psi,
      severity: this._classifyDrift(psi, 'psi')
    };
  }

  /**
   * Calculate Kullback-Leibler Divergence
   * KL(P||Q) = Σ P(x) * log(P(x) / Q(x))
   */
  calculateKLDivergence(p, q, bins = 10) {
    const pHist = this._histogram(p, bins);
    const qHist = this._histogram(q, bins);
    
    let kl = 0;
    for (let i = 0; i < bins; i++) {
      const pVal = pHist[i] / p.length + 1e-10;
      const qVal = qHist[i] / q.length + 1e-10;
      kl += pVal * Math.log(pVal / qVal);
    }
    
    return {
      kl: kl,
      isDrifted: kl > this.thresholds.kl,
      severity: this._classifyDrift(kl, 'kl')
    };
  }

  /**
   * Kolmogorov-Smirnov Test
   */
  kolmogorovSmirnovTest(sample1, sample2) {
    const sorted1 = [...sample1].sort((a, b) => a - b);
    const sorted2 = [...sample2].sort((a, b) => a - b);
    
    let maxDiff = 0;
    let i = 0, j = 0;
    
    while (i < sorted1.length && j < sorted2.length) {
      const cdf1 = i / sorted1.length;
      const cdf2 = j / sorted2.length;
      maxDiff = Math.max(maxDiff, Math.abs(cdf1 - cdf2));
      
      if (sorted1[i] < sorted2[j]) i++;
      else j++;
    }
    
    return {
      statistic: maxDiff,
      isDrifted: maxDiff > this.thresholds.ks,
      severity: this._classifyDrift(maxDiff, 'ks')
    };
  }

  _histogram(data, bins) {
    const min = Math.min(...data);
    const max = Math.max(...data);
    const binWidth = (max - min) / bins;
    const hist = new Array(bins).fill(0);
    
    data.forEach(val => {
      const binIdx = Math.min(Math.floor((val - min) / binWidth), bins - 1);
      hist[binIdx]++;
    });
    
    return hist;
  }

  _classifyDrift(value, metric) {
    const threshold = this.thresholds[metric];
    if (value < threshold) return 'none';
    if (value < threshold * 1.5) return 'moderate';
    return 'severe';
  }
}

// ----------------------------------------------------------------------------
// 2. MODEL PERFORMANCE MONITOR
// ----------------------------------------------------------------------------

class PerformanceMonitor {
  constructor() {
    this.metrics = [];
    this.window = 100; // Rolling window size
  }

  /**
   * Calculate classification metrics
   */
  calculateMetrics(yTrue, yPred, threshold = 0.5) {
    const predictions = yPred.map(p => p >= threshold ? 1 : 0);
    
    let tp = 0, fp = 0, tn = 0, fn = 0;
    
    for (let i = 0; i < yTrue.length; i++) {
      if (yTrue[i] === 1 && predictions[i] === 1) tp++;
      else if (yTrue[i] === 0 && predictions[i] === 1) fp++;
      else if (yTrue[i] === 0 && predictions[i] === 0) tn++;
      else if (yTrue[i] === 1 && predictions[i] === 0) fn++;
    }
    
    const accuracy = (tp + tn) / (tp + tn + fp + fn);
    const precision = tp / (tp + fp) || 0;
    const recall = tp / (tp + fn) || 0;
    const f1 = 2 * (precision * recall) / (precision + recall) || 0;
    
    const metrics = {
      accuracy,
      precision,
      recall,
      f1,
      confusion: { tp, fp, tn, fn },
      timestamp: Date.now()
    };
    
    this.metrics.push(metrics);
    if (this.metrics.length > this.window) {
      this.metrics.shift();
    }
    
    return metrics;
  }

  /**
   * Calculate prediction confidence statistics
   */
  confidenceStats(predictions) {
    const confidences = predictions.map(p => Math.max(p, 1 - p));
    
    return {
      mean: this._mean(confidences),
      std: this._std(confidences),
      min: Math.min(...confidences),
      max: Math.max(...confidences),
      distribution: this._confidenceDistribution(predictions)
    };
  }

  /**
   * Detect performance degradation
   */
  detectDegradation(baseline, current, threshold = 0.05) {
    const degradation = {
      accuracy: baseline.accuracy - current.accuracy,
      precision: baseline.precision - current.precision,
      recall: baseline.recall - current.recall,
      f1: baseline.f1 - current.f1
    };
    
    const alerts = [];
    for (const [metric, value] of Object.entries(degradation)) {
      if (value > threshold) {
        alerts.push({
          metric,
          degradation: value,
          severity: value > threshold * 2 ? 'critical' : 'warning'
        });
      }
    }
    
    return { degradation, alerts };
  }

  _mean(arr) {
    return arr.reduce((a, b) => a + b, 0) / arr.length;
  }

  _std(arr) {
    const mean = this._mean(arr);
    const variance = arr.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / arr.length;
    return Math.sqrt(variance);
  }

  _confidenceDistribution(predictions) {
    const bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0];
    const dist = new Array(bins.length - 1).fill(0);
    
    predictions.forEach(p => {
      const conf = Math.max(p, 1 - p);
      for (let i = 0; i < bins.length - 1; i++) {
        if (conf >= bins[i] && conf < bins[i + 1]) {
          dist[i]++;
          break;
        }
      }
    });
    
    return dist;
  }
}

// ----------------------------------------------------------------------------
// 3. DATA QUALITY CHECKER
// ----------------------------------------------------------------------------

class DataQualityChecker {
  constructor() {
    this.rules = [];
  }

  /**
   * Add validation rule
   */
  addRule(name, validator) {
    this.rules.push({ name, validator });
  }

  /**
   * Check for missing values
   */
  checkMissing(data) {
    let missing = 0;
    let total = 0;
    
    data.forEach(row => {
      Object.values(row).forEach(val => {
        total++;
        if (val === null || val === undefined || val === '') {
          missing++;
        }
      });
    });
    
    const missingRate = missing / total;
    
    return {
      count: missing,
      rate: missingRate,
      passed: missingRate < 0.05,
      severity: missingRate > 0.1 ? 'critical' : 'warning'
    };
  }

  /**
   * Detect outliers using IQR method
   */
  detectOutliers(values) {
    const sorted = [...values].sort((a, b) => a - b);
    const q1 = sorted[Math.floor(sorted.length * 0.25)];
    const q3 = sorted[Math.floor(sorted.length * 0.75)];
    const iqr = q3 - q1;
    
    const lowerBound = q1 - 1.5 * iqr;
    const upperBound = q3 + 1.5 * iqr;
    
    const outliers = values.filter(v => v < lowerBound || v > upperBound);
    
    return {
      count: outliers.length,
      rate: outliers.length / values.length,
      bounds: { lower: lowerBound, upper: upperBound },
      outliers: outliers
    };
  }

  /**
   * Validate feature ranges
   */
  validateRanges(data, expectedRanges) {
    const violations = [];
    
    for (const [feature, range] of Object.entries(expectedRanges)) {
      const values = data.map(d => d[feature]).filter(v => v !== null);
      const min = Math.min(...values);
      const max = Math.max(...values);
      
      if (min < range.min || max > range.max) {
        violations.push({
          feature,
          expected: range,
          actual: { min, max },
          severity: 'warning'
        });
      }
    }
    
    return {
      passed: violations.length === 0,
      violations
    };
  }

  /**
   * Check for schema consistency
   */
  validateSchema(data, schema) {
    const errors = [];
    
    data.forEach((row, idx) => {
      for (const [field, type] of Object.entries(schema)) {
        if (!(field in row)) {
          errors.push({ row: idx, field, error: 'missing_field' });
        } else if (typeof row[field] !== type && row[field] !== null) {
          errors.push({ 
            row: idx, 
            field, 
            error: 'type_mismatch',
            expected: type,
            actual: typeof row[field]
          });
        }
      }
    });
    
    return {
      passed: errors.length === 0,
      errors,
      errorRate: errors.length / (data.length * Object.keys(schema).length)
    };
  }
}

// ----------------------------------------------------------------------------
// 4. TENSORFLOW.JS MODEL WRAPPER
// ----------------------------------------------------------------------------

class ModelMonitor {
  constructor(model) {
    this.model = model;
    this.driftDetector = new DriftDetector();
    this.performanceMonitor = new PerformanceMonitor();
    this.qualityChecker = new DataQualityChecker();
    this.predictionLog = [];
    this.maxLogSize = 1000;
  }

  /**
   * Make predictions and log for monitoring
   */
  async predict(inputData) {
    const tensor = tf.tensor2d(inputData);
    const predictions = await this.model.predict(tensor);
    const predArray = await predictions.array();
    
    // Log predictions
    this.logPredictions(inputData, predArray);
    
    tensor.dispose();
    predictions.dispose();
    
    return predArray;
  }

  /**
   * Log predictions for monitoring
   */
  logPredictions(inputs, predictions) {
    const timestamp = Date.now();
    
    inputs.forEach((input, idx) => {
      this.predictionLog.push({
        input,
        prediction: predictions[idx],
        timestamp
      });
    });
    
    // Keep log size manageable
    if (this.predictionLog.length > this.maxLogSize) {
      this.predictionLog = this.predictionLog.slice(-this.maxLogSize);
    }
  }

  /**
   * Generate monitoring report
   */
  generateReport(groundTruth = null) {
    const report = {
      timestamp: Date.now(),
      predictions: {
        count: this.predictionLog.length,
        timeRange: {
          start: this.predictionLog[0]?.timestamp,
          end: this.predictionLog[this.predictionLog.length - 1]?.timestamp
        }
      }
    };

    // Confidence statistics
    const predictions = this.predictionLog.map(p => p.prediction[0]);
    report.confidence = this.performanceMonitor.confidenceStats(predictions);

    // If ground truth available, calculate performance
    if (groundTruth && groundTruth.length === predictions.length) {
      report.performance = this.performanceMonitor.calculateMetrics(
        groundTruth,
        predictions
      );
    }

    // Feature distribution analysis
    const features = this.predictionLog.map(p => p.input);
    report.dataQuality = this._analyzeFeatureDistribution(features);

    return report;
  }

  /**
   * Check for drift against reference data
   */
  checkDrift(referenceData, currentData, featureIndex = 0) {
    const refFeature = referenceData.map(d => d[featureIndex]);
    const currFeature = currentData.map(d => d[featureIndex]);
    
    return {
      psi: this.driftDetector.calculatePSI(refFeature, currFeature),
      kl: this.driftDetector.calculateKLDivergence(refFeature, currFeature),
      ks: this.driftDetector.kolmogorovSmirnovTest(refFeature, currFeature)
    };
  }

  /**
   * Export monitoring data
   */
  exportData() {
    return {
      predictions: this.predictionLog,
      metrics: this.performanceMonitor.metrics,
      timestamp: Date.now()
    };
  }

  _analyzeFeatureDistribution(features) {
    if (features.length === 0) return null;
    
    const numFeatures = features[0].length;
    const analysis = {};
    
    for (let i = 0; i < numFeatures; i++) {
      const values = features.map(f => f[i]);
      analysis[`feature_${i}`] = {
        mean: values.reduce((a, b) => a + b, 0) / values.length,
        min: Math.min(...values),
        max: Math.max(...values),
        outliers: this.qualityChecker.detectOutliers(values)
      };
    }
    
    return analysis;
  }
}

// ----------------------------------------------------------------------------
// 5. ALERT SYSTEM
// ----------------------------------------------------------------------------

class AlertSystem {
  constructor() {
    this.alerts = [];
    this.handlers = [];
  }

  /**
   * Register alert handler
   */
  onAlert(handler) {
    this.handlers.push(handler);
  }

  /**
   * Create alert
   */
  createAlert(type, message, severity = 'info', metadata = {}) {
    const alert = {
      id: Date.now() + Math.random(),
      type,
      message,
      severity,
      metadata,
      timestamp: Date.now()
    };
    
    this.alerts.push(alert);
    this.handlers.forEach(h => h(alert));
    
    return alert;
  }

  /**
   * Check thresholds and create alerts
   */
  checkThresholds(metrics, thresholds) {
    const alerts = [];
    
    if (metrics.accuracy < thresholds.accuracy) {
      alerts.push(this.createAlert(
        'performance',
        `Accuracy dropped to ${(metrics.accuracy * 100).toFixed(2)}%`,
        'critical',
        { metric: 'accuracy', value: metrics.accuracy }
      ));
    }
    
    if (metrics.drift && metrics.drift.psi.isDrifted) {
      alerts.push(this.createAlert(
        'drift',
        `Data drift detected: PSI = ${metrics.drift.psi.psi.toFixed(3)}`,
        metrics.drift.psi.severity,
        { psi: metrics.drift.psi.psi }
      ));
    }
    
    return alerts;
  }

  /**
   * Clear old alerts
   */
  clearOldAlerts(maxAge = 86400000) { // 24 hours
    const cutoff = Date.now() - maxAge;
    this.alerts = this.alerts.filter(a => a.timestamp > cutoff);
  }
}

// ----------------------------------------------------------------------------
// 6. USAGE EXAMPLE
// ----------------------------------------------------------------------------

async function exampleUsage() {
  // Create a simple model for demonstration
  const model = tf.sequential({
    layers: [
      tf.layers.dense({ inputShape: [4], units: 8, activation: 'relu' }),
      tf.layers.dense({ units: 1, activation: 'sigmoid' })
    ]
  });

  model.compile({
    optimizer: 'adam',
    loss: 'binaryCrossentropy',
    metrics: ['accuracy']
  });

  // Initialize monitoring
  const monitor = new ModelMonitor(model);
  const alertSystem = new AlertSystem();

  // Set up alert handler
  alertSystem.onAlert(alert => {
    console.log(`[${alert.severity.toUpperCase()}] ${alert.message}`);
  });

  // Generate synthetic training data
  const trainingData = Array.from({ length: 100 }, () => 
    Array.from({ length: 4 }, () => Math.random())
  );

  // Generate production data (with drift)
  const productionData = Array.from({ length: 100 }, () => 
    Array.from({ length: 4 }, () => Math.random() * 1.5) // Intentional drift
  );

  // Make predictions
  const predictions = await monitor.predict(productionData);

  // Check for drift
  const driftReport = monitor.checkDrift(trainingData, productionData, 0);
  console.log('Drift Detection:', driftReport);

  // Generate monitoring report
  const report = monitor.generateReport();
  console.log('Monitoring Report:', JSON.stringify(report, null, 2));

  // Check data quality
  const qualityCheck = monitor.qualityChecker.detectOutliers(
    productionData.map(d => d[0])
  );
  console.log('Quality Check:', qualityCheck);

  // Check alerts
  alertSystem.checkThresholds(
    { accuracy: 0.85, drift: { psi: driftReport.psi } },
    { accuracy: 0.90 }
  );

  // Export data for analysis
  const exportedData = monitor.exportData();
  console.log(`Exported ${exportedData.predictions.length} predictions`);
}

// Export modules
if (typeof module !== 'undefined' && module.exports) {
  module.exports = {
    DriftDetector,
    PerformanceMonitor,
    DataQualityChecker,
    ModelMonitor,
    AlertSystem
  };
}

// ----------------------------------------------------------------------------
// EXAMPLE: Run monitoring simulation
// ----------------------------------------------------------------------------
console.log('Model Monitoring System - TensorFlow.js Implementation');
console.log('========================================================\n');

// Uncomment to run example
// exampleUsage().then(() => console.log('Monitoring example completed'));