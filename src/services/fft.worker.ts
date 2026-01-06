
/**
*    FFT WORKER (Web Worker for Heart Rate Detection)
* ====================================================
*
* WHY WEB WORKER?
* - FFT is CPU-intensive (even optimized O(N log N))
* - Running on main thread blocks UI at 60fps
* - Worker runs on separate thread → smooth 60fps rendering
*
* ALGORITHM: Cooley-Tukey FFT
* - Complexity: O(N log N) vs naive DFT O(N²)
*/

export interface FFTRequest {
  type: 'compute_fft';
  signal: number[]; // Time-domain signal
  sampleRate: number; // Hz (e.g., 60 for 60fps)
  minFreq: number; // Min heart rate in Hz (e.g., 0.75 = 45 BPM)
  maxFreq: number; // Max heart rate in Hz (e.g., 3.0 = 180 BPM)
}

export interface FFTResponse {
  type: 'fft_result';
  heartRate: number; // Detected BPM
  confidence: number; // [0,1]
  peakFrequency: number; // Hz
  powerSpectrum: number[]; // For visualization (optional)
}

export interface ErrorResponse {
  type: 'error';
  message: string;
}

class FFT {
  private n: number;
  private levels: number;
  // Precomputed twiddle factors (complex exponentials)
  private cosTable: Float64Array;
  private sinTable: Float64Array;

  constructor(size: number) {
    // Validate size is power of 2
    if (!this.isPowerOfTwo(size)) {
      throw new Error(`FFT size must be power of 2, got ${size}`);
    }
    this.n = size;
    this.levels = Math.log2(size);
    // Precompute twiddle factors for efficiency
    this.cosTable = new Float64Array(size / 2);
    this.sinTable = new Float64Array(size / 2);
    for (let i = 0; i < size / 2; i++) {
      const angle = -2 * Math.PI * i / size;
      this.cosTable[i] = Math.cos(angle);
      this.sinTable[i] = Math.sin(angle);
    }
  }

  /**
   * Compute FFT (in-place, decimation-in-time)
   *
   * @param real - Real part of input (will be modified)
   * @param imag - Imaginary part (will be modified, initialize to zeros)
   */
  public transform(real: Float64Array, imag: Float64Array): void {
    if (real.length !== this.n || imag.length !== this.n) {
      throw new Error(`Expected arrays of length ${this.n}`);
    }
    // Bit-reversal permutation (for in-place algorithm)
    this.bitReversePermutation(real, imag);
    // Cooley-Tukey butterfly operations
    for (let size = 2; size <= this.n; size *= 2) {
      const halfSize = size / 2;
      const tableStep = this.n / size;
      for (let i = 0; i < this.n; i += size) {
        for (let j = i, k = 0; j < i + halfSize; j++, k += tableStep) {
          const tpre = real[j + halfSize] * this.cosTable[k] - imag[j + halfSize] * this.sinTable[k];
          const tpim = real[j + halfSize] * this.sinTable[k] + imag[j + halfSize] * this.cosTable[k];
          real[j + halfSize] = real[j] - tpre;
          imag[j + halfSize] = imag[j] - tpim;
          real[j] += tpre;
          imag[j] += tpim;
        }
      }
    }
  }

  /**
   * Compute magnitude spectrum (|X[k]|)
   */
  public getMagnitude(real: Float64Array, imag: Float64Array): Float64Array {
    const magnitude = new Float64Array(this.n / 2); // Only positive frequencies
    for (let i = 0; i < this.n / 2; i++) {
      magnitude[i] = Math.sqrt(real[i] * real[i] + imag[i] * imag[i]);
    }
    return magnitude;
  }

  private bitReversePermutation(real: Float64Array, imag: Float64Array): void {
    let j = 0;
    for (let i = 0; i < this.n - 1; i++) {
      if (i < j) {
        // Swap
        [real[i], real[j]] = [real[j], real[i]];
        [imag[i], imag[j]] = [imag[j], imag[i]];
      }
      let k = this.n / 2;
      while (k <= j) {
        j -= k;
        k /= 2;
      }
      j += k;
    }
  }

  private isPowerOfTwo(n: number): boolean {
    return n > 0 && (n & (n - 1)) === 0;
  }
}

// --- SIGNAL PREPROCESSING ---

function detrend(signal: number[]): Float64Array {
  const mean = signal.reduce((a, b) => a + b, 0) / signal.length;
  return Float64Array.from(signal, x => x - mean);
}

function applyHammingWindow(signal: Float64Array): Float64Array {
  const n = signal.length;
  const windowed = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    const window = 0.54 - 0.46 * Math.cos(2 * Math.PI * i / (n - 1));
    windowed[i] = signal[i] * window;
  }
  return windowed;
}

function zeroPad(signal: Float64Array): Float64Array {
  const n = signal.length;
  const nextPowerOf2 = Math.pow(2, Math.ceil(Math.log2(n)));
  if (n === nextPowerOf2) return signal;
  const padded = new Float64Array(nextPowerOf2);
  padded.set(signal);
  return padded;
}

// --- HEART RATE DETECTION ---

function parabolicInterpolation(spectrum: Float64Array, peakBin: number, freqResolution: number): number {
    if (peakBin <= 0 || peakBin >= spectrum.length - 1) {
        return peakBin * freqResolution;
    }
    const alpha = spectrum[peakBin - 1];
    const beta = spectrum[peakBin];
    const gamma = spectrum[peakBin + 1];
    const offset = 0.5 * (alpha - gamma) / (alpha - 2 * beta + gamma);
    return (peakBin + offset) * freqResolution;
}

function computeConfidence(spectrum: Float64Array, peakBin: number, minBin: number, maxBin: number): number {
    const peakPower = spectrum[peakBin];
    const excludeRadius = 5; 
    let noiseSum = 0;
    let noiseCount = 0;
    for (let i = minBin; i <= maxBin; i++) {
        if (Math.abs(i - peakBin) > excludeRadius) {
            noiseSum += spectrum[i];
            noiseCount++;
        }
    }
    const noisePower = noiseCount > 0 ? noiseSum / noiseCount : 1e-6;
    const snr = peakPower / noisePower;
    // Map SNR to confidence [0,1]. SNR > 10 is excellent, SNR < 2 is poor
    return Math.min(1, Math.max(0, (snr - 2) / 8));
}

function detectHeartRate(request: FFTRequest): FFTResponse {
  const { signal, sampleRate, minFreq, maxFreq } = request;
  
  let processed = detrend(signal);
  processed = applyHammingWindow(processed);
  processed = zeroPad(processed);
  const n = processed.length;
  
  const fft = new FFT(n);
  const real = processed;
  const imag = new Float64Array(n);
  fft.transform(real, imag);
  
  const magnitude = fft.getMagnitude(real, imag);
  
  const freqResolution = sampleRate / n;
  const minBin = Math.floor(minFreq / freqResolution);
  const maxBin = Math.ceil(maxFreq / freqResolution);
  
  let peakBin = minBin;
  let peakPower = magnitude[minBin];
  
  for (let i = minBin + 1; i <= maxBin && i < magnitude.length; i++) {
    if (magnitude[i] > peakPower) {
      peakPower = magnitude[i];
      peakBin = i;
    }
  }
  
  const peakFreq = parabolicInterpolation(magnitude, peakBin, freqResolution);
  const heartRate = peakFreq * 60;
  const confidence = computeConfidence(magnitude, peakBin, minBin, maxBin);
  
  return {
    type: 'fft_result',
    heartRate: Math.round(heartRate * 10) / 10,
    confidence,
    peakFrequency: peakFreq,
    powerSpectrum: Array.from(magnitude.slice(0, magnitude.length / 2))
  };
}

// --- WORKER MESSAGE HANDLER ---

self.onmessage = (event: MessageEvent<FFTRequest>) => {
  try {
    const request = event.data;
    if (request.type !== 'compute_fft') {
      throw new Error(`Unknown request type: ${request.type}`);
    }
    if (!Array.isArray(request.signal) || request.signal.length < 32) {
      throw new Error('Signal must be array with at least 32 samples');
    }
    if (request.sampleRate <= 0) throw new Error('Sample rate must be positive');
    
    const result = detectHeartRate(request);
    self.postMessage(result);
  } catch (error) {
    const errorResponse: ErrorResponse = {
      type: 'error',
      message: error instanceof Error ? error.message : String(error)
    };
    self.postMessage(errorResponse);
  }
};
