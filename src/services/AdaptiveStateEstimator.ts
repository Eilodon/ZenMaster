
import { Observation, BreathPattern, BeliefState, BREATHING_PATTERNS } from '../types';

/**
 * ADAPTIVE STATE ESTIMATOR
 * ============================
 * 
 * HONESTY FIRST: This is NOT true Active Inference (Friston 2010).
 * This is a Kalman-inspired adaptive filter for psychophysiological state estimation.
 * 
 * WHAT WE DO INSTEAD:
 * - Bayesian state estimation with discrete observations
 * - Kalman-style prediction + correction steps
 * - Empirically validated state transitions based on HRV research
 */

interface TargetState {
  arousal: number;
  attention: number;
  rhythm_alignment: number;
}

const PROTOCOL_TARGETS: Record<string, TargetState> = {
    // Parasympathetic dominance (sleep, anxiety reduction)
    'parasympathetic': {
        arousal: 0.2, 
        attention: 0.5, 
        rhythm_alignment: 0.8
    },
    // Balanced autonomic state (HRV coherence)
    'balanced': {
        arousal: 0.4,
        attention: 0.7,
        rhythm_alignment: 0.9
    },
    // Sympathetic activation (energizing)
    'sympathetic': {
        arousal: 0.7,
        attention: 0.8,
        rhythm_alignment: 0.6
    },
    'default': {
        arousal: 0.5,
        attention: 0.6,
        rhythm_alignment: 0.7
    }
};

const PATTERN_TO_TARGET: Record<string, keyof typeof PROTOCOL_TARGETS> = {
    '4-7-8': 'parasympathetic',
    'deep-relax': 'parasympathetic',
    '7-11': 'parasympathetic',
    'coherence': 'balanced',
    'calm': 'balanced',
    'box': 'balanced',
    'triangle': 'balanced',
    'tactical': 'balanced',
    'awake': 'sympathetic',
    'wim-hof': 'sympathetic',
    'buteyko': 'parasympathetic',
};

export class AdaptiveStateEstimator {
    private belief: BeliefState;
    private target: TargetState;
    
    // Kalman-style process noise (how much we trust dynamics model)
    private readonly PROCESS_NOISE = 0.01;
    // Measurement noise (how much we trust observations)
    private readonly MEASUREMENT_NOISE_HR = 0.15;
    private readonly MEASUREMENT_NOISE_CONTEXT = 0.05;

    // Time constants for state evolution (seconds)
    private readonly TAU_AROUSAL = 15.0; 
    private readonly TAU_ATTENTION = 5.0; 
    private readonly TAU_RHYTHM = 10.0; 

    constructor() {
        this.belief = {
            arousal: 0.5,
            attention: 0.5,
            rhythm_alignment: 0.0,
            arousal_variance: 0.2,
            attention_variance: 0.2,
            rhythm_variance: 0.3,
            prediction_error: 0.0,
            confidence: 0.0
        };
        this.target = PROTOCOL_TARGETS.default;
    }

    public setProtocol(pattern: BreathPattern | null): void {
        if (!pattern) {
            this.target = PROTOCOL_TARGETS.default;
            return;
        }
        const targetKey = PATTERN_TO_TARGET[pattern.id] || 'default';
        this.target = PROTOCOL_TARGETS[targetKey];
    }

    public update(obs: Observation, dt: number): BeliefState {
        // Step 1: PREDICTION (where we expect the state to be)
        const predicted = this.predict(dt);

        // Step 2: CORRECTION (incorporate new observation)
        const corrected = this.correct(predicted, obs, dt);

        // Step 3: DIAGNOSTICS
        corrected.prediction_error = this.computePredictionError(corrected);
        corrected.confidence = this.computeConfidence(corrected, obs);

        this.belief = corrected;
        return { ...this.belief };
    }

    private predict(dt: number): BeliefState {
        const { arousal, attention, rhythm_alignment } = this.belief;
        const { arousal_variance, attention_variance, rhythm_variance } = this.belief;

        // Compute decay factors
        const alpha_arousal = 1 - Math.exp(-dt / this.TAU_AROUSAL);
        const alpha_attention = 1 - Math.exp(-dt / this.TAU_ATTENTION);
        const alpha_rhythm = 1 - Math.exp(-dt / this.TAU_RHYTHM);

        // Predict state (settle toward target)
        const predicted_arousal = arousal + alpha_arousal * (this.target.arousal - arousal);
        const predicted_attention = attention + alpha_attention * (this.target.attention - attention);
        const predicted_rhythm = rhythm_alignment + alpha_rhythm * (this.target.rhythm_alignment - rhythm_alignment);

        // Predict uncertainty (increases due to process noise)
        const predicted_arousal_var = arousal_variance + this.PROCESS_NOISE * dt;
        const predicted_attention_var = attention_variance + this.PROCESS_NOISE * dt;
        const predicted_rhythm_var = rhythm_variance + this.PROCESS_NOISE * dt;

        return {
            arousal: this.clamp(predicted_arousal),
            attention: this.clamp(predicted_attention),
            rhythm_alignment: this.clamp(predicted_rhythm),
            arousal_variance: predicted_arousal_var,
            attention_variance: predicted_attention_var,
            rhythm_variance: predicted_rhythm_var,
            prediction_error: 0,
            confidence: 0
        };
    }

    private correct(predicted: BeliefState, obs: Observation, dt: number): BeliefState {
        let corrected = { ...predicted };

        // ---- AROUSAL CORRECTION (from Heart Rate) ----
        if (obs.heart_rate !== undefined && obs.hr_confidence !== undefined && obs.hr_confidence > 0.5) {
            // Normalize HR (Resting 50-70 -> ~0.2, Active 90-120 -> ~0.7)
            const normalized_hr = this.clamp((obs.heart_rate - 50) / 70);
            
            const K_arousal = predicted.arousal_variance / (predicted.arousal_variance + this.MEASUREMENT_NOISE_HR);
            const innovation = normalized_hr - predicted.arousal;
            
            corrected.arousal = predicted.arousal + K_arousal * innovation;
            corrected.arousal_variance = (1 - K_arousal) * predicted.arousal_variance;
        }

        // ---- ATTENTION CORRECTION (from Interaction) ----
        const isDistracted = obs.user_interaction === 'pause' || obs.visibilty_state === 'hidden';
        
        if (isDistracted) {
            const K_attention = predicted.attention_variance / (predicted.attention_variance + this.MEASUREMENT_NOISE_CONTEXT);
            const target_attention = 0.1; 
            const innovation = target_attention - predicted.attention;
            
            corrected.attention = predicted.attention + K_attention * innovation;
            corrected.attention_variance = (1 - K_attention) * predicted.attention_variance;
            
            // Rhythm breaks
            corrected.rhythm_alignment = Math.max(0, corrected.rhythm_alignment - 0.5 * dt);
        } else {
            // Gradual attention recovery
            corrected.attention = Math.min(1, corrected.attention + 0.15 * dt);
            corrected.attention_variance = Math.max(0.05, corrected.attention_variance - 0.02 * dt);
            
            // Rhythm builds
            corrected.rhythm_alignment = Math.min(1, corrected.rhythm_alignment + 0.1 * dt);
            corrected.rhythm_variance = Math.max(0.05, corrected.rhythm_variance - 0.01 * dt);
        }

        return {
            ...corrected,
            arousal: this.clamp(corrected.arousal),
            attention: this.clamp(corrected.attention),
            rhythm_alignment: this.clamp(corrected.rhythm_alignment)
        };
    }

    private computePredictionError(state: BeliefState): number {
        // PE = sqrt(sum of squared errors from target) - "Free Energy" proxy
        const error_arousal = Math.pow(state.arousal - this.target.arousal, 2);
        const error_attention = Math.pow(state.attention - this.target.attention, 2);
        const error_rhythm = Math.pow(state.rhythm_alignment - this.target.rhythm_alignment, 2);
        const mse = 0.4 * error_arousal + 0.3 * error_attention + 0.3 * error_rhythm;
        return Math.sqrt(mse);
    }

    private computeConfidence(state: BeliefState, obs: Observation): number {
        const certainty = 1 - Math.min(1, (state.arousal_variance + state.attention_variance + state.rhythm_variance) / 3);
        const sensor_quality = obs.hr_confidence ?? 0.0;
        const attention_stability = state.attention;
        // Geometric mean
        const confidence = Math.pow(certainty * (sensor_quality || 1.0) * attention_stability, 1/3);
        return this.clamp(confidence);
    }

    private clamp(value: number, min = 0, max = 1): number {
        return Math.max(min, Math.min(max, value));
    }
}
