
import { KernelEvent } from '../types';
import { RuntimeState } from './PureZenBKernel';
import { playCue } from './audio';
import { hapticPhase } from './haptics';
import { useSettingsStore } from '../stores/settingsStore';

export type Middleware = (
  event: KernelEvent,
  beforeState: RuntimeState,
  afterState: RuntimeState
) => void;

function phaseToCueType(phase: string): 'inhale' | 'exhale' | 'hold' {
  if (phase === 'holdIn' || phase === 'holdOut') return 'hold';
  return phase as 'inhale' | 'exhale';
}

/**
 * Middleware to handle audio cues on phase transitions
 */
export const audioMiddleware: Middleware = (event, before, after) => {
  if (event.type === 'PHASE_TRANSITION' && after.status === 'RUNNING') {
    const cueType = phaseToCueType(after.phase);
    // Get settings from store (singleton usage is safe here as this runs in event loop)
    const settings = useSettingsStore.getState().userSettings;
    
    playCue(
      cueType,
      settings.soundEnabled,
      settings.soundPack,
      after.phaseDuration,
      settings.language
    );
  }
};

/**
 * Middleware to handle haptic feedback
 */
export const hapticMiddleware: Middleware = (event, before, after) => {
  if (event.type === 'PHASE_TRANSITION' && after.status === 'RUNNING') {
    const settings = useSettingsStore.getState().userSettings;
    const cueType = phaseToCueType(after.phase);
    
    hapticPhase(settings.hapticEnabled, settings.hapticStrength, cueType);
  }
};
