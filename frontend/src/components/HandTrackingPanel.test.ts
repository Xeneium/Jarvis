import { describe, expect, it } from 'vitest';
import type { NormalizedLandmark } from '@mediapipe/tasks-vision';
import { classifyHandGesture, preferredCameraDevice } from './HandTrackingPanel';

function landmarks(
  extendedFingers: number[],
  thumbExtended = false,
): NormalizedLandmark[] {
  const points = Array.from({ length: 21 }, (_, index) => ({
    x: index === 4 && thumbExtended ? 0.8 : index === 3 && thumbExtended ? 0.6 : 0.5,
    y: 0.7,
    z: 0,
    visibility: 1,
  }));
  points[0].x = 0.5;
  points[0].y = 0.8;

  const fingerTips = [8, 12, 16, 20];
  const fingerPips = [6, 10, 14, 18];
  for (let index = 0; index < 4; index += 1) {
    points[fingerTips[index]].y = extendedFingers.includes(index) ? 0.2 : 0.8;
    points[fingerPips[index]].y = 0.5;
  }
  return points;
}

describe('hand gesture classification', () => {
  it('recognizes basic hand poses from local landmarks', () => {
    expect(classifyHandGesture(landmarks([0, 1, 2, 3], true))).toBe('Open palm');
    expect(classifyHandGesture(landmarks([]))).toBe('Fist');
    expect(classifyHandGesture(landmarks([0, 1]))).toBe('Peace sign');
    expect(classifyHandGesture(landmarks([0]))).toBe('Pointing');
  });

  describe('camera device selection', () => {
    it('prefers the built-in webcam over a phone camera when both are available', () => {
      const cameras = [
        { deviceId: 'phone', label: 'Phone Link Camera' },
        { deviceId: 'webcam', label: 'Integrated Webcam' },
      ];

      expect(preferredCameraDevice(cameras)).toEqual(cameras[1]);
    });

    it('uses the first available camera when no built-in camera is listed', () => {
      const cameras = [{ deviceId: 'phone', label: 'Phone Link Camera' }];

      expect(preferredCameraDevice(cameras)).toEqual(cameras[0]);
      expect(preferredCameraDevice([])).toBeUndefined();
    });
  });

  it('avoids classifying incomplete landmark sets as a specific gesture', () => {
    expect(classifyHandGesture(landmarks([0]).slice(0, 10))).toBe('Hand detected');
  });
});
