import { useCallback, useEffect, useRef, useState } from 'react';
import {
  FilesetResolver,
  HandLandmarker,
  type NormalizedLandmark,
} from '@mediapipe/tasks-vision';
import simdLoaderUrl from '@mediapipe/tasks-vision/vision_wasm_internal.js?url';
import simdWasmUrl from '@mediapipe/tasks-vision/vision_wasm_internal.wasm?url';
import noSimdLoaderUrl from '@mediapipe/tasks-vision/vision_wasm_nosimd_internal.js?url';
import noSimdWasmUrl from '@mediapipe/tasks-vision/vision_wasm_nosimd_internal.wasm?url';
import { Camera, Hand, StopCircle } from 'lucide-react';

const HAND_CONNECTIONS = [
  [0, 1], [1, 2], [2, 3], [3, 4],
  [0, 5], [5, 6], [6, 7], [7, 8],
  [5, 9], [9, 10], [10, 11], [11, 12],
  [9, 13], [13, 14], [14, 15], [15, 16],
  [13, 17], [17, 18], [18, 19], [19, 20], [0, 17],
];

export type HandGesture =
  | 'Open palm'
  | 'Fist'
  | 'Peace sign'
  | 'Pointing'
  | 'Hand detected';

export interface CameraDevice {
  deviceId: string;
  label: string;
}

export function preferredCameraDevice(devices: CameraDevice[]): CameraDevice | undefined {
  return devices.find((device) => /integrated|built[\s-]?in|internal|webcam/i.test(device.label))
    ?? devices[0];
}

export function classifyHandGesture(points: NormalizedLandmark[]): HandGesture {
  if (points.length < 21) return 'Hand detected';

  const extended = [
    points[8].y < points[6].y - 0.02,
    points[12].y < points[10].y - 0.02,
    points[16].y < points[14].y - 0.02,
    points[20].y < points[18].y - 0.02,
  ];
  const distanceFromWrist = (index: number) =>
    Math.hypot(points[index].x - points[0].x, points[index].y - points[0].y);
  const thumbExtended = distanceFromWrist(4) > distanceFromWrist(3) * 1.1;
  const extendedCount = extended.filter(Boolean).length;

  if (extendedCount === 4 && thumbExtended) return 'Open palm';
  if (extendedCount === 0 && !thumbExtended) return 'Fist';
  if (extended[0] && extended[1] && !extended[2] && !extended[3]) return 'Peace sign';
  if (extended[0] && !extended[1] && !extended[2] && !extended[3]) return 'Pointing';
  return 'Hand detected';
}

function localVisionFileset() {
  return FilesetResolver.isSimdSupported().then(async (simdSupported) => {
    const fileset = await FilesetResolver.forVisionTasks('/');
    fileset.wasmLoaderPath = simdSupported ? simdLoaderUrl : noSimdLoaderUrl;
    fileset.wasmBinaryPath = simdSupported ? simdWasmUrl : noSimdWasmUrl;
    return fileset;
  });
}

type TrackerState = 'idle' | 'choosing' | 'starting' | 'active' | 'error';

export function HandTrackingPanel() {
  const [state, setState] = useState<TrackerState>('idle');
  const [error, setError] = useState('');
  const [landmarks, setLandmarks] = useState<NormalizedLandmark[][]>([]);
  const [gesture, setGesture] = useState('No hand detected');
  const [cameras, setCameras] = useState<CameraDevice[]>([]);
  const [selectedCameraId, setSelectedCameraId] = useState('');
  const [focusOnOpenPalm, setFocusOnOpenPalm] = useState(
    () => localStorage.getItem('openjarvis-hand-focus') === 'true',
  );
  const focusOnOpenPalmRef = useRef(focusOnOpenPalm);
  const videoRef = useRef<HTMLVideoElement>(null);
  const detectorRef = useRef<HandLandmarker | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const frameRef = useRef<number | null>(null);
  const generationRef = useRef(0);
  const previousGestureRef = useRef<string>('');

  const releaseTracking = useCallback(() => {
    generationRef.current += 1;
    if (frameRef.current !== null) cancelAnimationFrame(frameRef.current);
    frameRef.current = null;
    const detector = detectorRef.current;
    detectorRef.current = null;
    // Camera cleanup must still run if the WASM runtime failed during shutdown.
    try { detector?.close(); } catch { /* Already failed/closed. */ }
    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;
    if (videoRef.current) videoRef.current.srcObject = null;
  }, []);

  const stopTracking = useCallback(() => {
    releaseTracking();
    setLandmarks([]);
    setGesture('No hand detected');
    previousGestureRef.current = '';
    setState('idle');
    setError('');
  }, [releaseTracking]);

  const chooseCamera = useCallback(async () => {
    if (!navigator.mediaDevices?.getUserMedia) {
      setError('Camera access is not available in this browser or context.');
      setState('error');
      return;
    }

    const generation = ++generationRef.current;
    setError('');
    setState('choosing');
    try {
      const permissionStream = await navigator.mediaDevices.getUserMedia({
        video: true,
        audio: false,
      });
      if (generationRef.current !== generation) {
        permissionStream.getTracks().forEach((track) => track.stop());
        return;
      }

      // Permission reveals device labels. Stop this temporary stream before
      // enumerating or opening the user-selected camera.
      permissionStream.getTracks().forEach((track) => track.stop());
      const devices = (await navigator.mediaDevices.enumerateDevices())
        .filter((device) => device.kind === 'videoinput')
        .map(({ deviceId, label }, index) => ({
          deviceId,
          label: label || `Camera ${index + 1}`,
        }));
      if (generationRef.current !== generation) return;
      if (devices.length === 0) throw new Error('No cameras were found.');

      setCameras(devices);
      setSelectedCameraId(preferredCameraDevice(devices)?.deviceId ?? devices[0].deviceId);
      setState('idle');
    } catch (cause) {
      if (generationRef.current !== generation) return;
      setError(
        cause instanceof DOMException && cause.name === 'NotAllowedError'
          ? 'Camera permission was denied. Allow camera access in your browser settings, then try again.'
          : cause instanceof Error
            ? cause.message
            : 'Could not list available cameras.',
      );
      setState('error');
    }
  }, []);

  const startTracking = useCallback(async () => {
    if (!navigator.mediaDevices?.getUserMedia) {
      setError('Camera access is not available in this browser or context.');
      setState('error');
      return;
    }

    const generation = ++generationRef.current;
    setError('');
    setState('starting');

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { deviceId: { exact: selectedCameraId } },
        audio: false,
      });
      if (generationRef.current !== generation) {
        stream.getTracks().forEach((track) => track.stop());
        return;
      }
      streamRef.current = stream;
      const video = videoRef.current;
      if (!video) throw new Error('Camera preview is unavailable.');
      video.srcObject = stream;
      await video.play();
      if (generationRef.current !== generation) return;

      const fileset = await localVisionFileset();
      if (generationRef.current !== generation) return;
      const detector = await HandLandmarker.createFromOptions(fileset, {
        baseOptions: {
          modelAssetPath: new URL('/models/hand_landmarker.task', window.location.origin).href,
        },
        runningMode: 'VIDEO',
        numHands: 2,
        minHandDetectionConfidence: 0.55,
        minTrackingConfidence: 0.55,
      });

      if (generationRef.current !== generation) {
        detector.close();
        stream.getTracks().forEach((track) => track.stop());
        return;
      }

      detectorRef.current = detector;
      setState('active');
      let lastVideoTime = -1;
      let lastUiUpdate = 0;

      const detectFrame = () => {
        if (generationRef.current !== generation) return;
        const currentDetector = detectorRef.current;
        if (
          currentDetector &&
          video.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA &&
          video.currentTime !== lastVideoTime
        ) {
          lastVideoTime = video.currentTime;
          let result;
          try {
            result = currentDetector.detectForVideo(video, performance.now());
          } catch {
            stopTracking();
            setError('Hand tracking stopped after a camera or detector error. Start the camera to try again.');
            setState('error');
            return;
          }
          const detectedHands = result.landmarks;
          const currentGesture = detectedHands.length
            ? classifyHandGesture(detectedHands[0])
            : 'No hand detected';

          if (performance.now() - lastUiUpdate > 80) {
            lastUiUpdate = performance.now();
            setLandmarks(detectedHands);
            setGesture(currentGesture);
          }

          if (
            focusOnOpenPalmRef.current &&
            currentGesture === 'Open palm' &&
            previousGestureRef.current !== 'Open palm'
          ) {
            window.dispatchEvent(new Event('openjarvis:focus-chat-input'));
          }
          previousGestureRef.current = currentGesture;
        }
        frameRef.current = requestAnimationFrame(detectFrame);
      };
      frameRef.current = requestAnimationFrame(detectFrame);
    } catch (cause) {
      if (generationRef.current !== generation) return;
      releaseTracking();
      setError(
        cause instanceof DOMException && cause.name === 'NotAllowedError'
          ? 'Camera permission was denied. Allow camera access in system settings, then try again.'
          : cause instanceof Error
            ? cause.message
            : 'Could not start local hand tracking.',
      );
      setState('error');
    }
  }, [selectedCameraId, releaseTracking, stopTracking]);

  useEffect(() => {
    const onVisibilityChange = () => {
      if (document.hidden) stopTracking();
    };
    document.addEventListener('visibilitychange', onVisibilityChange);
    window.addEventListener('pagehide', stopTracking);
    return () => {
      document.removeEventListener('visibilitychange', onVisibilityChange);
      window.removeEventListener('pagehide', stopTracking);
      releaseTracking();
    };
  }, [releaseTracking, stopTracking]);

  const updateFocusOption = (enabled: boolean) => {
    setFocusOnOpenPalm(enabled);
    focusOnOpenPalmRef.current = enabled;
    localStorage.setItem('openjarvis-hand-focus', String(enabled));
  };

  return (
    <div className="pt-3">
      <div className="flex items-start justify-between gap-4">
        <div className="min-w-0">
          <div className="text-sm" style={{ color: 'var(--color-text)' }}>Camera hand tracking</div>
          <p className="text-xs mt-1" style={{ color: 'var(--color-text-tertiary)' }}>
            Optional, on-device landmark and basic gesture detection. Camera frames stay in this app and are not recorded.
          </p>
        </div>
        <button
          type="button"
          onClick={() => {
            if (state === 'active' || state === 'starting' || state === 'choosing') {
              stopTracking();
            } else if (cameras.length === 0) {
              void chooseCamera();
            } else {
              void startTracking();
            }
          }}
          aria-pressed={state === 'active' || state === 'starting'}
          className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium cursor-pointer shrink-0"
          style={{
            background: state === 'active' ? 'var(--color-error)' : 'var(--color-bg-secondary)',
            border: '1px solid var(--color-border)',
            color: state === 'active' ? 'white' : 'var(--color-text)',
          }}
        >
          {state === 'choosing'
            ? <><StopCircle size={13} /> Cancel camera access</>
            : state === 'starting'
            ? <><StopCircle size={13} /> Stop camera</>
            : state === 'active'
              ? <><StopCircle size={13} /> Stop camera</>
              : cameras.length === 0
                ? <><Camera size={13} /> Choose camera</>
                : <><Camera size={13} /> Start camera</>}
        </button>
      </div>

      {cameras.length > 0 && state !== 'active' && (
        <div className="flex items-center gap-3 mt-3">
          <label htmlFor="openjarvis-camera-device" className="text-xs" style={{ color: 'var(--color-text-secondary)' }}>
            Camera device
          </label>
          <select
            id="openjarvis-camera-device"
            aria-label="Camera device"
            value={selectedCameraId}
            onChange={(event) => setSelectedCameraId(event.target.value)}
            disabled={state === 'starting'}
            className="text-xs px-2 py-1 rounded-lg outline-none max-w-full"
            style={{ background: 'var(--color-bg-secondary)', color: 'var(--color-text)', border: '1px solid var(--color-border)' }}
          >
            {cameras.map((camera) => (
              <option key={camera.deviceId} value={camera.deviceId}>{camera.label}</option>
            ))}
          </select>
          <button
            type="button"
            onClick={() => void chooseCamera()}
            disabled={state === 'choosing' || state === 'starting'}
            className="text-xs underline cursor-pointer"
            style={{ color: 'var(--color-accent)' }}
          >
            Refresh
          </button>
        </div>
      )}

      {/* Keep the ref available even when camera permission resolves before React
          commits the starting state. An empty video never starts capture. */}
      <div className={state === 'active' || state === 'starting' ? 'flex flex-col sm:flex-row items-start gap-4 mt-4' : 'hidden'}>
        <div className="relative w-full sm:w-64 aspect-video overflow-hidden rounded-lg bg-black">
          <video
            ref={videoRef}
            muted
            playsInline
            aria-label="Live camera preview for hand tracking"
            className="absolute inset-0 w-full h-full object-cover"
            style={{ transform: 'scaleX(-1)' }}
          />
          <svg
            aria-hidden="true"
            viewBox="0 0 1 1"
            preserveAspectRatio="none"
            className="absolute inset-0 w-full h-full"
            style={{ transform: 'scaleX(-1)' }}
          >
            {landmarks.map((hand, handIndex) => (
              <g key={handIndex}>
                {HAND_CONNECTIONS.map(([start, end]) => (
                  <line
                    key={`${start}-${end}`}
                    x1={hand[start].x}
                    y1={hand[start].y}
                    x2={hand[end].x}
                    y2={hand[end].y}
                    stroke="#34d399"
                    strokeWidth="0.009"
                  />
                ))}
                {hand.map((point, pointIndex) => (
                  <circle key={pointIndex} cx={point.x} cy={point.y} r="0.014" fill="#fbbf24" />
                ))}
              </g>
            ))}
          </svg>
        </div>
        <div aria-live="polite" className="flex items-center gap-2 text-xs" style={{ color: 'var(--color-text-secondary)' }}>
          <Hand size={15} style={{ color: landmarks.length ? 'var(--color-success)' : 'var(--color-text-tertiary)' }} />
          {state === 'starting'
            ? 'Starting local hand tracker…'
            : landmarks.length
            ? `${landmarks.length} hand${landmarks.length === 1 ? '' : 's'} — ${gesture}`
            : 'No hand detected'}
        </div>
      </div>

      {state === 'error' && (
        <p role="alert" className="text-xs mt-3" style={{ color: 'var(--color-error)' }}>{error}</p>
      )}
      {cameras.length > 0 && state !== 'active' && (
        <p className="text-xs mt-2" style={{ color: 'var(--color-text-tertiary)' }}>
          Choose the webcam by name before starting. The camera used to reveal device names was stopped immediately.
        </p>
      )}

      <label className="flex items-center gap-2 mt-3 text-xs cursor-pointer" style={{ color: 'var(--color-text-secondary)' }}>
        <input
          type="checkbox"
          checked={focusOnOpenPalm}
          onChange={(event) => updateFocusOption(event.target.checked)}
          className="accent-[var(--color-accent)]"
        />
        Focus chat input when an open palm is detected
      </label>
      <p className="text-xs mt-1 pl-5" style={{ color: 'var(--color-text-tertiary)' }}>
        Optional and off by default. Gestures never send messages or trigger tools.
      </p>
    </div>
  );
}
