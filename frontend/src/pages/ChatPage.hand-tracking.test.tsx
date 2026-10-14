// @vitest-environment happy-dom
// @vitest-environment-options { "url": "http://localhost:5173" }
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { createMemoryRouter, RouterProvider } from 'react-router';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { ChatPage } from './ChatPage';
import { useAppStore } from '../lib/store';

const vision = vi.hoisted(() => ({
  create: vi.fn(),
  detect: vi.fn(),
  close: vi.fn(),
}));
const send = vi.hoisted(() => vi.fn());

vi.mock('@mediapipe/tasks-vision', () => ({
  FilesetResolver: {
    isSimdSupported: vi.fn().mockResolvedValue(true),
    forVisionTasks: vi.fn().mockResolvedValue({}),
  },
  HandLandmarker: { createFromOptions: vision.create },
}));
vi.mock('../lib/connectors-api', () => ({ listConnectors: vi.fn().mockResolvedValue([]) }));
vi.mock('../lib/sse', () => ({ streamChat: send, streamResearch: send }));
vi.mock('../components/Chat/MessageBubble', () => ({ MessageBubble: () => null }));
vi.mock('../hooks/useSpeech', () => ({
  useSpeech: () => ({ state: 'idle', error: null, available: false }),
}));

function openPalm() {
  const points = Array.from({ length: 21 }, () => ({ x: 0.5, y: 0.7, z: 0 }));
  points[0].y = 0.8;
  points[3].x = 0.6;
  points[4].x = 0.8;
  for (const tip of [8, 12, 16, 20]) points[tip].y = 0.2;
  for (const pip of [6, 10, 14, 18]) points[pip].y = 0.5;
  return points;
}

function mockStream() {
  const stop = vi.fn();
  const stream = new MediaStream();
  Object.defineProperty(stream, 'getTracks', { value: () => [{ stop }] });
  return { stop, stream };
}

let root: Root;
let container: HTMLDivElement;
let router: ReturnType<typeof createMemoryRouter>;
let frames: Map<number, FrameRequestCallback>;
let camera: ReturnType<typeof mockStream>;
let permission: ReturnType<typeof mockStream>;
let getUserMedia: ReturnType<typeof vi.fn>;
let hidden: boolean;
let videoTime: number;

beforeEach(() => {
  (globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;
  localStorage.clear();
  vi.clearAllMocks();
  useAppStore.setState({
    selectedModel: 'test-model', modelLoading: false, systemPanelOpen: false,
    deepResearch: false, activeId: null, messages: [],
    streamState: { ...useAppStore.getState().streamState, isStreaming: false },
  });
  hidden = false;
  videoTime = 0;
  vi.spyOn(document, 'hidden', 'get').mockImplementation(() => hidden);
  vi.spyOn(HTMLMediaElement.prototype, 'play').mockResolvedValue();
  // happy-dom exposes media elements but does not define their readiness constants.
  Object.defineProperty(HTMLMediaElement, 'HAVE_CURRENT_DATA', { configurable: true, value: 2 });
  vi.spyOn(HTMLMediaElement.prototype, 'readyState', 'get').mockReturnValue(2);
  vi.spyOn(HTMLMediaElement.prototype, 'currentTime', 'get').mockImplementation(() => videoTime);
  frames = new Map();
  let id = 0;
  vi.stubGlobal('requestAnimationFrame', vi.fn((callback: FrameRequestCallback) => {
    frames.set(++id, callback);
    return id;
  }));
  vi.stubGlobal('cancelAnimationFrame', vi.fn((frame: number) => frames.delete(frame)));
  // Catch accidental backend/model downloads rather than making real requests.
  vi.stubGlobal('fetch', vi.fn(() => Promise.reject(new Error('Unexpected network request'))));
  permission = mockStream();
  camera = mockStream();
  getUserMedia = vi.fn().mockResolvedValueOnce(permission.stream).mockResolvedValue(camera.stream);
  Object.defineProperty(navigator, 'mediaDevices', {
    configurable: true,
    value: {
      getUserMedia,
      enumerateDevices: vi.fn().mockResolvedValue([
        { kind: 'videoinput', deviceId: 'phone', label: 'Phone camera' },
        { kind: 'videoinput', deviceId: 'webcam', label: 'Integrated Webcam' },
      ]),
    },
  });
  vision.create.mockResolvedValue({ detectForVideo: vision.detect, close: vision.close });
  vision.detect.mockReturnValue({ landmarks: [openPalm()] });
  container = document.createElement('div');
  document.body.append(container);
  root = createRoot(container);
});

afterEach(async () => {
  await act(async () => root?.unmount());
  router?.dispose();
  container?.remove();
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
});

async function mount(path = '/?handTracking=1') {
  router = createMemoryRouter([
    { path: '/', element: <ChatPage /> },
    { path: '/settings', element: <div>Settings</div> },
  ], { initialEntries: [path] });
  await act(async () => root.render(<RouterProvider router={router} />));
}

function button(text: string) {
  const result = [...container.querySelectorAll('button')].find((node) => node.textContent?.trim() === text);
  expect(result, `Button: ${text}`).toBeDefined();
  return result!;
}

async function click(text: string) {
  await act(async () => button(text).click());
}

async function start() {
  await click('Choose camera');
  expect(permission.stop).toHaveBeenCalledOnce();
  await click('Start camera');
  expect(container.querySelector('[role="alert"]')?.textContent).toBeUndefined();
}

async function frame() {
  const [id, callback] = frames.entries().next().value!;
  frames.delete(id);
  videoTime += 1;
  await act(async () => callback(performance.now()));
}

describe('chat hand tracking route and camera lifecycle', () => {
  it('opens without permission and focuses the real chat input only after opt-in, without sending', async () => {
    await mount('/');
    expect(container.querySelector('video')).toBeNull();
    await click('Hand tracking');
    expect(getUserMedia).not.toHaveBeenCalled();
    const focusOption = container.querySelector<HTMLInputElement>('input[type="checkbox"]')!;
    expect(focusOption.checked).toBe(false);
    await start();
    button('Stop camera').focus();
    await frame();
    expect(document.activeElement).toBe(button('Stop camera'));
    await act(async () => focusOption.click());
    vision.detect.mockReturnValueOnce({ landmarks: [] });
    await frame();
    await frame();
    expect(document.activeElement).toBe(container.querySelector('textarea'));
    expect(getUserMedia).toHaveBeenNthCalledWith(1, { video: true, audio: false });
    expect(getUserMedia).toHaveBeenNthCalledWith(2, { video: { deviceId: { exact: 'webcam' } }, audio: false });
    expect(send).not.toHaveBeenCalled();
    expect(useAppStore.getState().messages).toEqual([]);
    await click('Close hand tracking (stops camera)');
    expect(camera.stop).toHaveBeenCalledOnce();
    expect(vision.close).toHaveBeenCalledOnce();
    expect(frames.size).toBe(0);
    await click('Hand tracking');
    expect(getUserMedia).toHaveBeenCalledTimes(2);
  });

  it('stops on route changes and does not resume on returning to chat', async () => {
    await mount();
    await start();
    await act(async () => { await router.navigate('/settings'); });
    expect(camera.stop).toHaveBeenCalledOnce();
    expect(vision.close).toHaveBeenCalledOnce();
    expect(frames.size).toBe(0);
    await act(async () => { await router.navigate('/?handTracking=1'); });
    expect(getUserMedia).toHaveBeenCalledTimes(2);
    expect(container.querySelector('video')?.srcObject).toBeFalsy();
  });

  it.each(['hidden', 'pagehide', 'stop'])('stops capture on %s and requires another start', async (reason) => {
    await mount();
    await start();
    await act(async () => {
      if (reason === 'hidden') {
        hidden = true;
        document.dispatchEvent(new Event('visibilitychange'));
      } else if (reason === 'pagehide') window.dispatchEvent(new Event('pagehide'));
      else button('Stop camera').click();
    });
    expect(camera.stop).toHaveBeenCalledOnce();
    expect(vision.close).toHaveBeenCalledOnce();
    expect(frames.size).toBe(0);
    await act(async () => {
      hidden = false;
      document.dispatchEvent(new Event('visibilitychange'));
    });
    expect(getUserMedia).toHaveBeenCalledTimes(2);
    expect(button('Start camera')).toBeDefined();
  });

  it('closes a permission stream that arrives after leaving chat', async () => {
    let resolvePermission!: (stream: MediaStream) => void;
    getUserMedia.mockReset().mockReturnValue(new Promise((resolve) => { resolvePermission = resolve; }));
    await mount();
    await click('Choose camera');
    await act(async () => { await router.navigate('/settings'); });
    await act(async () => resolvePermission(permission.stream));
    expect(permission.stop).toHaveBeenCalledOnce();
    expect(vision.create).not.toHaveBeenCalled();
  });

  it('stops the camera immediately and closes a detector that finishes loading after cancellation', async () => {
    let resolveDetector!: (detector: unknown) => void;
    vision.create.mockReturnValueOnce(new Promise((resolve) => { resolveDetector = resolve; }));
    await mount();
    await start();
    await click('Close hand tracking (stops camera)');
    expect(camera.stop).toHaveBeenCalledOnce();
    await act(async () => resolveDetector({ close: vision.close }));
    expect(vision.close).toHaveBeenCalledOnce();
    expect(frames.size).toBe(0);
  });

  it('reports permission denial without loading the detector', async () => {
    getUserMedia.mockReset().mockRejectedValue(new DOMException('Denied', 'NotAllowedError'));
    await mount();
    await click('Choose camera');
    expect(container.querySelector('[role="alert"]')?.textContent).toContain('Camera permission was denied');
    expect(vision.create).not.toHaveBeenCalled();
  });

  it('releases the camera if inference or detector cleanup fails', async () => {
    vision.detect.mockImplementationOnce(() => { throw new Error('WASM failed'); });
    vision.close.mockImplementationOnce(() => { throw new Error('Already failed'); });
    await mount();
    await start();
    await frame();
    expect(camera.stop).toHaveBeenCalledOnce();
    expect(frames.size).toBe(0);
    expect(container.querySelector('[role="alert"]')?.textContent).toContain('Hand tracking stopped');
  });
});
