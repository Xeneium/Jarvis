import { Hand } from 'lucide-react';
import { useSearchParams } from 'react-router';
import { HandTrackingPanel } from '../HandTrackingPanel';

/** Keep the opt-in camera and its focus target on the same route. */
export function ChatHandTracking() {
  const [searchParams, setSearchParams] = useSearchParams();
  const open = searchParams.get('handTracking') === '1';

  const togglePanel = () => {
    const next = new URLSearchParams(searchParams);
    if (open) next.delete('handTracking');
    else next.set('handTracking', '1');
    setSearchParams(next, { replace: true });
  };

  return (
    <div className="shrink-0 px-4 max-w-[var(--chat-max-width)] w-full mx-auto">
      <button
        type="button"
        aria-expanded={open}
        aria-controls="chat-hand-tracking"
        onClick={togglePanel}
        className="flex items-center gap-1.5 text-xs py-2 cursor-pointer"
        style={{ color: 'var(--color-text-secondary)' }}
      >
        <Hand size={14} />
        {open ? 'Close hand tracking (stops camera)' : 'Hand tracking'}
      </button>
      {open && (
        <div id="chat-hand-tracking" className="max-h-[45vh] overflow-y-auto pb-3">
          <HandTrackingPanel />
          <p className="text-xs mt-2" style={{ color: 'var(--color-text-tertiary)' }}>
            Camera access starts only when you choose a camera. Closing this panel,
            leaving chat, or hiding the app stops it; returning does not restart it.
          </p>
        </div>
      )}
    </div>
  );
}
