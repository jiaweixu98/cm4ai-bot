import {useEffect, useRef} from 'react';

export default function ConfirmPopover({ open, title, body, confirmLabel, cancelLabel = "Keep going", onConfirm, onCancel }) {
  const panel=useRef(null);
  useEffect(()=>{
    if(!open) return;
    const previous=document.activeElement;
    panel.current?.querySelector('button')?.focus();
    const keydown=event=>{
      if(event.key==='Escape') {event.preventDefault();event.stopPropagation();onCancel();}
      if(event.key==='Tab') {
        const controls=panel.current?.querySelectorAll('button:not(:disabled)');
        const first=controls?.[0],last=controls?.[controls.length-1];
        if(event.shiftKey&&document.activeElement===first) {event.preventDefault();last?.focus();}
        if(!event.shiftKey&&document.activeElement===last) {event.preventDefault();first?.focus();}
      }
    };
    const current=panel.current;
    current?.addEventListener('keydown',keydown);
    return ()=>{current?.removeEventListener('keydown',keydown);previous?.focus?.();};
  },[open,onCancel]);
  if (!open) return null;
  return (
    <div ref={panel} className="confirm-popover" role="alertdialog" aria-modal="true" aria-labelledby="confirm-title">
      <div className="confirm-popover-card">
        <h3 id="confirm-title">{title}</h3>
        <p>{body}</p>
        <div className="confirm-popover-actions">
          <button type="button" className="ghost-btn" onClick={onCancel}>
            {cancelLabel}
          </button>
          <button type="button" className="ghost-btn danger" onClick={onConfirm}>
            {confirmLabel}
          </button>
        </div>
      </div>
    </div>
  );
}
