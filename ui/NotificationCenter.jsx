import React, { useState, useEffect } from 'react';

export default function NotificationCenter() {
  const [notifications, setNotifications] = useState([]);
  const [visible, setVisible] = useState(true);

  // Example: Simulate receiving notifications
  useEffect(() => {
    // TODO: Replace with real-time backend events (WebSocket, SSE, or polling)
    const timer = setTimeout(() => {
      setNotifications(n => [
        ...n,
        { id: Date.now(), type: 'success', message: 'Automation complete! All outputs are ready.' }
      ]);
    }, 3000);
    return () => clearTimeout(timer);
  }, []);

  const dismiss = id => setNotifications(n => n.filter(notif => notif.id !== id));

  if (!visible) return null;
  return (
    <div className="notification-center" aria-live="polite">
      <button className="close-center" onClick={() => setVisible(false)} aria-label="Hide notifications">×</button>
      {notifications.map(notif => (
        <div key={notif.id} className={`notification ${notif.type}`}> 
          <span>{notif.message}</span>
          <button onClick={() => dismiss(notif.id)} aria-label="Dismiss notification">×</button>
        </div>
      ))}
      {notifications.length === 0 && <div className="notification empty">No notifications</div>}
    </div>
  );
}