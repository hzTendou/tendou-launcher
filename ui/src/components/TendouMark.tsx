import React from 'react';

/**
 * Tendou Launcher official mascot icon.
 */
export const TendouMark: React.FC<{ className?: string }> = ({ className = 'w-6 h-6' }) => (
  <img
    src="/icon.png"
    alt="Tendou"
    draggable={false}
    className={`object-contain select-none ${className}`}
  />
);
