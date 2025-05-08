// AccessibilityHelper: Utility for a11y best practices
import { AccessibilityInfo } from 'react-native';

export function announceForAccessibility(message: string) {
  AccessibilityInfo.announceForAccessibility(message);
}

// Add more helpers as needed for screen reader, contrast, etc.
// TODO: Add helpers for color contrast, focus management, and screen reader hints
// TODO: Add unit tests for accessibility helpers
