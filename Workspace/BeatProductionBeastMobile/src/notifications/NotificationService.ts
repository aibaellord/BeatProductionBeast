// NotificationService: Handles push notifications and in-app alerts
import messaging from '@react-native-firebase/messaging';

export async function requestUserPermission() {
  const authStatus = await messaging().requestPermission();
  return (
    authStatus === messaging.AuthorizationStatus.AUTHORIZED ||
    authStatus === messaging.AuthorizationStatus.PROVISIONAL
  );
}

export function onNotificationReceived(callback: (msg: any) => void) {
  return messaging().onMessage(async remoteMessage => {
    callback(remoteMessage);
  });
}

// TODO: Integrate with backend for sales, new beats, and challenge notifications
// TODO: Track analytics events for notification delivery and open
// TODO: Add automation for scheduled and triggered notifications
// TODO: Add unit and integration tests for NotificationService
