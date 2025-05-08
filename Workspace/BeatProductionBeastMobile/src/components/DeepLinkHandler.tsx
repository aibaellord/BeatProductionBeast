import { useEffect } from 'react';
import { Linking } from 'react-native';

export function useDeepLinks(onLink: (url: string) => void) {
  useEffect(() => {
    const handleUrl = ({ url }: { url: string }) => onLink(url);
    Linking.addEventListener('url', handleUrl);
    return () => Linking.removeEventListener('url', handleUrl);
  }, [onLink]);
}
