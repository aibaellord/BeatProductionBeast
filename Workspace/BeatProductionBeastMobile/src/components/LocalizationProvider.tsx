import React, { createContext, useContext, useState } from 'react';

const defaultLocale = 'en';
const translations: Record<string, Record<string, string>> = {
  en: {},
  es: { 'Welcome to BeatProductionBeast!': '¡Bienvenido a BeatProductionBeast!' },
  fr: { 'Welcome to BeatProductionBeast!': 'Bienvenue sur BeatProductionBeast!' },
  // Add more translations as needed
};

// TODO: Add more translations and support for RTL languages

const LocalizationContext = createContext({
  locale: defaultLocale,
  setLocale: (l: string) => {},
  t: (key: string) => key,
});

export function LocalizationProvider({ children }: { children: React.ReactNode }) {
  const [locale, setLocale] = useState(defaultLocale);
  const t = (key: string) => translations[locale]?.[key] || key;
  return (
    <LocalizationContext.Provider value={{ locale, setLocale, t }}>
      {children}
    </LocalizationContext.Provider>
  );
}

// TODO: Add unit tests for localization provider

export function useLocalization() {
  return useContext(LocalizationContext);
}
