import React, { useState, useEffect } from 'react';
import { BrowserRouter as Router, Route, Switch, useLocation } from 'react-router-dom';
import BeatLab from './BeatLab';
import Marketplace from './Marketplace';
import CollabUniverse from './CollabUniverse';
import UserProfile from './UserProfile';
import SettingsPage from './SettingsPage';
import HelpCenter from './HelpCenter';
import NotificationCenter from './NotificationCenter';
import SmartAssistant from './SmartAssistant';
import PluginManager from './PluginManager';
import QuickActionsBar from './QuickActionsBar';
import Dashboard from './Dashboard';
import Sidebar from './Sidebar';

// Global context for user preferences (theme, language, automation)
const AppContext = React.createContext();

export default function App() {
  const location = useLocation();
  // Global state for preferences
  const [theme, setTheme] = useState('system');
  const [language, setLanguage] = useState('en');
  const [automation, setAutomation] = useState({ oneClick: true, batch: true, advanced: false });

  // Accessibility: Focus main content on route change
  useEffect(() => {
    const main = document.querySelector('.main-content');
    if (main) main.focus();
  }, [location]);

  // Theme/Accessibility: Apply theme from settings
  useEffect(() => {
    document.body.className = theme;
  }, [theme]);

  // Responsive/mobile: Add viewport meta tag if not present
  useEffect(() => {
    if (!document.querySelector('meta[name="viewport"]')) {
      const meta = document.createElement('meta');
      meta.name = 'viewport';
      meta.content = 'width=device-width, initial-scale=1';
      document.head.appendChild(meta);
    }
  }, []);

  return (
    <AppContext.Provider value={{ theme, setTheme, language, setLanguage, automation, setAutomation }}>
      <Router>
        <div className={`app-root theme-${theme}`}>
          <Sidebar />
          <NotificationCenter />
          <SmartAssistant />
          <QuickActionsBar />
          <PluginManager />
          {/* Main content area for routed pages */}
          <div className="main-content" tabIndex={-1} aria-label="Main Content Area">
            <Switch>
              <Route path="/beat-lab" component={BeatLab} />
              <Route path="/marketplace" component={Marketplace} />
              <Route path="/collab" component={CollabUniverse} />
              <Route path="/profile" component={UserProfile} />
              <Route path="/settings" component={SettingsPage} />
              <Route path="/help" component={HelpCenter} />
              <Route path="/" component={Dashboard} />
            </Switch>
          </div>
        </div>
      </Router>
    </AppContext.Provider>
  );
}