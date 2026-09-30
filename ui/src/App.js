import React from 'react';
import { Routes, Route, Navigate } from 'react-router-dom';
import { Container } from 'rsuite';
import { useAuth } from './context/AuthContext';
import './App.css';
import './styles/theme.css';

// Pages
import Login from './pages/Login';
import Register from './pages/Register';
import Dashboard from './pages/Dashboard';
import ChatPage from './pages/ChatPage';
import AgentChatPage from './pages/AgentChatPage';
import KnowledgeGraphPage from './pages/KnowledgeGraphPage';
import DocumentsPage from './pages/DocumentsPage';
import EvidenceTracePage from './pages/EvidenceTracePage';
import InnovationPage from './pages/InnovationPage';
import InnovationHubPage from './pages/InnovationHubPage';

// Protected Route Component
const ProtectedRoute = ({ children }) => {
  const { isAuthenticated } = useAuth();

  if (!isAuthenticated) {
    return <Navigate to="/login" />;
  }

  return children;
};

function App() {
  return (
    <Container className="app-container">
      <Routes>
        <Route path="/login" element={<Login />} />
        <Route path="/register" element={<Register />} />
        <Route path="/" element={
          <ProtectedRoute>
            <Dashboard />
          </ProtectedRoute>
        } />
        <Route path="/chat" element={
          <ProtectedRoute>
            <ChatPage />
          </ProtectedRoute>
        } />
        <Route path="/chat/:threadId" element={
          <ProtectedRoute>
            <ChatPage />
          </ProtectedRoute>
        } />
        <Route path="/agent-chat" element={
          <ProtectedRoute>
            <AgentChatPage />
          </ProtectedRoute>
        } />
        <Route path="/agent-chat/:threadId" element={
          <ProtectedRoute>
            <AgentChatPage />
          </ProtectedRoute>
        } />
        <Route path="/knowledge-graph" element={
          <ProtectedRoute>
            <KnowledgeGraphPage />
          </ProtectedRoute>
        } />
        <Route path="/evidence-trace" element={
          <ProtectedRoute>
            <EvidenceTracePage />
          </ProtectedRoute>
        } />
        <Route path="/innovations" element={
          <ProtectedRoute>
            <InnovationPage />
          </ProtectedRoute>
        } />
        <Route path="/innovation-hub" element={
          <ProtectedRoute>
            <InnovationHubPage />
          </ProtectedRoute>
        } />
        <Route path="/documents" element={
          <ProtectedRoute>
            <DocumentsPage />
          </ProtectedRoute>
        } />
      </Routes>
    </Container>
  );
}

export default App;
