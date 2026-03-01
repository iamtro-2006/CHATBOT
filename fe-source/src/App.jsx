import { Routes, Route, Navigate } from "react-router-dom";
import Login from "./pages/Login.jsx";
import Signup from "./pages/Signup.jsx";
import Menu from "./pages/Menu.jsx";
import Conversation from "./pages/Conversation.jsx";

function RequireAuth({ children }) {
  const userId = localStorage.getItem("user_id");
  if (!userId) {
    return <Navigate to="/login" replace />;
  }
  return children;
}

function BlockAuthed({ children }) {
  const userId = localStorage.getItem("user_id");
  if (userId) {
    return <Navigate to="/menu" replace />;
  }
  return children;
}

export default function App() {
  return (
    <Routes>
      <Route path="/" element={<Navigate to="/login" replace />} />
      <Route
        path="/login"
        element={
          <BlockAuthed>
            <Login />
          </BlockAuthed>
        }
      />
      <Route
        path="/signup"
        element={
          <BlockAuthed>
            <Signup />
          </BlockAuthed>
        }
      />
      <Route
        path="/menu"
        element={
          <RequireAuth>
            <Menu />
          </RequireAuth>
        }
      />
      <Route
        path="/conversation"
        element={
          <RequireAuth>
            <Conversation />
          </RequireAuth>
        }
      />
    </Routes>
  );
}
