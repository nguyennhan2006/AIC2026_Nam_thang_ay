import { StrictMode } from "react";
import { createRoot } from "react-dom/client";

// Thứ tự import QUAN TRỌNG: tokens → base (reset) → primitives → layout →
// feature. File sau ghi đè file trước cho cùng selector, nên feature CSS
// không bao giờ bị primitives ghi đè ngược.
import "./styles/tokens.css";
import "./styles/base.css";
import "./styles/primitives.css";
import "./styles/shell.css";
import "./styles/studio.css";
import "./styles/weights.css";
import "./styles/results.css";
import "./styles/inspector.css";
import "./styles/misc.css";

import App from "./App.tsx";

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <App />
  </StrictMode>
);
