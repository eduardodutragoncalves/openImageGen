import { useEffect, useState } from "react";
import { Route, Routes } from "react-router-dom";
import { useAuth, useHealth, useModel } from "./hooks/useApi";
import { Rail } from "./components/Rail";
import { AuthGate } from "./components/AuthGate";
import { Studio } from "./routes/Studio";
import { Models } from "./routes/Models";
import { JobDetail } from "./routes/JobDetail";

type Theme = "dark" | "light";

/** Dark is the default because of the room this runs in, not the category:
 *  a workstation beside the rig, judging photographs for hours. The choice is
 *  still the operator's, and it sticks. */
function useTheme() {
  const [theme, setTheme] = useState<Theme>(
    () => (localStorage.getItem("oig-theme") as Theme) ?? "dark",
  );
  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    localStorage.setItem("oig-theme", theme);
  }, [theme]);
  return [theme, () => setTheme((current) => (current === "dark" ? "light" : "dark"))] as const;
}

export function App() {
  const [theme, toggleTheme] = useTheme();
  const auth = useAuth();
  const authenticated = auth.data?.authenticated ?? false;
  const health = useHealth();
  const model = useModel(authenticated);

  if (auth.isLoading) {
    return <div className="flex h-dvh items-center justify-center text-xs text-[var(--ink-muted)]" />;
  }

  if (!authenticated) {
    return (
      <div className="flex h-dvh flex-col">
        <AuthGate />
      </div>
    );
  }

  return (
    <div className="flex h-dvh flex-col overflow-hidden">
      <Rail health={health.data} precision={model.data?.precision} theme={theme} onToggleTheme={toggleTheme} />
      {health.isError ? <Unreachable /> : null}
      <Routes>
        <Route path="/" element={<Studio health={health.data} />} />
        <Route path="/models" element={<Models health={health.data} />} />
        <Route path="/j/:jobId" element={<JobDetail />} />
        <Route path="*" element={<Studio health={health.data} />} />
      </Routes>
    </div>
  );
}

function Unreachable() {
  return (
    <p
      role="alert"
      className="shrink-0 border-b border-[var(--alarm)] px-4 py-2 text-xs text-[var(--alarm-ink)]"
    >
      The API is not answering. The studio keeps showing the last state it had; nothing
      you submitted was lost.
    </p>
  );
}
