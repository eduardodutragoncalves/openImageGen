import { Route, Routes } from "react-router-dom";
import { useAuth, useHealth, useModel } from "./hooks/useApi";
import { useVisual } from "./hooks/useVisual";
import { Rail } from "./components/Rail";
import { AuthGate } from "./components/AuthGate";
import { Studio } from "./routes/Studio";
import { Models } from "./routes/Models";
import { JobDetail } from "./routes/JobDetail";

export function App() {
  const [visual, setVisual] = useVisual();
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
      <Rail
        health={health.data}
        precision={model.data?.precision}
        visual={visual}
        onVisual={setVisual}
      />
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
