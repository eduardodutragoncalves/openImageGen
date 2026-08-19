import { useState } from "react";
import { useSignIn } from "../hooks/useApi";
import { ApiError, Unauthorized } from "../lib/api";
import { Diagonal } from "./primitives";
import { IconKey } from "./Icons";

/**
 * The key gate. This server is reachable from outside the LAN, so a key is
 * required — but it is a key, not an account: no signup, no password reset,
 * no identity. It is stored as an HttpOnly cookie so images load without ever
 * putting the key in a URL.
 */
export function AuthGate() {
  const [key, setKey] = useState("");
  const signIn = useSignIn();

  return (
    <main className="flex min-h-0 flex-1 items-center justify-center p-6">
      <div className="w-full max-w-[420px] border border-[var(--rule)] bg-[var(--ground)] p-6">
        <div className="mb-4 flex items-center gap-2">
          <IconKey className="text-[var(--accent-ink)]" />
          <h1 className="text-base font-semibold tracking-tight">openImageGen</h1>
        </div>
        <p className="mb-5 text-xs leading-relaxed text-[var(--ink-muted)]">
          This server runs generation on its own GPUs and answers from outside the
          local network, so it needs one of its API keys. Your archive is scoped to
          the key you use.
        </p>
        <form
          className="flex flex-col gap-2"
          onSubmit={(event) => {
            event.preventDefault();
            if (key.trim()) signIn.mutate(key.trim());
          }}
        >
          <label className="label" htmlFor="api-key">
            api key
          </label>
          <input
            id="api-key"
            type="password"
            autoComplete="current-password"
            value={key}
            onChange={(event) => setKey(event.target.value)}
            placeholder="from OIG_API_KEYS"
            className="field font-mono text-xs"
          />
          {signIn.isError ? (
            <p role="alert" className="text-[11px] text-[var(--alarm-ink)]">
              {signIn.error instanceof ApiError || signIn.error instanceof Unauthorized
                ? signIn.error.message
                : "That key was not accepted."}
            </p>
          ) : null}
          <button type="submit" className="btn btn-primary mt-2" disabled={!key.trim() || signIn.isPending}>
            <span>{signIn.isPending ? "Checking" : "Unlock"}</span>
            <Diagonal size={14} />
          </button>
        </form>
      </div>
    </main>
  );
}
