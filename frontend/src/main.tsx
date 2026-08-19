import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { BrowserRouter } from "react-router-dom";
import { App } from "./App";
import "./styles/theme.css";

const client = new QueryClient({
  defaultOptions: {
    queries: {
      // Every screen except the compose form is server state; refetching on
      // focus is what makes an alt-tabbed operator see the truth on return.
      refetchOnWindowFocus: true,
      staleTime: 1000,
    },
  },
});

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <QueryClientProvider client={client}>
      <BrowserRouter>
        <App />
      </BrowserRouter>
    </QueryClientProvider>
  </StrictMode>,
);
