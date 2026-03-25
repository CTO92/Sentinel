import { BrowserRouter, Routes, Route } from "react-router-dom";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { Layout } from "./components/Layout";
import FleetOverview from "./pages/FleetOverview";
import GpuDetail from "./pages/GpuDetail";
import AuditTrail from "./pages/AuditTrail";
import QuarantineQueue from "./pages/QuarantineQueue";
import TrustGraph from "./pages/TrustGraph";

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      retry: 2,
      refetchOnWindowFocus: false,
      staleTime: 5_000,
    },
  },
});

function SettingsPage() {
  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold tracking-tight">Settings</h1>
        <p className="text-sm text-muted-foreground mt-1">
          SENTINEL configuration
        </p>
      </div>
      <div className="rounded-lg border border-border bg-card p-6">
        <h2 className="text-sm font-semibold mb-4">API Configuration</h2>
        <div className="space-y-4 max-w-md">
          <div>
            <label
              htmlFor="api-url"
              className="block text-sm font-medium mb-1"
            >
              API Base URL
            </label>
            <input
              id="api-url"
              type="text"
              defaultValue="/api"
              className="h-9 w-full rounded-md border border-input bg-background px-3 text-sm focus:outline-none focus:ring-2 focus:ring-ring"
            />
          </div>
          <div>
            <label
              htmlFor="refresh-interval"
              className="block text-sm font-medium mb-1"
            >
              Refresh Interval (ms)
            </label>
            <input
              id="refresh-interval"
              type="number"
              defaultValue={10000}
              className="h-9 w-full rounded-md border border-input bg-background px-3 text-sm focus:outline-none focus:ring-2 focus:ring-ring"
            />
          </div>
        </div>
      </div>
      <div className="rounded-lg border border-border bg-card p-6">
        <h2 className="text-sm font-semibold mb-4">Display</h2>
        <div className="space-y-3">
          <label className="flex items-center gap-3 text-sm">
            <input
              type="checkbox"
              defaultChecked
              className="h-4 w-4 rounded border-input"
            />
            Dark mode
          </label>
          <label className="flex items-center gap-3 text-sm">
            <input
              type="checkbox"
              defaultChecked
              className="h-4 w-4 rounded border-input"
            />
            Show GPU heatmap on fleet overview
          </label>
          <label className="flex items-center gap-3 text-sm">
            <input
              type="checkbox"
              defaultChecked
              className="h-4 w-4 rounded border-input"
            />
            Enable real-time anomaly notifications
          </label>
        </div>
      </div>
    </div>
  );
}

export default function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        <Routes>
          <Route element={<Layout />}>
            <Route index element={<FleetOverview />} />
            <Route path="/gpu" element={<GpuDetail />} />
            <Route path="/gpu/:uuid" element={<GpuDetail />} />
            <Route path="/audit" element={<AuditTrail />} />
            <Route path="/quarantine" element={<QuarantineQueue />} />
            <Route path="/trust" element={<TrustGraph />} />
            <Route path="/settings" element={<SettingsPage />} />
          </Route>
        </Routes>
      </BrowserRouter>
    </QueryClientProvider>
  );
}
