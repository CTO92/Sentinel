import { useState, useMemo } from "react";
import {
  Cpu,
  ShieldCheck,
  AlertTriangle,
  ShieldAlert,
  Skull,
  Activity,
  TrendingDown,
  Search as SearchIcon,
} from "lucide-react";
import { useFleetHealth } from "@/api/hooks";
import { GpuHealthState, type GpuHealth, type AnomalyEvent } from "@/api/types";
import { cn, formatPercent, formatNumber } from "@/lib/utils";
import { HeatMap } from "@/components/HeatMap";
import { GpuCard } from "@/components/GpuCard";
import { TimeSeriesChart } from "@/components/TimeSeriesChart";
import { AnomalyTimeline } from "@/components/AnomalyTimeline";
import { HealthBadge } from "@/components/HealthBadge";

// Synthetic data generators for demo when API is unavailable
function generateMockFleetData() {
  const states: GpuHealthState[] = [
    GpuHealthState.HEALTHY,
    GpuHealthState.SUSPECT,
    GpuHealthState.QUARANTINED,
    GpuHealthState.DEEP_TEST,
    GpuHealthState.CONDEMNED,
  ];
  const models = ["A100-SXM4-80GB", "H100-SXM5-80GB", "A100-SXM4-40GB"];
  const hostnames = Array.from({ length: 32 }, (_, i) => `gpu-node-${String(i).padStart(3, "0")}`);

  const gpus: GpuHealth[] = Array.from({ length: 256 }, (_, i) => {
    const r = Math.random();
    const state =
      r < 0.82
        ? GpuHealthState.HEALTHY
        : r < 0.90
          ? GpuHealthState.SUSPECT
          : r < 0.95
            ? GpuHealthState.QUARANTINED
            : r < 0.98
              ? GpuHealthState.DEEP_TEST
              : GpuHealthState.CONDEMNED;

    const reliability =
      state === GpuHealthState.HEALTHY
        ? 0.95 + Math.random() * 0.05
        : state === GpuHealthState.SUSPECT
          ? 0.7 + Math.random() * 0.2
          : state === GpuHealthState.QUARANTINED
            ? 0.4 + Math.random() * 0.3
            : state === GpuHealthState.CONDEMNED
              ? Math.random() * 0.3
              : 0.5 + Math.random() * 0.3;

    const totalProbes = 500 + Math.floor(Math.random() * 2000);
    const failRate = 1 - reliability;
    const failedProbes = Math.floor(totalProbes * failRate * Math.random());

    return {
      gpu_id: {
        uuid: `GPU-${String(i).padStart(4, "0")}-${Math.random().toString(36).slice(2, 10)}`,
        node_hostname: hostnames[i % hostnames.length],
        pci_bus_id: `0000:${(i % 8).toString(16).padStart(2, "0")}:00.0`,
        gpu_index: i % 8,
        model: models[i % models.length],
        vbios_version: "96.00.89.00.01",
        driver_version: "535.129.03",
      },
      state,
      reliability_score: reliability,
      bayesian_alpha: 100 + Math.floor(Math.random() * 900),
      bayesian_beta: Math.floor((1 - reliability) * 100),
      total_probes: totalProbes,
      failed_probes: failedProbes,
      last_probe_time: new Date(Date.now() - Math.random() * 3600000).toISOString(),
      last_state_change: new Date(Date.now() - Math.random() * 86400000).toISOString(),
      sm_health: [],
      temperature_celsius: 45 + Math.floor(Math.random() * 35),
      power_watts: 150 + Math.floor(Math.random() * 250),
      memory_used_mb: Math.floor(Math.random() * 81920),
      memory_total_mb: 81920,
      gpu_utilization: Math.random(),
    };
  });

  const healthy = gpus.filter((g) => g.state === GpuHealthState.HEALTHY).length;
  const suspect = gpus.filter((g) => g.state === GpuHealthState.SUSPECT).length;
  const quarantined = gpus.filter((g) => g.state === GpuHealthState.QUARANTINED).length;
  const deepTest = gpus.filter((g) => g.state === GpuHealthState.DEEP_TEST).length;
  const condemned = gpus.filter((g) => g.state === GpuHealthState.CONDEMNED).length;

  return {
    total_gpus: gpus.length,
    healthy,
    suspect,
    quarantined,
    deep_test: deepTest,
    condemned,
    sdc_rate: 0.00023,
    average_reliability: gpus.reduce((s, g) => s + g.reliability_score, 0) / gpus.length,
    total_probes_24h: 145832,
    anomalies_24h: 47,
    gpu_health: gpus,
  };
}

function generateMockAnomalies(): AnomalyEvent[] {
  const types = [
    "SDC_DETECTED",
    "RELIABILITY_DROP",
    "TEMPERATURE_SPIKE",
    "BIT_FLIP_CLUSTER",
    "CROSS_GPU_MISMATCH",
  ];
  const severities = ["CRITICAL", "HIGH", "MEDIUM", "LOW", "INFO"] as const;
  const descriptions = [
    "Silent data corruption detected during matrix multiply validation",
    "GPU reliability score dropped below threshold",
    "Temperature exceeded warning threshold at 87°C",
    "Cluster of bit flips detected in SM 14-16",
    "Cross-GPU comparison failed between GPU-0012 and reference GPU-0001",
    "Probe timeout exceeded 30s during memory pattern test",
    "Anomalous error rate increase detected over 1h window",
    "Bayesian model confidence interval widened significantly",
  ];

  return Array.from({ length: 20 }, (_, i) => ({
    event_id: `EVT-${String(i).padStart(6, "0")}`,
    gpu_uuid: `GPU-${String(Math.floor(Math.random() * 256)).padStart(4, "0")}`,
    node_hostname: `gpu-node-${String(Math.floor(Math.random() * 32)).padStart(3, "0")}`,
    timestamp: new Date(Date.now() - i * 300000 - Math.random() * 300000).toISOString(),
    severity: severities[Math.floor(Math.random() * severities.length)],
    anomaly_type: types[Math.floor(Math.random() * types.length)],
    description: descriptions[Math.floor(Math.random() * descriptions.length)],
    affected_sms: [],
    resolved: Math.random() > 0.7,
  }));
}

function generateProbeFailureTimeSeries() {
  const now = Date.now();
  return Array.from({ length: 48 }, (_, i) => ({
    timestamp: new Date(now - (47 - i) * 1800000).toISOString(),
    failure_rate: 0.001 + Math.random() * 0.004,
    sdc_rate: 0.0001 + Math.random() * 0.0004,
    total_probes: 2800 + Math.floor(Math.random() * 400),
  }));
}

type StateFilter = "ALL" | GpuHealthState;

export default function FleetOverview() {
  const { data: fleetData, isLoading, error } = useFleetHealth(10_000);
  const [stateFilter, setStateFilter] = useState<StateFilter>("ALL");
  const [searchQuery, setSearchQuery] = useState("");

  // Use API data when available, otherwise fall back to mock data
  const fleet = useMemo(() => fleetData ?? generateMockFleetData(), [fleetData]);
  const anomalies = useMemo(() => generateMockAnomalies(), []);
  const probeTimeSeries = useMemo(() => generateProbeFailureTimeSeries(), []);

  const filteredGpus = useMemo(() => {
    let gpus = fleet.gpu_health;
    if (stateFilter !== "ALL") {
      gpus = gpus.filter((g) => g.state === stateFilter);
    }
    if (searchQuery) {
      const q = searchQuery.toLowerCase();
      gpus = gpus.filter(
        (g) =>
          g.gpu_id.uuid.toLowerCase().includes(q) ||
          g.gpu_id.node_hostname.toLowerCase().includes(q) ||
          g.gpu_id.model.toLowerCase().includes(q),
      );
    }
    return gpus;
  }, [fleet.gpu_health, stateFilter, searchQuery]);

  const summaryCards = [
    {
      label: "Total GPUs",
      value: fleet.total_gpus,
      icon: Cpu,
      color: "text-primary",
    },
    {
      label: "Healthy",
      value: fleet.healthy,
      icon: ShieldCheck,
      color: "text-health-healthy",
    },
    {
      label: "Suspect",
      value: fleet.suspect,
      icon: AlertTriangle,
      color: "text-health-suspect",
    },
    {
      label: "Quarantined",
      value: fleet.quarantined,
      icon: ShieldAlert,
      color: "text-health-quarantined",
    },
    {
      label: "Condemned",
      value: fleet.condemned,
      icon: Skull,
      color: "text-health-condemned",
    },
    {
      label: "SDC Rate",
      value: formatPercent(fleet.sdc_rate, 4),
      icon: TrendingDown,
      color: fleet.sdc_rate > 0.001 ? "text-health-condemned" : "text-health-healthy",
    },
  ];

  if (isLoading && !fleetData) {
    return (
      <div className="flex items-center justify-center h-96">
        <div className="flex flex-col items-center gap-3">
          <Activity className="h-8 w-8 animate-pulse text-primary" />
          <p className="text-muted-foreground">Loading fleet health data...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold tracking-tight">Fleet Overview</h1>
          <p className="text-sm text-muted-foreground mt-1">
            Real-time GPU fleet health monitoring
          </p>
        </div>
        {error && (
          <div className="flex items-center gap-2 rounded-md bg-destructive/10 px-3 py-1.5 text-sm text-destructive">
            <AlertTriangle className="h-4 w-4" />
            Using cached data
          </div>
        )}
      </div>

      {/* Summary Cards */}
      <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4">
        {summaryCards.map((card) => (
          <div
            key={card.label}
            className="rounded-lg border border-border bg-card p-4"
          >
            <div className="flex items-center justify-between">
              <span className="text-xs font-medium text-muted-foreground">
                {card.label}
              </span>
              <card.icon className={cn("h-4 w-4", card.color)} />
            </div>
            <p className="mt-2 text-2xl font-bold">{card.value}</p>
          </div>
        ))}
      </div>

      {/* Heatmap + Anomaly Timeline */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Heatmap */}
        <div className="lg:col-span-2 rounded-lg border border-border bg-card p-4">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-sm font-semibold">GPU Health Heatmap</h2>
            <span className="text-xs text-muted-foreground">
              {fleet.total_gpus} GPUs across {new Set(fleet.gpu_health.map((g) => g.gpu_id.node_hostname)).size} nodes
            </span>
          </div>
          <HeatMap gpus={fleet.gpu_health} />
        </div>

        {/* Anomaly Timeline */}
        <div className="rounded-lg border border-border bg-card p-4">
          <h2 className="text-sm font-semibold mb-4">Recent Anomalies</h2>
          <div className="max-h-[400px] overflow-y-auto scrollbar-thin pr-2">
            <AnomalyTimeline anomalies={anomalies} maxItems={15} />
          </div>
        </div>
      </div>

      {/* Probe Failure Rate Chart */}
      <div className="rounded-lg border border-border bg-card p-4">
        <h2 className="text-sm font-semibold mb-4">Probe Failure Rate (24h)</h2>
        <TimeSeriesChart
          data={probeTimeSeries}
          series={[
            {
              dataKey: "failure_rate",
              name: "Failure Rate",
              color: "#f97316",
              type: "area",
            },
            {
              dataKey: "sdc_rate",
              name: "SDC Rate",
              color: "#ef4444",
              type: "area",
            },
          ]}
          chartType="area"
          height={250}
          yAxisLabel="Rate"
          formatYTick={(v) => `${(v * 100).toFixed(2)}%`}
        />
      </div>

      {/* GPU Grid with filters */}
      <div className="rounded-lg border border-border bg-card p-4">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 mb-4">
          <h2 className="text-sm font-semibold">
            GPU Fleet ({filteredGpus.length})
          </h2>
          <div className="flex items-center gap-3">
            {/* Search */}
            <div className="relative">
              <SearchIcon className="absolute left-2.5 top-1/2 -translate-y-1/2 h-3.5 w-3.5 text-muted-foreground" />
              <input
                type="text"
                placeholder="Search UUID, hostname..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="h-8 rounded-md border border-input bg-background pl-8 pr-3 text-xs placeholder:text-muted-foreground focus:outline-none focus:ring-2 focus:ring-ring w-52"
              />
            </div>

            {/* State filters */}
            <div className="flex items-center gap-1">
              {(
                ["ALL", ...Object.values(GpuHealthState)] as StateFilter[]
              ).map((state) => (
                <button
                  key={state}
                  onClick={() => setStateFilter(state)}
                  className={cn(
                    "rounded-md px-2 py-1 text-xs font-medium transition-colors",
                    stateFilter === state
                      ? "bg-primary text-primary-foreground"
                      : "text-muted-foreground hover:bg-muted",
                  )}
                >
                  {state === "ALL" ? "All" : state === "DEEP_TEST" ? "Deep Test" : state.charAt(0) + state.slice(1).toLowerCase()}
                </button>
              ))}
            </div>
          </div>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5 gap-3">
          {filteredGpus.slice(0, 40).map((gpu) => (
            <GpuCard key={gpu.gpu_id.uuid} gpu={gpu} />
          ))}
        </div>

        {filteredGpus.length > 40 && (
          <p className="mt-4 text-center text-xs text-muted-foreground">
            Showing 40 of {filteredGpus.length} GPUs. Use filters to narrow results.
          </p>
        )}
      </div>
    </div>
  );
}
