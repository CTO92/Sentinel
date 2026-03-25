import { useState, useMemo } from "react";
import { useParams, useNavigate } from "react-router-dom";
import * as Tabs from "@radix-ui/react-tabs";
import {
  ArrowLeft,
  Activity,
  Cpu,
  Thermometer,
  Zap,
  MemoryStick,
  ShieldAlert,
  Clock,
  BarChart3,
} from "lucide-react";
import { format, parseISO } from "date-fns";
import { useGpuHealth, useGpuHistory, useQuarantineMutation } from "@/api/hooks";
import {
  GpuHealthState,
  type GpuHealth,
  type GpuHistoryData,
  type TimeRange,
  type SmHealth,
  type ProbeExecution,
  type AnomalyEvent,
  ProbeType,
  ProbeResult,
  AnomalySeverity,
} from "@/api/types";
import { cn, formatPercent, healthStateColor } from "@/lib/utils";
import { HealthBadge } from "@/components/HealthBadge";
import { TimeSeriesChart } from "@/components/TimeSeriesChart";
import { ProbeResultTable } from "@/components/ProbeResultTable";
import { AnomalyTimeline } from "@/components/AnomalyTimeline";
import { QuarantineDialog } from "@/components/QuarantineDialog";

// Mock data for when API is unavailable
function generateMockGpuHealth(uuid: string): GpuHealth {
  const stateRoll = Math.random();
  const state =
    stateRoll < 0.5 ? GpuHealthState.HEALTHY :
    stateRoll < 0.7 ? GpuHealthState.SUSPECT :
    stateRoll < 0.85 ? GpuHealthState.QUARANTINED :
    stateRoll < 0.95 ? GpuHealthState.DEEP_TEST :
    GpuHealthState.CONDEMNED;

  const reliability = state === GpuHealthState.HEALTHY
    ? 0.95 + Math.random() * 0.05
    : 0.3 + Math.random() * 0.5;

  const smHealth: SmHealth[] = Array.from({ length: 108 }, (_, i) => ({
    sm_id: { gpu_uuid: uuid, sm_index: i },
    state: Math.random() < 0.95 ? GpuHealthState.HEALTHY : GpuHealthState.SUSPECT,
    failure_count: Math.floor(Math.random() * 5),
    last_failure_time: new Date(Date.now() - Math.random() * 86400000).toISOString(),
    error_pattern: Math.random() < 0.05 ? "bit_flip_cluster" : "",
  }));

  return {
    gpu_id: {
      uuid,
      node_hostname: "gpu-node-007",
      pci_bus_id: "0000:3b:00.0",
      gpu_index: 3,
      model: "H100-SXM5-80GB",
      vbios_version: "96.00.89.00.01",
      driver_version: "535.129.03",
    },
    state,
    reliability_score: reliability,
    bayesian_alpha: 847,
    bayesian_beta: Math.floor((1 - reliability) * 100),
    total_probes: 2347,
    failed_probes: Math.floor(2347 * (1 - reliability) * 0.1),
    last_probe_time: new Date(Date.now() - 120000).toISOString(),
    last_state_change: new Date(Date.now() - 3600000 * 6).toISOString(),
    sm_health: smHealth,
    temperature_celsius: 62,
    power_watts: 312,
    memory_used_mb: 45000,
    memory_total_mb: 81920,
    gpu_utilization: 0.78,
  };
}

function generateMockHistory(uuid: string): GpuHistoryData {
  const now = Date.now();
  const probeTypes = Object.values(ProbeType);
  const probeResults = Object.values(ProbeResult);

  return {
    gpu_uuid: uuid,
    reliability_scores: Array.from({ length: 96 }, (_, i) => ({
      timestamp: new Date(now - (95 - i) * 900000).toISOString(),
      value: 0.92 + Math.random() * 0.08 - (i > 70 ? Math.random() * 0.1 : 0),
    })),
    probe_results: Array.from({ length: 50 }, (_, i) => {
      const started = new Date(now - (49 - i) * 600000);
      const duration = 200 + Math.floor(Math.random() * 3000);
      const result = Math.random() < 0.92 ? ProbeResult.PASS : probeResults[1 + Math.floor(Math.random() * 4)];
      return {
        probe_id: `PRB-${String(i).padStart(6, "0")}`,
        gpu_id: {
          uuid,
          node_hostname: "gpu-node-007",
          pci_bus_id: "0000:3b:00.0",
          gpu_index: 3,
          model: "H100-SXM5-80GB",
          vbios_version: "96.00.89.00.01",
          driver_version: "535.129.03",
        },
        probe_type: probeTypes[i % probeTypes.length],
        result,
        started_at: started.toISOString(),
        completed_at: new Date(started.getTime() + duration).toISOString(),
        duration_ms: duration,
        max_relative_error: result === ProbeResult.PASS ? 0 : Math.random() * 0.01,
        bit_flip_count: result === ProbeResult.FAIL_SILENT_CORRUPTION ? Math.floor(Math.random() * 10) : 0,
        affected_sm_indices: result !== ProbeResult.PASS ? [14, 15] : [],
        reference_gpu_uuid: "GPU-0001",
        error_details: result !== ProbeResult.PASS ? "Deviation detected in output tensor" : "",
        metadata: {},
      };
    }),
    anomalies: Array.from({ length: 8 }, (_, i) => ({
      event_id: `EVT-${String(i).padStart(6, "0")}`,
      gpu_uuid: uuid,
      node_hostname: "gpu-node-007",
      timestamp: new Date(now - i * 3600000 - Math.random() * 3600000).toISOString(),
      severity: ([AnomalySeverity.CRITICAL, AnomalySeverity.HIGH, AnomalySeverity.MEDIUM, AnomalySeverity.LOW])[Math.floor(Math.random() * 4)],
      anomaly_type: ["SDC_DETECTED", "RELIABILITY_DROP", "BIT_FLIP_CLUSTER"][Math.floor(Math.random() * 3)],
      description: [
        "Silent data corruption detected in matrix multiply",
        "Reliability score dropped below 0.90 threshold",
        "Bit flip cluster detected in SM 14-16",
      ][Math.floor(Math.random() * 3)],
      affected_sms: [14, 15, 16],
      resolved: Math.random() > 0.5,
    })),
    temperature: Array.from({ length: 96 }, (_, i) => ({
      timestamp: new Date(now - (95 - i) * 900000).toISOString(),
      value: 55 + Math.random() * 25,
    })),
    power: Array.from({ length: 96 }, (_, i) => ({
      timestamp: new Date(now - (95 - i) * 900000).toISOString(),
      value: 200 + Math.random() * 200,
    })),
    utilization: Array.from({ length: 96 }, (_, i) => ({
      timestamp: new Date(now - (95 - i) * 900000).toISOString(),
      value: Math.random(),
    })),
  };
}

export default function GpuDetail() {
  const { uuid } = useParams<{ uuid: string }>();
  const navigate = useNavigate();
  const [timeRange, setTimeRange] = useState<TimeRange>("24h");
  const [quarantineOpen, setQuarantineOpen] = useState(false);

  const gpuUuid = uuid ?? "";
  const { data: gpuData } = useGpuHealth(gpuUuid);
  const { data: historyData } = useGpuHistory(gpuUuid, timeRange);
  const quarantineMutation = useQuarantineMutation();

  const gpu = useMemo(() => gpuData ?? generateMockGpuHealth(gpuUuid || "GPU-0007-demo"), [gpuData, gpuUuid]);
  const history = useMemo(() => historyData ?? generateMockHistory(gpuUuid || "GPU-0007-demo"), [historyData, gpuUuid]);

  const timeRanges: TimeRange[] = ["1h", "6h", "24h", "7d", "30d"];

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center gap-4">
        <button
          onClick={() => navigate(-1)}
          className="rounded-md p-1.5 hover:bg-muted transition-colors"
        >
          <ArrowLeft className="h-5 w-5" />
        </button>
        <div className="flex-1">
          <div className="flex items-center gap-3">
            <h1 className="text-2xl font-bold tracking-tight font-mono">
              {gpu.gpu_id.uuid}
            </h1>
            <HealthBadge state={gpu.state} size="lg" />
          </div>
          <p className="text-sm text-muted-foreground mt-1">
            {gpu.gpu_id.node_hostname} | {gpu.gpu_id.model} | PCI {gpu.gpu_id.pci_bus_id}
          </p>
        </div>
        <button
          onClick={() => setQuarantineOpen(true)}
          className="flex items-center gap-2 rounded-md bg-primary px-4 py-2 text-sm font-medium text-primary-foreground hover:bg-primary/90 transition-colors"
        >
          <ShieldAlert className="h-4 w-4" />
          Action
        </button>
      </div>

      {/* Summary Cards */}
      <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-4">
        <div className="rounded-lg border border-border bg-card p-4">
          <div className="flex items-center justify-between">
            <span className="text-xs text-muted-foreground">Reliability</span>
            <Activity className="h-4 w-4 text-primary" />
          </div>
          <p className="mt-2 text-2xl font-bold">{formatPercent(gpu.reliability_score)}</p>
        </div>
        <div className="rounded-lg border border-border bg-card p-4">
          <div className="flex items-center justify-between">
            <span className="text-xs text-muted-foreground">Bayesian a/b</span>
            <BarChart3 className="h-4 w-4 text-primary" />
          </div>
          <p className="mt-2 text-2xl font-bold">
            {gpu.bayesian_alpha}<span className="text-sm text-muted-foreground">/{gpu.bayesian_beta}</span>
          </p>
        </div>
        <div className="rounded-lg border border-border bg-card p-4">
          <div className="flex items-center justify-between">
            <span className="text-xs text-muted-foreground">Probes</span>
            <Cpu className="h-4 w-4 text-primary" />
          </div>
          <p className="mt-2 text-2xl font-bold">
            {gpu.total_probes - gpu.failed_probes}
            <span className="text-sm text-muted-foreground">/{gpu.total_probes}</span>
          </p>
        </div>
        <div className="rounded-lg border border-border bg-card p-4">
          <div className="flex items-center justify-between">
            <span className="text-xs text-muted-foreground">Temperature</span>
            <Thermometer className="h-4 w-4 text-health-quarantined" />
          </div>
          <p className="mt-2 text-2xl font-bold">{gpu.temperature_celsius}°C</p>
        </div>
        <div className="rounded-lg border border-border bg-card p-4">
          <div className="flex items-center justify-between">
            <span className="text-xs text-muted-foreground">Power</span>
            <Zap className="h-4 w-4 text-health-suspect" />
          </div>
          <p className="mt-2 text-2xl font-bold">{gpu.power_watts}W</p>
        </div>
        <div className="rounded-lg border border-border bg-card p-4">
          <div className="flex items-center justify-between">
            <span className="text-xs text-muted-foreground">Memory</span>
            <MemoryStick className="h-4 w-4 text-primary" />
          </div>
          <p className="mt-2 text-2xl font-bold">
            {(gpu.memory_used_mb / 1024).toFixed(0)}
            <span className="text-sm text-muted-foreground">/{(gpu.memory_total_mb / 1024).toFixed(0)}GB</span>
          </p>
        </div>
      </div>

      {/* Time Range Selector */}
      <div className="flex items-center gap-2">
        <Clock className="h-4 w-4 text-muted-foreground" />
        <span className="text-sm text-muted-foreground">Time Range:</span>
        {timeRanges.map((tr) => (
          <button
            key={tr}
            onClick={() => setTimeRange(tr)}
            className={cn(
              "rounded-md px-3 py-1 text-xs font-medium transition-colors",
              timeRange === tr
                ? "bg-primary text-primary-foreground"
                : "text-muted-foreground hover:bg-muted",
            )}
          >
            {tr}
          </button>
        ))}
      </div>

      {/* Tabs */}
      <Tabs.Root defaultValue="reliability">
        <Tabs.List className="flex border-b border-border">
          {[
            { value: "reliability", label: "Reliability" },
            { value: "probes", label: "Probe Results" },
            { value: "telemetry", label: "Telemetry" },
            { value: "anomalies", label: "Anomalies" },
            { value: "sm-health", label: "SM Health" },
          ].map((tab) => (
            <Tabs.Trigger
              key={tab.value}
              value={tab.value}
              className={cn(
                "px-4 py-2 text-sm font-medium transition-colors border-b-2 border-transparent -mb-px",
                "data-[state=active]:border-primary data-[state=active]:text-foreground",
                "text-muted-foreground hover:text-foreground",
              )}
            >
              {tab.label}
            </Tabs.Trigger>
          ))}
        </Tabs.List>

        {/* Reliability Tab */}
        <Tabs.Content value="reliability" className="mt-4">
          <div className="rounded-lg border border-border bg-card p-4">
            <h3 className="text-sm font-semibold mb-4">Reliability Score Over Time</h3>
            <TimeSeriesChart
              data={history.reliability_scores}
              series={[
                { dataKey: "value", name: "Reliability Score", color: "#3b82f6" },
              ]}
              height={350}
              yDomain={[0, 1]}
              formatYTick={(v) => `${(v * 100).toFixed(0)}%`}
              yAxisLabel="Reliability"
            />
          </div>
        </Tabs.Content>

        {/* Probe Results Tab */}
        <Tabs.Content value="probes" className="mt-4">
          <div className="rounded-lg border border-border bg-card p-4">
            <h3 className="text-sm font-semibold mb-4">
              Probe Execution History ({history.probe_results.length} probes)
            </h3>
            <ProbeResultTable probes={history.probe_results} />
          </div>
        </Tabs.Content>

        {/* Telemetry Tab */}
        <Tabs.Content value="telemetry" className="mt-4 space-y-4">
          <div className="rounded-lg border border-border bg-card p-4">
            <h3 className="text-sm font-semibold mb-4">Temperature</h3>
            <TimeSeriesChart
              data={history.temperature}
              series={[
                { dataKey: "value", name: "Temperature (°C)", color: "#f97316" },
              ]}
              height={250}
              yAxisLabel="°C"
            />
          </div>
          <div className="rounded-lg border border-border bg-card p-4">
            <h3 className="text-sm font-semibold mb-4">Power Draw</h3>
            <TimeSeriesChart
              data={history.power}
              series={[
                { dataKey: "value", name: "Power (W)", color: "#eab308" },
              ]}
              height={250}
              yAxisLabel="Watts"
            />
          </div>
          <div className="rounded-lg border border-border bg-card p-4">
            <h3 className="text-sm font-semibold mb-4">GPU Utilization</h3>
            <TimeSeriesChart
              data={history.utilization}
              series={[
                { dataKey: "value", name: "Utilization", color: "#8b5cf6", type: "area" },
              ]}
              chartType="area"
              height={250}
              yDomain={[0, 1]}
              formatYTick={(v) => `${(v * 100).toFixed(0)}%`}
              yAxisLabel="Utilization"
            />
          </div>
        </Tabs.Content>

        {/* Anomalies Tab */}
        <Tabs.Content value="anomalies" className="mt-4">
          <div className="rounded-lg border border-border bg-card p-4">
            <h3 className="text-sm font-semibold mb-4">
              Anomaly Events ({history.anomalies.length})
            </h3>
            <AnomalyTimeline anomalies={history.anomalies} />
          </div>
        </Tabs.Content>

        {/* SM Health Tab */}
        <Tabs.Content value="sm-health" className="mt-4">
          <div className="rounded-lg border border-border bg-card p-4">
            <h3 className="text-sm font-semibold mb-4">
              Streaming Multiprocessor Health ({gpu.sm_health.length} SMs)
            </h3>
            <div className="grid grid-cols-12 sm:grid-cols-18 md:grid-cols-27 gap-1">
              {gpu.sm_health.map((sm) => (
                <div
                  key={sm.sm_id.sm_index}
                  className="group relative aspect-square rounded-sm cursor-default"
                  style={{ backgroundColor: healthStateColor(sm.state) }}
                  title={`SM ${sm.sm_id.sm_index} | ${sm.state} | ${sm.failure_count} failures`}
                >
                  <div className="absolute bottom-full left-1/2 -translate-x-1/2 mb-1 hidden group-hover:block z-10">
                    <div className="bg-popover text-popover-foreground rounded px-2 py-1 text-[10px] whitespace-nowrap shadow-lg border border-border">
                      <p className="font-semibold">SM {sm.sm_id.sm_index}</p>
                      <p>{sm.state}</p>
                      <p>{sm.failure_count} failures</p>
                    </div>
                  </div>
                </div>
              ))}
            </div>
            <div className="mt-4">
              <h4 className="text-xs font-medium mb-2 text-muted-foreground">
                Failing SMs
              </h4>
              {gpu.sm_health.filter((s) => s.state !== GpuHealthState.HEALTHY).length === 0 ? (
                <p className="text-xs text-muted-foreground">All SMs healthy</p>
              ) : (
                <div className="space-y-1">
                  {gpu.sm_health
                    .filter((s) => s.state !== GpuHealthState.HEALTHY)
                    .map((sm) => (
                      <div
                        key={sm.sm_id.sm_index}
                        className="flex items-center justify-between rounded-md bg-muted px-3 py-2 text-xs"
                      >
                        <span className="font-mono">SM {sm.sm_id.sm_index}</span>
                        <HealthBadge state={sm.state} size="sm" />
                        <span>{sm.failure_count} failures</span>
                        <span className="text-muted-foreground">
                          Last: {format(parseISO(sm.last_failure_time), "HH:mm:ss")}
                        </span>
                      </div>
                    ))}
                </div>
              )}
            </div>
          </div>
        </Tabs.Content>
      </Tabs.Root>

      {/* Quarantine Dialog */}
      <QuarantineDialog
        open={quarantineOpen}
        onOpenChange={setQuarantineOpen}
        gpuUuid={gpu.gpu_id.uuid}
        gpuHostname={gpu.gpu_id.node_hostname}
        currentState={gpu.state}
        onConfirm={(directive) => {
          quarantineMutation.mutate(directive, {
            onSuccess: () => setQuarantineOpen(false),
          });
        }}
        isSubmitting={quarantineMutation.isPending}
      />
    </div>
  );
}
