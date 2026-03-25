import { useState, useMemo } from "react";
import { useNavigate } from "react-router-dom";
import { format, parseISO, formatDistanceToNow } from "date-fns";
import {
  ShieldAlert,
  ShieldCheck,
  Search,
  Skull,
  Clock,
  AlertTriangle,
  ExternalLink,
  Filter,
} from "lucide-react";
import { useQuarantineQueue, useQuarantineMutation } from "@/api/hooks";
import {
  GpuHealthState,
  ProbeResult,
  QuarantineAction,
  type QuarantineEntry,
  type QuarantineDirective,
} from "@/api/types";
import { cn, formatPercent } from "@/lib/utils";
import { HealthBadge } from "@/components/HealthBadge";
import { QuarantineDialog } from "@/components/QuarantineDialog";

function generateMockQueue(): QuarantineEntry[] {
  const reasons = [
    "Reliability score dropped below 0.80 threshold",
    "Silent data corruption detected in consecutive probes",
    "Bit flip cluster detected in SM 14-16",
    "Cross-GPU comparison failed 3 consecutive times",
    "Temperature spike correlated with SDC event",
    "Bayesian posterior indicates high failure probability",
    "Manual quarantine by operator due to suspected hardware fault",
  ];

  const states = [
    GpuHealthState.SUSPECT,
    GpuHealthState.QUARANTINED,
    GpuHealthState.DEEP_TEST,
    GpuHealthState.CONDEMNED,
  ];

  return Array.from({ length: 18 }, (_, i) => {
    const state = states[i % states.length];
    return {
      id: `QRN-${String(i).padStart(6, "0")}`,
      gpu_id: {
        uuid: `GPU-${String(100 + i).padStart(4, "0")}-${Math.random().toString(36).slice(2, 10)}`,
        node_hostname: `gpu-node-${String(Math.floor(Math.random() * 32)).padStart(3, "0")}`,
        pci_bus_id: `0000:${(i % 8).toString(16).padStart(2, "0")}:00.0`,
        gpu_index: i % 8,
        model: i % 2 === 0 ? "A100-SXM4-80GB" : "H100-SXM5-80GB",
        vbios_version: "96.00.89.00.01",
        driver_version: "535.129.03",
      },
      state,
      entered_at: new Date(
        Date.now() - Math.random() * 86400000 * 7,
      ).toISOString(),
      reason: reasons[i % reasons.length],
      evidence_ids: [`EVT-${String(i * 3).padStart(6, "0")}`, `PRB-${String(i * 7).padStart(6, "0")}`],
      initiated_by: i % 3 === 0 ? "dashboard_user" : "correlation-engine",
      reliability_score: 0.2 + Math.random() * 0.5,
      anomaly_count: 1 + Math.floor(Math.random() * 12),
      last_probe_result:
        state === GpuHealthState.CONDEMNED
          ? ProbeResult.FAIL_SILENT_CORRUPTION
          : state === GpuHealthState.DEEP_TEST
            ? ProbeResult.PASS
            : ProbeResult.FAIL_DETECTED_ERROR,
    };
  });
}

type QueueFilter = "ALL" | GpuHealthState;

export default function QuarantineQueue() {
  const navigate = useNavigate();
  const { data: queueData } = useQuarantineQueue();
  const quarantineMutation = useQuarantineMutation();
  const [filter, setFilter] = useState<QueueFilter>("ALL");
  const [dialogState, setDialogState] = useState<{
    open: boolean;
    entry: QuarantineEntry | null;
  }>({ open: false, entry: null });

  const queue = useMemo(() => queueData ?? generateMockQueue(), [queueData]);

  const filteredQueue = useMemo(() => {
    if (filter === "ALL") return queue;
    return queue.filter((e) => e.state === filter);
  }, [queue, filter]);

  const stateCountMap = useMemo(() => {
    const m: Record<string, number> = {};
    for (const e of queue) {
      m[e.state] = (m[e.state] || 0) + 1;
    }
    return m;
  }, [queue]);

  const handleAction = (directive: QuarantineDirective) => {
    quarantineMutation.mutate(directive, {
      onSuccess: () => setDialogState({ open: false, entry: null }),
    });
  };

  const filterStates: QueueFilter[] = [
    "ALL",
    GpuHealthState.SUSPECT,
    GpuHealthState.QUARANTINED,
    GpuHealthState.DEEP_TEST,
    GpuHealthState.CONDEMNED,
  ];

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold tracking-tight">Quarantine Queue</h1>
        <p className="text-sm text-muted-foreground mt-1">
          Manage GPUs requiring attention
        </p>
      </div>

      {/* Summary Cards */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <div className="rounded-lg border border-border bg-card p-4">
          <div className="flex items-center justify-between">
            <span className="text-xs text-muted-foreground">Suspect</span>
            <AlertTriangle className="h-4 w-4 text-health-suspect" />
          </div>
          <p className="mt-2 text-2xl font-bold">
            {stateCountMap[GpuHealthState.SUSPECT] ?? 0}
          </p>
        </div>
        <div className="rounded-lg border border-border bg-card p-4">
          <div className="flex items-center justify-between">
            <span className="text-xs text-muted-foreground">Quarantined</span>
            <ShieldAlert className="h-4 w-4 text-health-quarantined" />
          </div>
          <p className="mt-2 text-2xl font-bold">
            {stateCountMap[GpuHealthState.QUARANTINED] ?? 0}
          </p>
        </div>
        <div className="rounded-lg border border-border bg-card p-4">
          <div className="flex items-center justify-between">
            <span className="text-xs text-muted-foreground">Deep Test</span>
            <Search className="h-4 w-4 text-health-deep-test" />
          </div>
          <p className="mt-2 text-2xl font-bold">
            {stateCountMap[GpuHealthState.DEEP_TEST] ?? 0}
          </p>
        </div>
        <div className="rounded-lg border border-border bg-card p-4">
          <div className="flex items-center justify-between">
            <span className="text-xs text-muted-foreground">Condemned</span>
            <Skull className="h-4 w-4 text-health-condemned" />
          </div>
          <p className="mt-2 text-2xl font-bold">
            {stateCountMap[GpuHealthState.CONDEMNED] ?? 0}
          </p>
        </div>
      </div>

      {/* Filter tabs */}
      <div className="flex items-center gap-2">
        <Filter className="h-4 w-4 text-muted-foreground" />
        {filterStates.map((state) => (
          <button
            key={state}
            onClick={() => setFilter(state)}
            className={cn(
              "rounded-md px-3 py-1.5 text-xs font-medium transition-colors",
              filter === state
                ? "bg-primary text-primary-foreground"
                : "text-muted-foreground hover:bg-muted",
            )}
          >
            {state === "ALL"
              ? `All (${queue.length})`
              : `${state.replace(/_/g, " ")} (${stateCountMap[state] ?? 0})`}
          </button>
        ))}
      </div>

      {/* Queue Items */}
      <div className="space-y-3">
        {filteredQueue.map((entry) => (
          <div
            key={entry.id}
            className="rounded-lg border border-border bg-card p-4 hover:border-primary/30 transition-colors"
          >
            <div className="flex items-start justify-between gap-4">
              <div className="min-w-0 flex-1">
                <div className="flex items-center gap-3 mb-2">
                  <button
                    onClick={() => navigate(`/gpu/${entry.gpu_id.uuid}`)}
                    className="font-mono text-sm font-medium hover:text-primary transition-colors inline-flex items-center gap-1"
                  >
                    {entry.gpu_id.uuid}
                    <ExternalLink className="h-3 w-3" />
                  </button>
                  <HealthBadge state={entry.state} size="sm" />
                  <span className="text-xs text-muted-foreground">
                    {entry.gpu_id.model}
                  </span>
                </div>
                <p className="text-sm text-muted-foreground mb-2">
                  {entry.reason}
                </p>
                <div className="flex flex-wrap items-center gap-3 text-xs text-muted-foreground">
                  <span className="flex items-center gap-1">
                    <Clock className="h-3 w-3" />
                    {formatDistanceToNow(parseISO(entry.entered_at), {
                      addSuffix: true,
                    })}
                  </span>
                  <span>
                    Reliability: {formatPercent(entry.reliability_score)}
                  </span>
                  <span>{entry.anomaly_count} anomalies</span>
                  <span>By: {entry.initiated_by}</span>
                  <span>Node: {entry.gpu_id.node_hostname}</span>
                </div>
              </div>

              <div className="flex items-center gap-2 flex-shrink-0">
                {entry.state !== GpuHealthState.CONDEMNED && (
                  <>
                    <button
                      onClick={() =>
                        setDialogState({ open: true, entry })
                      }
                      className="rounded-md border border-border px-3 py-1.5 text-xs font-medium hover:bg-muted transition-colors"
                    >
                      Action
                    </button>
                  </>
                )}
              </div>
            </div>

            {/* Evidence */}
            {entry.evidence_ids.length > 0 && (
              <div className="mt-3 pt-3 border-t border-border/50">
                <p className="text-[10px] font-medium text-muted-foreground mb-1">
                  Evidence
                </p>
                <div className="flex flex-wrap gap-1">
                  {entry.evidence_ids.map((id) => (
                    <span
                      key={id}
                      className="rounded bg-muted px-1.5 py-0.5 text-[10px] font-mono"
                    >
                      {id}
                    </span>
                  ))}
                </div>
              </div>
            )}
          </div>
        ))}

        {filteredQueue.length === 0 && (
          <div className="flex flex-col items-center justify-center py-12 text-muted-foreground">
            <ShieldCheck className="h-12 w-12 mb-3 opacity-50" />
            <p className="text-sm">No GPUs in this state</p>
          </div>
        )}
      </div>

      {/* Action Dialog */}
      {dialogState.entry && (
        <QuarantineDialog
          open={dialogState.open}
          onOpenChange={(open) =>
            setDialogState((s) => ({ ...s, open }))
          }
          gpuUuid={dialogState.entry.gpu_id.uuid}
          gpuHostname={dialogState.entry.gpu_id.node_hostname}
          currentState={dialogState.entry.state}
          onConfirm={handleAction}
          isSubmitting={quarantineMutation.isPending}
        />
      )}
    </div>
  );
}
