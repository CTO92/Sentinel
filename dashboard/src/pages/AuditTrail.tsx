import { useState, useMemo, useCallback } from "react";
import {
  Search,
  Download,
  ChevronDown,
  ChevronRight,
  CheckCircle,
  XCircle,
  Filter,
  Calendar,
} from "lucide-react";
import { format, parseISO, subDays } from "date-fns";
import * as Select from "@radix-ui/react-select";
import { useAuditTrail } from "@/api/hooks";
import {
  AuditEntryType,
  AnomalySeverity,
  type AuditEntry,
  type AuditFilters,
  type PaginatedAuditResponse,
} from "@/api/types";
import { cn, severityColor } from "@/lib/utils";

function generateMockAuditData(): PaginatedAuditResponse {
  const types = Object.values(AuditEntryType);
  const severities = Object.values(AnomalySeverity);
  const summaries = [
    "Probe MATRIX_MULTIPLY completed with PASS result on GPU-0042",
    "GPU-0127 state changed from HEALTHY to SUSPECT",
    "Quarantine action QUARANTINE issued for GPU-0127",
    "Anomaly detected: SDC in matrix multiply on GPU-0042",
    "Correlation found: 3 GPUs on node-007 showing elevated error rates",
    "Configuration updated: probe interval changed to 300s",
    "System startup: Correlation Engine initialized",
    "GPU-0089 deep test completed: 47/48 probes passed",
    "Trust graph coverage reached 94.2%",
    "Bayesian model updated for GPU-0042: alpha=847, beta=12",
  ];

  const entries: AuditEntry[] = Array.from({ length: 100 }, (_, i) => {
    const type = types[i % types.length];
    return {
      entry_id: `AUD-${String(i).padStart(8, "0")}`,
      timestamp: new Date(Date.now() - i * 600000 - Math.random() * 300000).toISOString(),
      entry_type: type,
      gpu_uuid: Math.random() > 0.2 ? `GPU-${String(Math.floor(Math.random() * 256)).padStart(4, "0")}` : undefined,
      node_hostname: `gpu-node-${String(Math.floor(Math.random() * 32)).padStart(3, "0")}`,
      severity: severities[Math.floor(Math.random() * severities.length)],
      summary: summaries[Math.floor(Math.random() * summaries.length)],
      details: JSON.stringify({
        probe_type: "MATRIX_MULTIPLY",
        duration_ms: 1200 + Math.floor(Math.random() * 2000),
        max_relative_error: Math.random() * 0.001,
        previous_state: "HEALTHY",
        new_state: "SUSPECT",
      }, null, 2),
      chain_hash: Math.random().toString(36).slice(2) + Math.random().toString(36).slice(2),
      previous_hash: Math.random().toString(36).slice(2) + Math.random().toString(36).slice(2),
      chain_verified: Math.random() > 0.02,
      actor: ["system", "probe-agent", "correlation-engine", "dashboard_user"][Math.floor(Math.random() * 4)],
    };
  });

  return {
    entries,
    total_count: 2847,
    page: 1,
    page_size: 100,
    chain_integrity_verified: true,
  };
}

function SelectItem({ value, children }: { value: string; children: React.ReactNode }) {
  return (
    <Select.Item
      value={value}
      className="relative flex cursor-pointer select-none items-center rounded-sm px-2 py-1.5 text-xs outline-none data-[highlighted]:bg-accent data-[highlighted]:text-accent-foreground"
    >
      <Select.ItemText>{children}</Select.ItemText>
    </Select.Item>
  );
}

export default function AuditTrail() {
  const [filters, setFilters] = useState<AuditFilters>({
    page: 1,
    page_size: 50,
  });
  const [searchQuery, setSearchQuery] = useState("");
  const [expandedId, setExpandedId] = useState<string | null>(null);

  const { data: auditData } = useAuditTrail(filters);
  const data = useMemo(() => auditData ?? generateMockAuditData(), [auditData]);

  const filteredEntries = useMemo(() => {
    let entries = data.entries;
    if (searchQuery) {
      const q = searchQuery.toLowerCase();
      entries = entries.filter(
        (e) =>
          e.summary.toLowerCase().includes(q) ||
          e.entry_id.toLowerCase().includes(q) ||
          (e.gpu_uuid && e.gpu_uuid.toLowerCase().includes(q)) ||
          (e.node_hostname && e.node_hostname.toLowerCase().includes(q)),
      );
    }
    return entries;
  }, [data.entries, searchQuery]);

  const exportData = useCallback(
    (fmt: "csv" | "json") => {
      let content: string;
      let mimeType: string;
      let filename: string;

      if (fmt === "json") {
        content = JSON.stringify(filteredEntries, null, 2);
        mimeType = "application/json";
        filename = "sentinel-audit-trail.json";
      } else {
        const headers = [
          "entry_id", "timestamp", "entry_type", "severity",
          "gpu_uuid", "node_hostname", "summary", "actor", "chain_verified",
        ];
        const rows = filteredEntries.map((e) =>
          headers.map((h) => {
            const val = e[h as keyof AuditEntry];
            return typeof val === "string" && val.includes(",")
              ? `"${val}"`
              : String(val ?? "");
          }).join(","),
        );
        content = [headers.join(","), ...rows].join("\n");
        mimeType = "text/csv";
        filename = "sentinel-audit-trail.csv";
      }

      const blob = new Blob([content], { type: mimeType });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = filename;
      a.click();
      URL.revokeObjectURL(url);
    },
    [filteredEntries],
  );

  const entryTypeLabel = (type: AuditEntryType): string =>
    type.replace(/_/g, " ");

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold tracking-tight">Audit Trail</h1>
          <p className="text-sm text-muted-foreground mt-1">
            Tamper-evident ledger of all SENTINEL operations
          </p>
        </div>
        <div className="flex items-center gap-2">
          {data.chain_integrity_verified ? (
            <span className="flex items-center gap-1.5 rounded-full bg-health-healthy/10 px-3 py-1 text-xs font-medium text-health-healthy border border-health-healthy/30">
              <CheckCircle className="h-3.5 w-3.5" />
              Chain Verified
            </span>
          ) : (
            <span className="flex items-center gap-1.5 rounded-full bg-health-condemned/10 px-3 py-1 text-xs font-medium text-health-condemned border border-health-condemned/30">
              <XCircle className="h-3.5 w-3.5" />
              Chain Broken
            </span>
          )}
        </div>
      </div>

      {/* Filters */}
      <div className="flex flex-wrap items-center gap-3">
        {/* Search */}
        <div className="relative flex-1 min-w-[200px] max-w-md">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
          <input
            type="text"
            placeholder="Search entries..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="h-9 w-full rounded-md border border-input bg-background pl-9 pr-3 text-sm placeholder:text-muted-foreground focus:outline-none focus:ring-2 focus:ring-ring"
          />
        </div>

        {/* Entry Type Filter */}
        <Select.Root
          value={filters.entry_type ?? "ALL"}
          onValueChange={(v) =>
            setFilters((f) => ({
              ...f,
              entry_type: v === "ALL" ? undefined : (v as AuditEntryType),
            }))
          }
        >
          <Select.Trigger className="inline-flex items-center gap-2 rounded-md border border-input bg-background px-3 h-9 text-xs hover:bg-muted transition-colors">
            <Filter className="h-3.5 w-3.5 text-muted-foreground" />
            <Select.Value placeholder="Entry Type" />
            <Select.Icon>
              <ChevronDown className="h-3.5 w-3.5 text-muted-foreground" />
            </Select.Icon>
          </Select.Trigger>
          <Select.Portal>
            <Select.Content className="z-50 rounded-md border border-border bg-popover p-1 shadow-lg">
              <Select.Viewport>
                <SelectItem value="ALL">All Types</SelectItem>
                {Object.values(AuditEntryType).map((t) => (
                  <SelectItem key={t} value={t}>
                    {entryTypeLabel(t)}
                  </SelectItem>
                ))}
              </Select.Viewport>
            </Select.Content>
          </Select.Portal>
        </Select.Root>

        {/* Severity Filter */}
        <Select.Root
          value={filters.severity ?? "ALL"}
          onValueChange={(v) =>
            setFilters((f) => ({
              ...f,
              severity: v === "ALL" ? undefined : (v as AnomalySeverity),
            }))
          }
        >
          <Select.Trigger className="inline-flex items-center gap-2 rounded-md border border-input bg-background px-3 h-9 text-xs hover:bg-muted transition-colors">
            <Filter className="h-3.5 w-3.5 text-muted-foreground" />
            <Select.Value placeholder="Severity" />
            <Select.Icon>
              <ChevronDown className="h-3.5 w-3.5 text-muted-foreground" />
            </Select.Icon>
          </Select.Trigger>
          <Select.Portal>
            <Select.Content className="z-50 rounded-md border border-border bg-popover p-1 shadow-lg">
              <Select.Viewport>
                <SelectItem value="ALL">All Severities</SelectItem>
                {Object.values(AnomalySeverity).map((s) => (
                  <SelectItem key={s} value={s}>{s}</SelectItem>
                ))}
              </Select.Viewport>
            </Select.Content>
          </Select.Portal>
        </Select.Root>

        {/* Export */}
        <div className="flex items-center gap-1 ml-auto">
          <button
            onClick={() => exportData("csv")}
            className="inline-flex items-center gap-1.5 rounded-md border border-input px-3 h-9 text-xs hover:bg-muted transition-colors"
          >
            <Download className="h-3.5 w-3.5" />
            CSV
          </button>
          <button
            onClick={() => exportData("json")}
            className="inline-flex items-center gap-1.5 rounded-md border border-input px-3 h-9 text-xs hover:bg-muted transition-colors"
          >
            <Download className="h-3.5 w-3.5" />
            JSON
          </button>
        </div>
      </div>

      {/* Results count */}
      <div className="text-xs text-muted-foreground">
        Showing {filteredEntries.length} of {data.total_count} entries
      </div>

      {/* Audit Table */}
      <div className="rounded-lg border border-border bg-card overflow-hidden">
        <div className="overflow-auto scrollbar-thin max-h-[calc(100vh-300px)]">
          <table className="w-full text-sm">
            <thead className="sticky top-0 bg-card z-10">
              <tr className="border-b border-border">
                <th className="w-8 px-3 py-2" />
                <th className="px-3 py-2 text-left text-xs font-medium text-muted-foreground">
                  Time
                </th>
                <th className="px-3 py-2 text-left text-xs font-medium text-muted-foreground">
                  Type
                </th>
                <th className="px-3 py-2 text-left text-xs font-medium text-muted-foreground">
                  Severity
                </th>
                <th className="px-3 py-2 text-left text-xs font-medium text-muted-foreground">
                  GPU
                </th>
                <th className="px-3 py-2 text-left text-xs font-medium text-muted-foreground">
                  Summary
                </th>
                <th className="px-3 py-2 text-left text-xs font-medium text-muted-foreground">
                  Actor
                </th>
                <th className="px-3 py-2 text-left text-xs font-medium text-muted-foreground">
                  Chain
                </th>
              </tr>
            </thead>
            <tbody>
              {filteredEntries.map((entry) => (
                <>
                  <tr
                    key={entry.entry_id}
                    className="border-b border-border/50 hover:bg-muted/30 cursor-pointer transition-colors"
                    onClick={() =>
                      setExpandedId(
                        expandedId === entry.entry_id ? null : entry.entry_id,
                      )
                    }
                  >
                    <td className="px-3 py-2">
                      {expandedId === entry.entry_id ? (
                        <ChevronDown className="h-3.5 w-3.5 text-muted-foreground" />
                      ) : (
                        <ChevronRight className="h-3.5 w-3.5 text-muted-foreground" />
                      )}
                    </td>
                    <td className="px-3 py-2 text-xs font-mono whitespace-nowrap">
                      {format(parseISO(entry.timestamp), "MMM dd HH:mm:ss")}
                    </td>
                    <td className="px-3 py-2">
                      <span className="rounded bg-muted px-1.5 py-0.5 text-[10px] font-medium">
                        {entryTypeLabel(entry.entry_type)}
                      </span>
                    </td>
                    <td className="px-3 py-2">
                      <span
                        className={cn(
                          "text-xs font-semibold",
                          severityColor(entry.severity),
                        )}
                      >
                        {entry.severity}
                      </span>
                    </td>
                    <td className="px-3 py-2 text-xs font-mono">
                      {entry.gpu_uuid?.slice(0, 12) ?? "-"}
                    </td>
                    <td className="px-3 py-2 text-xs max-w-[300px] truncate">
                      {entry.summary}
                    </td>
                    <td className="px-3 py-2 text-xs text-muted-foreground">
                      {entry.actor}
                    </td>
                    <td className="px-3 py-2">
                      {entry.chain_verified ? (
                        <CheckCircle className="h-3.5 w-3.5 text-health-healthy" />
                      ) : (
                        <XCircle className="h-3.5 w-3.5 text-health-condemned" />
                      )}
                    </td>
                  </tr>
                  {expandedId === entry.entry_id && (
                    <tr key={`${entry.entry_id}-detail`} className="bg-muted/20">
                      <td colSpan={8} className="px-6 py-4">
                        <div className="grid grid-cols-2 gap-4 text-xs">
                          <div>
                            <p className="font-medium text-muted-foreground mb-1">
                              Entry ID
                            </p>
                            <p className="font-mono">{entry.entry_id}</p>
                          </div>
                          <div>
                            <p className="font-medium text-muted-foreground mb-1">
                              Node
                            </p>
                            <p>{entry.node_hostname ?? "-"}</p>
                          </div>
                          <div>
                            <p className="font-medium text-muted-foreground mb-1">
                              Chain Hash
                            </p>
                            <p className="font-mono break-all">
                              {entry.chain_hash}
                            </p>
                          </div>
                          <div>
                            <p className="font-medium text-muted-foreground mb-1">
                              Previous Hash
                            </p>
                            <p className="font-mono break-all">
                              {entry.previous_hash}
                            </p>
                          </div>
                          <div className="col-span-2">
                            <p className="font-medium text-muted-foreground mb-1">
                              Details
                            </p>
                            <pre className="rounded-md bg-muted p-3 text-[11px] font-mono overflow-auto max-h-[200px] scrollbar-thin">
                              {entry.details}
                            </pre>
                          </div>
                        </div>
                      </td>
                    </tr>
                  )}
                </>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
