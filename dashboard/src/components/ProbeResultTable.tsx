import { useState, useMemo } from "react";
import { format, parseISO } from "date-fns";
import {
  ChevronUp,
  ChevronDown,
  ChevronsUpDown,
  CheckCircle,
  XCircle,
  AlertTriangle,
  Clock,
  Server,
} from "lucide-react";
import { cn } from "@/lib/utils";
import type { ProbeExecution, ProbeResult } from "@/api/types";

interface ProbeResultTableProps {
  probes: ProbeExecution[];
  className?: string;
}

type SortField = "started_at" | "probe_type" | "result" | "duration_ms" | "max_relative_error";
type SortDir = "asc" | "desc";

function resultIcon(result: ProbeResult) {
  switch (result) {
    case "PASS":
      return <CheckCircle className="h-4 w-4 text-health-healthy" />;
    case "FAIL_SILENT_CORRUPTION":
      return <XCircle className="h-4 w-4 text-health-condemned" />;
    case "FAIL_DETECTED_ERROR":
      return <AlertTriangle className="h-4 w-4 text-health-quarantined" />;
    case "FAIL_TIMEOUT":
      return <Clock className="h-4 w-4 text-health-suspect" />;
    case "FAIL_INFRASTRUCTURE":
      return <Server className="h-4 w-4 text-muted-foreground" />;
    default:
      return null;
  }
}

function resultLabel(result: ProbeResult): string {
  return result.replace(/_/g, " ").replace("FAIL ", "");
}

export function ProbeResultTable({ probes, className }: ProbeResultTableProps) {
  const [sortField, setSortField] = useState<SortField>("started_at");
  const [sortDir, setSortDir] = useState<SortDir>("desc");

  const handleSort = (field: SortField) => {
    if (sortField === field) {
      setSortDir((d) => (d === "asc" ? "desc" : "asc"));
    } else {
      setSortField(field);
      setSortDir("desc");
    }
  };

  const sorted = useMemo(() => {
    const arr = [...probes];
    arr.sort((a, b) => {
      let cmp = 0;
      const fa = a[sortField];
      const fb = b[sortField];
      if (typeof fa === "string" && typeof fb === "string") {
        cmp = fa.localeCompare(fb);
      } else if (typeof fa === "number" && typeof fb === "number") {
        cmp = fa - fb;
      }
      return sortDir === "asc" ? cmp : -cmp;
    });
    return arr;
  }, [probes, sortField, sortDir]);

  const SortIcon = ({ field }: { field: SortField }) => {
    if (sortField !== field) return <ChevronsUpDown className="h-3 w-3 opacity-50" />;
    return sortDir === "asc" ? (
      <ChevronUp className="h-3 w-3" />
    ) : (
      <ChevronDown className="h-3 w-3" />
    );
  };

  return (
    <div className={cn("overflow-auto scrollbar-thin", className)}>
      <table className="w-full text-sm">
        <thead>
          <tr className="border-b border-border">
            {(
              [
                ["started_at", "Time"],
                ["probe_type", "Probe Type"],
                ["result", "Result"],
                ["duration_ms", "Duration"],
                ["max_relative_error", "Max Error"],
              ] as [SortField, string][]
            ).map(([field, label]) => (
              <th
                key={field}
                className="cursor-pointer select-none px-3 py-2 text-left font-medium text-muted-foreground hover:text-foreground transition-colors"
                onClick={() => handleSort(field)}
              >
                <span className="inline-flex items-center gap-1">
                  {label}
                  <SortIcon field={field} />
                </span>
              </th>
            ))}
            <th className="px-3 py-2 text-left font-medium text-muted-foreground">
              Details
            </th>
          </tr>
        </thead>
        <tbody>
          {sorted.map((probe) => (
            <tr
              key={probe.probe_id}
              className="border-b border-border/50 hover:bg-muted/50 transition-colors"
            >
              <td className="px-3 py-2 font-mono text-xs whitespace-nowrap">
                {format(parseISO(probe.started_at), "MMM dd HH:mm:ss")}
              </td>
              <td className="px-3 py-2 text-xs">
                {probe.probe_type.replace(/_/g, " ")}
              </td>
              <td className="px-3 py-2">
                <span className="inline-flex items-center gap-1.5 text-xs">
                  {resultIcon(probe.result)}
                  {resultLabel(probe.result)}
                </span>
              </td>
              <td className="px-3 py-2 text-xs font-mono">
                {probe.duration_ms}ms
              </td>
              <td className="px-3 py-2 text-xs font-mono">
                {probe.max_relative_error > 0
                  ? probe.max_relative_error.toExponential(2)
                  : "-"}
              </td>
              <td className="px-3 py-2 text-xs text-muted-foreground max-w-[200px] truncate">
                {probe.error_details || "-"}
              </td>
            </tr>
          ))}
          {sorted.length === 0 && (
            <tr>
              <td
                colSpan={6}
                className="px-3 py-8 text-center text-muted-foreground"
              >
                No probe results available
              </td>
            </tr>
          )}
        </tbody>
      </table>
    </div>
  );
}
