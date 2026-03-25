import { format, parseISO } from "date-fns";
import { AlertTriangle, AlertOctagon, Info, Bell, ShieldAlert } from "lucide-react";
import { cn, severityColor } from "@/lib/utils";
import type { AnomalyEvent, AnomalySeverity } from "@/api/types";

interface AnomalyTimelineProps {
  anomalies: AnomalyEvent[];
  maxItems?: number;
  className?: string;
}

function severityIcon(severity: AnomalySeverity) {
  switch (severity) {
    case "CRITICAL":
      return <AlertOctagon className="h-4 w-4" />;
    case "HIGH":
      return <ShieldAlert className="h-4 w-4" />;
    case "MEDIUM":
      return <AlertTriangle className="h-4 w-4" />;
    case "LOW":
      return <Bell className="h-4 w-4" />;
    case "INFO":
      return <Info className="h-4 w-4" />;
    default:
      return <Info className="h-4 w-4" />;
  }
}

export function AnomalyTimeline({
  anomalies,
  maxItems,
  className,
}: AnomalyTimelineProps) {
  const displayed = maxItems ? anomalies.slice(0, maxItems) : anomalies;

  return (
    <div className={cn("space-y-0", className)}>
      {displayed.map((event, idx) => (
        <div key={event.event_id} className="relative flex gap-3 pb-4">
          {/* Timeline connector */}
          {idx < displayed.length - 1 && (
            <div className="absolute left-[11px] top-7 h-full w-px bg-border" />
          )}

          {/* Icon */}
          <div
            className={cn(
              "mt-0.5 flex-shrink-0",
              severityColor(event.severity),
            )}
          >
            {severityIcon(event.severity)}
          </div>

          {/* Content */}
          <div className="min-w-0 flex-1">
            <div className="flex items-start justify-between gap-2">
              <p className="text-sm font-medium leading-snug">
                {event.description}
              </p>
              <time className="flex-shrink-0 text-[10px] text-muted-foreground whitespace-nowrap">
                {format(parseISO(event.timestamp), "HH:mm:ss")}
              </time>
            </div>
            <div className="mt-1 flex flex-wrap items-center gap-2 text-xs text-muted-foreground">
              <span className="font-mono">
                {event.gpu_uuid.slice(0, 12)}
              </span>
              <span>{event.node_hostname}</span>
              <span
                className={cn(
                  "rounded-full px-1.5 py-0.5 text-[10px] font-semibold uppercase",
                  severityColor(event.severity),
                )}
              >
                {event.severity}
              </span>
              {event.anomaly_type && (
                <span className="rounded bg-muted px-1.5 py-0.5 text-[10px]">
                  {event.anomaly_type}
                </span>
              )}
              {event.resolved && (
                <span className="rounded bg-health-healthy/10 text-health-healthy px-1.5 py-0.5 text-[10px] font-semibold">
                  RESOLVED
                </span>
              )}
            </div>
          </div>
        </div>
      ))}

      {displayed.length === 0 && (
        <p className="py-6 text-center text-sm text-muted-foreground">
          No anomaly events
        </p>
      )}
    </div>
  );
}
