import { useNavigate } from "react-router-dom";
import { Cpu, Thermometer, Zap } from "lucide-react";
import { cn, healthStateColor, formatPercent } from "@/lib/utils";
import { HealthBadge } from "./HealthBadge";
import type { GpuHealth } from "@/api/types";

interface GpuCardProps {
  gpu: GpuHealth;
  compact?: boolean;
}

export function GpuCard({ gpu, compact = false }: GpuCardProps) {
  const navigate = useNavigate();

  return (
    <button
      onClick={() => navigate(`/gpu/${gpu.gpu_id.uuid}`)}
      className={cn(
        "w-full text-left rounded-lg border border-border bg-card p-3 transition-all hover:shadow-md hover:border-primary/40 focus:outline-none focus:ring-2 focus:ring-ring",
        compact && "p-2",
      )}
    >
      <div className="flex items-start justify-between gap-2">
        <div className="min-w-0 flex-1">
          <p className="truncate text-xs font-mono text-muted-foreground">
            {gpu.gpu_id.uuid.slice(0, 12)}...
          </p>
          <p className="truncate text-xs text-muted-foreground mt-0.5">
            {gpu.gpu_id.node_hostname}
          </p>
        </div>
        <HealthBadge state={gpu.state} size="sm" />
      </div>

      {!compact && (
        <>
          <div className="mt-3 flex items-center gap-3 text-xs text-muted-foreground">
            <span className="flex items-center gap-1">
              <Cpu className="h-3 w-3" />
              {formatPercent(gpu.reliability_score)}
            </span>
            <span className="flex items-center gap-1">
              <Thermometer className="h-3 w-3" />
              {gpu.temperature_celsius}°C
            </span>
            <span className="flex items-center gap-1">
              <Zap className="h-3 w-3" />
              {gpu.power_watts}W
            </span>
          </div>

          <div className="mt-2">
            <div className="flex items-center justify-between text-[10px] text-muted-foreground mb-1">
              <span>Reliability</span>
              <span>{formatPercent(gpu.reliability_score)}</span>
            </div>
            <div className="h-1.5 w-full rounded-full bg-muted overflow-hidden">
              <div
                className="h-full rounded-full transition-all duration-500"
                style={{
                  width: `${gpu.reliability_score * 100}%`,
                  backgroundColor: healthStateColor(gpu.state),
                }}
              />
            </div>
          </div>

          <div className="mt-2 flex items-center justify-between text-[10px] text-muted-foreground">
            <span>
              {gpu.total_probes - gpu.failed_probes}/{gpu.total_probes} probes
              passed
            </span>
            <span>{gpu.gpu_id.model}</span>
          </div>
        </>
      )}
    </button>
  );
}
