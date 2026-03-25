import { useNavigate } from "react-router-dom";
import { cn, healthStateColor } from "@/lib/utils";
import type { GpuHealth } from "@/api/types";

interface HeatMapProps {
  gpus: GpuHealth[];
  className?: string;
}

export function HeatMap({ gpus, className }: HeatMapProps) {
  const navigate = useNavigate();

  return (
    <div className={cn("w-full", className)}>
      <div className="grid grid-cols-8 sm:grid-cols-12 md:grid-cols-16 lg:grid-cols-20 xl:grid-cols-24 gap-1">
        {gpus.map((gpu) => (
          <button
            key={gpu.gpu_id.uuid}
            onClick={() => navigate(`/gpu/${gpu.gpu_id.uuid}`)}
            className="group relative aspect-square rounded-sm transition-transform hover:scale-150 hover:z-10 focus:outline-none focus:ring-2 focus:ring-ring"
            style={{ backgroundColor: healthStateColor(gpu.state) }}
            title={`${gpu.gpu_id.uuid.slice(0, 8)} | ${gpu.gpu_id.node_hostname} | ${gpu.state} | ${(gpu.reliability_score * 100).toFixed(1)}%`}
          >
            <div className="absolute bottom-full left-1/2 -translate-x-1/2 mb-1 hidden group-hover:block z-20">
              <div className="bg-popover text-popover-foreground rounded px-2 py-1 text-[10px] whitespace-nowrap shadow-lg border border-border">
                <p className="font-mono">{gpu.gpu_id.uuid.slice(0, 12)}</p>
                <p>{gpu.gpu_id.node_hostname}</p>
                <p className="font-semibold">{gpu.state}</p>
              </div>
            </div>
          </button>
        ))}
      </div>

      <div className="mt-3 flex items-center gap-4 text-xs text-muted-foreground">
        <span className="flex items-center gap-1.5">
          <span className="h-2.5 w-2.5 rounded-sm bg-health-healthy" />
          Healthy
        </span>
        <span className="flex items-center gap-1.5">
          <span className="h-2.5 w-2.5 rounded-sm bg-health-suspect" />
          Suspect
        </span>
        <span className="flex items-center gap-1.5">
          <span className="h-2.5 w-2.5 rounded-sm bg-health-quarantined" />
          Quarantined
        </span>
        <span className="flex items-center gap-1.5">
          <span className="h-2.5 w-2.5 rounded-sm bg-health-deep-test" />
          Deep Test
        </span>
        <span className="flex items-center gap-1.5">
          <span className="h-2.5 w-2.5 rounded-sm bg-health-condemned" />
          Condemned
        </span>
      </div>
    </div>
  );
}
