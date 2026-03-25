import { cn, healthStateBg } from "@/lib/utils";
import type { GpuHealthState } from "@/api/types";

interface HealthBadgeProps {
  state: GpuHealthState | string;
  size?: "sm" | "md" | "lg";
  className?: string;
}

const sizeClasses = {
  sm: "px-1.5 py-0.5 text-[10px]",
  md: "px-2 py-0.5 text-xs",
  lg: "px-3 py-1 text-sm",
};

export function HealthBadge({
  state,
  size = "md",
  className,
}: HealthBadgeProps) {
  const label = state.replace(/_/g, " ");

  return (
    <span
      className={cn(
        "inline-flex items-center rounded-full border font-semibold uppercase tracking-wider",
        healthStateBg(state),
        sizeClasses[size],
        className,
      )}
    >
      <span
        className={cn("mr-1.5 h-1.5 w-1.5 rounded-full", {
          "bg-health-healthy": state === "HEALTHY",
          "bg-health-suspect animate-pulse": state === "SUSPECT",
          "bg-health-quarantined animate-pulse": state === "QUARANTINED",
          "bg-health-deep-test animate-pulse-slow": state === "DEEP_TEST",
          "bg-health-condemned": state === "CONDEMNED",
        })}
      />
      {label}
    </span>
  );
}
