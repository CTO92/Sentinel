import { type ClassValue, clsx } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

export function formatPercent(value: number, decimals = 2): string {
  return `${(value * 100).toFixed(decimals)}%`;
}

export function formatNumber(value: number, decimals = 2): string {
  if (value >= 1_000_000) {
    return `${(value / 1_000_000).toFixed(decimals)}M`;
  }
  if (value >= 1_000) {
    return `${(value / 1_000).toFixed(decimals)}K`;
  }
  return value.toFixed(decimals);
}

export function formatDuration(ms: number): string {
  const seconds = Math.floor(ms / 1000);
  if (seconds < 60) return `${seconds}s`;
  const minutes = Math.floor(seconds / 60);
  if (minutes < 60) return `${minutes}m ${seconds % 60}s`;
  const hours = Math.floor(minutes / 60);
  if (hours < 24) return `${hours}h ${minutes % 60}m`;
  const days = Math.floor(hours / 24);
  return `${days}d ${hours % 24}h`;
}

export function healthStateColor(state: string): string {
  switch (state) {
    case "HEALTHY":
      return "#22c55e";
    case "SUSPECT":
      return "#eab308";
    case "QUARANTINED":
      return "#f97316";
    case "DEEP_TEST":
      return "#8b5cf6";
    case "CONDEMNED":
      return "#ef4444";
    default:
      return "#6b7280";
  }
}

export function healthStateBg(state: string): string {
  switch (state) {
    case "HEALTHY":
      return "bg-health-healthy/10 text-health-healthy border-health-healthy/30";
    case "SUSPECT":
      return "bg-health-suspect/10 text-health-suspect border-health-suspect/30";
    case "QUARANTINED":
      return "bg-health-quarantined/10 text-health-quarantined border-health-quarantined/30";
    case "DEEP_TEST":
      return "bg-health-deep-test/10 text-health-deep-test border-health-deep-test/30";
    case "CONDEMNED":
      return "bg-health-condemned/10 text-health-condemned border-health-condemned/30";
    default:
      return "bg-muted text-muted-foreground border-muted";
  }
}

export function severityColor(severity: string): string {
  switch (severity) {
    case "CRITICAL":
      return "text-red-500";
    case "HIGH":
      return "text-orange-500";
    case "MEDIUM":
      return "text-yellow-500";
    case "LOW":
      return "text-blue-500";
    case "INFO":
      return "text-gray-400";
    default:
      return "text-gray-400";
  }
}
