import {
  useQuery,
  useMutation,
  useQueryClient,
} from "@tanstack/react-query";
import { apiClient } from "./client";
import type {
  AuditFilters,
  QuarantineDirective,
  TimeRange,
} from "./types";

// ─── Query Keys ────────────────────────────────────────────────

export const queryKeys = {
  fleetHealth: ["fleet", "health"] as const,
  gpuHealth: (uuid: string) => ["gpu", uuid, "health"] as const,
  gpuHistory: (uuid: string, range: TimeRange) =>
    ["gpu", uuid, "history", range] as const,
  auditTrail: (filters: AuditFilters) => ["audit", filters] as const,
  quarantineQueue: ["quarantine", "queue"] as const,
  trustGraph: ["trust", "graph"] as const,
  correlations: (range: TimeRange) => ["correlations", range] as const,
};

// ─── Fleet Health ──────────────────────────────────────────────

export function useFleetHealth(refetchInterval = 10_000) {
  return useQuery({
    queryKey: queryKeys.fleetHealth,
    queryFn: () => apiClient.getFleetHealth(),
    refetchInterval,
    staleTime: 5_000,
  });
}

// ─── GPU Health ────────────────────────────────────────────────

export function useGpuHealth(uuid: string) {
  return useQuery({
    queryKey: queryKeys.gpuHealth(uuid),
    queryFn: () => apiClient.getGpuHealth(uuid),
    enabled: !!uuid,
    refetchInterval: 10_000,
    staleTime: 5_000,
  });
}

export function useGpuHistory(uuid: string, timeRange: TimeRange) {
  return useQuery({
    queryKey: queryKeys.gpuHistory(uuid, timeRange),
    queryFn: () => apiClient.getGpuHistory(uuid, timeRange),
    enabled: !!uuid,
    staleTime: 30_000,
  });
}

// ─── Audit Trail ───────────────────────────────────────────────

export function useAuditTrail(filters: AuditFilters) {
  return useQuery({
    queryKey: queryKeys.auditTrail(filters),
    queryFn: () => apiClient.getAuditTrail(filters),
    staleTime: 10_000,
  });
}

// ─── Quarantine Queue ──────────────────────────────────────────

export function useQuarantineQueue() {
  return useQuery({
    queryKey: queryKeys.quarantineQueue,
    queryFn: () => apiClient.getQuarantineQueue(),
    refetchInterval: 15_000,
    staleTime: 5_000,
  });
}

export function useQuarantineMutation() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (directive: QuarantineDirective) =>
      apiClient.issueQuarantine(directive),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: queryKeys.quarantineQueue });
      queryClient.invalidateQueries({ queryKey: queryKeys.fleetHealth });
    },
  });
}

// ─── Trust Graph ───────────────────────────────────────────────

export function useTrustGraph() {
  return useQuery({
    queryKey: queryKeys.trustGraph,
    queryFn: () => apiClient.getTrustGraph(),
    refetchInterval: 30_000,
    staleTime: 15_000,
  });
}

// ─── Correlations ──────────────────────────────────────────────

export function useCorrelations(timeRange: TimeRange) {
  return useQuery({
    queryKey: queryKeys.correlations(timeRange),
    queryFn: () => apiClient.getCorrelations(timeRange),
    staleTime: 30_000,
  });
}
