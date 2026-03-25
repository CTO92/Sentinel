import type {
  FleetHealthSummary,
  GpuHealth,
  GpuHistoryData,
  PaginatedAuditResponse,
  AuditFilters,
  QuarantineEntry,
  QuarantineDirective,
  TrustGraphData,
  CorrelationEvent,
  TimeRange,
} from "./types";

class ApiError extends Error {
  constructor(
    public status: number,
    public statusText: string,
    public body?: unknown,
  ) {
    super(`API Error ${status}: ${statusText}`);
    this.name = "ApiError";
  }
}

class SentinelApiClient {
  private baseUrl: string;
  private authToken: string | null = null;

  constructor(baseUrl?: string) {
    this.baseUrl = baseUrl ?? import.meta.env.VITE_API_BASE_URL ?? "/api";
  }

  setAuthToken(token: string) {
    this.authToken = token;
  }

  private async request<T>(
    path: string,
    options: RequestInit = {},
  ): Promise<T> {
    const headers: Record<string, string> = {
      "Content-Type": "application/json",
      ...(options.headers as Record<string, string>),
    };

    if (this.authToken) {
      headers["Authorization"] = `Bearer ${this.authToken}`;
    }

    const response = await fetch(`${this.baseUrl}${path}`, {
      ...options,
      headers,
    });

    if (!response.ok) {
      let body: unknown;
      try {
        body = await response.json();
      } catch {
        // Body not parseable
      }
      throw new ApiError(response.status, response.statusText, body);
    }

    return response.json() as Promise<T>;
  }

  // ─── Fleet Health ──────────────────────────────────────────

  async getFleetHealth(): Promise<FleetHealthSummary> {
    return this.request<FleetHealthSummary>("/v1/fleet/health");
  }

  // ─── GPU Health ────────────────────────────────────────────

  async getGpuHealth(uuid: string): Promise<GpuHealth> {
    return this.request<GpuHealth>(`/v1/gpu/${uuid}/health`);
  }

  async getGpuHistory(
    uuid: string,
    timeRange: TimeRange,
  ): Promise<GpuHistoryData> {
    return this.request<GpuHistoryData>(
      `/v1/gpu/${uuid}/history?range=${timeRange}`,
    );
  }

  // ─── Audit Trail ───────────────────────────────────────────

  async getAuditTrail(filters: AuditFilters): Promise<PaginatedAuditResponse> {
    const params = new URLSearchParams();
    if (filters.gpu_uuid) params.set("gpu_uuid", filters.gpu_uuid);
    if (filters.node_hostname)
      params.set("node_hostname", filters.node_hostname);
    if (filters.entry_type) params.set("entry_type", filters.entry_type);
    if (filters.severity) params.set("severity", filters.severity);
    if (filters.start_time) params.set("start_time", filters.start_time);
    if (filters.end_time) params.set("end_time", filters.end_time);
    if (filters.search) params.set("search", filters.search);
    if (filters.page !== undefined)
      params.set("page", String(filters.page));
    if (filters.page_size !== undefined)
      params.set("page_size", String(filters.page_size));

    return this.request<PaginatedAuditResponse>(
      `/v1/audit?${params.toString()}`,
    );
  }

  // ─── Quarantine Queue ──────────────────────────────────────

  async getQuarantineQueue(): Promise<QuarantineEntry[]> {
    return this.request<QuarantineEntry[]>("/v1/quarantine/queue");
  }

  async issueQuarantine(directive: QuarantineDirective): Promise<void> {
    await this.request("/v1/quarantine/action", {
      method: "POST",
      body: JSON.stringify(directive),
    });
  }

  // ─── Trust Graph ───────────────────────────────────────────

  async getTrustGraph(): Promise<TrustGraphData> {
    return this.request<TrustGraphData>("/v1/trust/graph");
  }

  // ─── Correlations ──────────────────────────────────────────

  async getCorrelations(timeRange: TimeRange): Promise<CorrelationEvent[]> {
    return this.request<CorrelationEvent[]>(
      `/v1/correlations?range=${timeRange}`,
    );
  }
}

export const apiClient = new SentinelApiClient();
export { ApiError };
