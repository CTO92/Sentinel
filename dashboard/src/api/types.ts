// ─── Identifiers ───────────────────────────────────────────────

export interface GpuIdentifier {
  uuid: string;
  node_hostname: string;
  pci_bus_id: string;
  gpu_index: number;
  model: string;
  vbios_version: string;
  driver_version: string;
}

export interface SmIdentifier {
  gpu_uuid: string;
  sm_index: number;
}

// ─── Health States ─────────────────────────────────────────────

export enum GpuHealthState {
  HEALTHY = "HEALTHY",
  SUSPECT = "SUSPECT",
  QUARANTINED = "QUARANTINED",
  DEEP_TEST = "DEEP_TEST",
  CONDEMNED = "CONDEMNED",
}

export interface SmHealth {
  sm_id: SmIdentifier;
  state: GpuHealthState;
  failure_count: number;
  last_failure_time: string;
  error_pattern: string;
}

export interface GpuHealth {
  gpu_id: GpuIdentifier;
  state: GpuHealthState;
  reliability_score: number;
  bayesian_alpha: number;
  bayesian_beta: number;
  total_probes: number;
  failed_probes: number;
  last_probe_time: string;
  last_state_change: string;
  sm_health: SmHealth[];
  temperature_celsius: number;
  power_watts: number;
  memory_used_mb: number;
  memory_total_mb: number;
  gpu_utilization: number;
}

// ─── Fleet Summary ─────────────────────────────────────────────

export interface FleetHealthSummary {
  total_gpus: number;
  healthy: number;
  suspect: number;
  quarantined: number;
  deep_test: number;
  condemned: number;
  sdc_rate: number;
  average_reliability: number;
  total_probes_24h: number;
  anomalies_24h: number;
  gpu_health: GpuHealth[];
}

// ─── Probe Execution ───────────────────────────────────────────

export enum ProbeType {
  MATRIX_MULTIPLY = "MATRIX_MULTIPLY",
  REDUCTION_CHECK = "REDUCTION_CHECK",
  BIT_FLIP_SCAN = "BIT_FLIP_SCAN",
  MEMORY_PATTERN = "MEMORY_PATTERN",
  CROSS_GPU_COMPARE = "CROSS_GPU_COMPARE",
  SM_ISOLATION = "SM_ISOLATION",
}

export enum ProbeResult {
  PASS = "PASS",
  FAIL_SILENT_CORRUPTION = "FAIL_SILENT_CORRUPTION",
  FAIL_DETECTED_ERROR = "FAIL_DETECTED_ERROR",
  FAIL_TIMEOUT = "FAIL_TIMEOUT",
  FAIL_INFRASTRUCTURE = "FAIL_INFRASTRUCTURE",
}

export interface ProbeExecution {
  probe_id: string;
  gpu_id: GpuIdentifier;
  probe_type: ProbeType;
  result: ProbeResult;
  started_at: string;
  completed_at: string;
  duration_ms: number;
  max_relative_error: number;
  bit_flip_count: number;
  affected_sm_indices: number[];
  reference_gpu_uuid: string;
  error_details: string;
  metadata: Record<string, string>;
}

// ─── Anomalies ─────────────────────────────────────────────────

export enum AnomalySeverity {
  CRITICAL = "CRITICAL",
  HIGH = "HIGH",
  MEDIUM = "MEDIUM",
  LOW = "LOW",
  INFO = "INFO",
}

export interface AnomalyEvent {
  event_id: string;
  gpu_uuid: string;
  node_hostname: string;
  timestamp: string;
  severity: AnomalySeverity;
  anomaly_type: string;
  description: string;
  probe_id?: string;
  affected_sms: number[];
  correlation_id?: string;
  resolved: boolean;
}

// ─── Quarantine ────────────────────────────────────────────────

export enum QuarantineAction {
  QUARANTINE = "QUARANTINE",
  REINSTATE = "REINSTATE",
  DEEP_TEST = "DEEP_TEST",
  CONDEMN = "CONDEMN",
}

export interface QuarantineDirective {
  gpu_uuid: string;
  action: QuarantineAction;
  reason: string;
  initiated_by: string;
  evidence_ids: string[];
}

export interface QuarantineEntry {
  id: string;
  gpu_id: GpuIdentifier;
  state: GpuHealthState;
  entered_at: string;
  reason: string;
  evidence_ids: string[];
  initiated_by: string;
  reliability_score: number;
  anomaly_count: number;
  last_probe_result: ProbeResult;
}

// ─── Audit Ledger ──────────────────────────────────────────────

export enum AuditEntryType {
  PROBE_RESULT = "PROBE_RESULT",
  STATE_CHANGE = "STATE_CHANGE",
  QUARANTINE_ACTION = "QUARANTINE_ACTION",
  ANOMALY_DETECTED = "ANOMALY_DETECTED",
  CORRELATION_FOUND = "CORRELATION_FOUND",
  CONFIGURATION_CHANGE = "CONFIGURATION_CHANGE",
  SYSTEM_EVENT = "SYSTEM_EVENT",
}

export interface AuditEntry {
  entry_id: string;
  timestamp: string;
  entry_type: AuditEntryType;
  gpu_uuid?: string;
  node_hostname?: string;
  severity: AnomalySeverity;
  summary: string;
  details: string;
  chain_hash: string;
  previous_hash: string;
  chain_verified: boolean;
  actor: string;
}

export interface AuditFilters {
  gpu_uuid?: string;
  node_hostname?: string;
  entry_type?: AuditEntryType;
  severity?: AnomalySeverity;
  start_time?: string;
  end_time?: string;
  search?: string;
  page?: number;
  page_size?: number;
}

export interface PaginatedAuditResponse {
  entries: AuditEntry[];
  total_count: number;
  page: number;
  page_size: number;
  chain_integrity_verified: boolean;
}

// ─── Trust Graph ───────────────────────────────────────────────

export interface TrustEdge {
  source_gpu: string;
  target_gpu: string;
  trust_score: number;
  comparisons: number;
  last_compared: string;
  agreement_rate: number;
}

export interface TrustGraphData {
  nodes: Array<{
    id: string;
    label: string;
    state: GpuHealthState;
    reliability_score: number;
    hostname: string;
  }>;
  edges: TrustEdge[];
  coverage_percent: number;
  total_comparisons: number;
}

// ─── Correlation ───────────────────────────────────────────────

export interface CorrelationEvent {
  correlation_id: string;
  timestamp: string;
  affected_gpus: string[];
  affected_nodes: string[];
  correlation_type: string;
  description: string;
  severity: AnomalySeverity;
  root_cause_hypothesis: string;
  confidence: number;
}

// ─── Time Series ───────────────────────────────────────────────

export interface TimeSeriesPoint {
  timestamp: string;
  value: number;
}

export interface GpuHistoryData {
  gpu_uuid: string;
  reliability_scores: TimeSeriesPoint[];
  probe_results: ProbeExecution[];
  anomalies: AnomalyEvent[];
  temperature: TimeSeriesPoint[];
  power: TimeSeriesPoint[];
  utilization: TimeSeriesPoint[];
}

export type TimeRange = "1h" | "6h" | "24h" | "7d" | "30d";
