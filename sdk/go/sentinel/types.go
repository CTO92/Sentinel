// Package sentinel provides Go types and client for the SENTINEL Silent Data
// Corruption detection framework.
package sentinel

import (
	"time"
)

// ---------------------------------------------------------------------------
// common.proto
// ---------------------------------------------------------------------------

// Severity levels for events, anomalies, and alerts.
type Severity int32

const (
	SeverityUnspecified Severity = 0
	SeverityInfo        Severity = 1
	SeverityWarning     Severity = 2
	SeverityHigh        Severity = 3
	SeverityCritical    Severity = 4
)

var severityNames = map[Severity]string{
	SeverityUnspecified: "SEVERITY_UNSPECIFIED",
	SeverityInfo:        "INFO",
	SeverityWarning:     "WARNING",
	SeverityHigh:        "HIGH",
	SeverityCritical:    "CRITICAL",
}

func (s Severity) String() string {
	if name, ok := severityNames[s]; ok {
		return name
	}
	return "SEVERITY_UNSPECIFIED"
}

// GpuIdentifier uniquely identifies a GPU within the fleet.
type GpuIdentifier struct {
	UUID            string `json:"uuid"`
	Hostname        string `json:"hostname"`
	DeviceIndex     uint32 `json:"device_index"`
	Model           string `json:"model"`
	DriverVersion   string `json:"driver_version"`
	FirmwareVersion string `json:"firmware_version"`
}

// SmIdentifier identifies a specific Streaming Multiprocessor on a GPU.
type SmIdentifier struct {
	GPU  *GpuIdentifier `json:"gpu,omitempty"`
	SmID uint32         `json:"sm_id"`
}

// ---------------------------------------------------------------------------
// probe.proto
// ---------------------------------------------------------------------------

// ProbeType enumerates hardware integrity probe types.
type ProbeType int32

const (
	ProbeTypeUnspecified  ProbeType = 0
	ProbeTypeFMA         ProbeType = 1
	ProbeTypeTensorCore  ProbeType = 2
	ProbeTypeTranscendental ProbeType = 3
	ProbeTypeAES         ProbeType = 4
	ProbeTypeMemory      ProbeType = 5
	ProbeTypeRegisterFile ProbeType = 6
	ProbeTypeSharedMemory ProbeType = 7
)

var probeTypeNames = map[ProbeType]string{
	ProbeTypeUnspecified:    "PROBE_TYPE_UNSPECIFIED",
	ProbeTypeFMA:            "PROBE_TYPE_FMA",
	ProbeTypeTensorCore:     "PROBE_TYPE_TENSOR_CORE",
	ProbeTypeTranscendental: "PROBE_TYPE_TRANSCENDENTAL",
	ProbeTypeAES:            "PROBE_TYPE_AES",
	ProbeTypeMemory:         "PROBE_TYPE_MEMORY",
	ProbeTypeRegisterFile:   "PROBE_TYPE_REGISTER_FILE",
	ProbeTypeSharedMemory:   "PROBE_TYPE_SHARED_MEMORY",
}

func (p ProbeType) String() string {
	if name, ok := probeTypeNames[p]; ok {
		return name
	}
	return "PROBE_TYPE_UNSPECIFIED"
}

// ProbeResult is the result of a single probe execution.
type ProbeResult int32

const (
	ProbeResultUnspecified ProbeResult = 0
	ProbeResultPass       ProbeResult = 1
	ProbeResultFail       ProbeResult = 2
	ProbeResultError      ProbeResult = 3
	ProbeResultTimeout    ProbeResult = 4
)

var probeResultNames = map[ProbeResult]string{
	ProbeResultUnspecified: "PROBE_RESULT_UNSPECIFIED",
	ProbeResultPass:        "PROBE_RESULT_PASS",
	ProbeResultFail:        "PROBE_RESULT_FAIL",
	ProbeResultError:       "PROBE_RESULT_ERROR",
	ProbeResultTimeout:     "PROBE_RESULT_TIMEOUT",
}

func (p ProbeResult) String() string {
	if name, ok := probeResultNames[p]; ok {
		return name
	}
	return "PROBE_RESULT_UNSPECIFIED"
}

// MismatchDetail contains bit-level mismatch information for failed probes.
type MismatchDetail struct {
	ByteOffset    uint64 `json:"byte_offset"`
	ExpectedValue []byte `json:"expected_value"`
	ActualValue   []byte `json:"actual_value"`
	DifferingBits []uint32 `json:"differing_bits"`
}

// ProbeExecution represents a single probe execution with its result.
type ProbeExecution struct {
	ExecutionID     string          `json:"execution_id"`
	ProbeType       ProbeType       `json:"probe_type"`
	SM              *SmIdentifier   `json:"sm,omitempty"`
	Result          ProbeResult     `json:"result"`
	ExpectedHash    []byte          `json:"expected_hash"`
	ActualHash      []byte          `json:"actual_hash"`
	MismatchDetail  *MismatchDetail `json:"mismatch_detail,omitempty"`
	ExecutionTimeNs uint64          `json:"execution_time_ns"`
	GpuClockMhz     uint32          `json:"gpu_clock_mhz"`
	GpuTemperatureC float32         `json:"gpu_temperature_c"`
	GpuPowerW       float32         `json:"gpu_power_w"`
	Timestamp       *time.Time      `json:"timestamp,omitempty"`
	HmacSignature   []byte          `json:"hmac_signature"`
}

// ProbeScheduleOverride is an override directive for a probe type schedule.
type ProbeScheduleOverride struct {
	ProbeType       ProbeType `json:"probe_type"`
	PeriodSeconds   uint32    `json:"period_seconds"`
	DurationSeconds uint32    `json:"duration_seconds"`
}

// ---------------------------------------------------------------------------
// anomaly.proto
// ---------------------------------------------------------------------------

// AnomalyType classifies anomaly types detected by monitoring subsystems.
type AnomalyType int32

const (
	AnomalyTypeUnspecified         AnomalyType = 0
	AnomalyTypeLogitDrift          AnomalyType = 1
	AnomalyTypeEntropyAnomaly      AnomalyType = 2
	AnomalyTypeKLDivergence        AnomalyType = 3
	AnomalyTypeGradientNormSpike   AnomalyType = 4
	AnomalyTypeLossSpike           AnomalyType = 5
	AnomalyTypeCrossRankDivergence AnomalyType = 6
	AnomalyTypeCheckpointDivergence AnomalyType = 7
	AnomalyTypeInvariantViolation  AnomalyType = 8
)

var anomalyTypeNames = map[AnomalyType]string{
	AnomalyTypeUnspecified:          "ANOMALY_TYPE_UNSPECIFIED",
	AnomalyTypeLogitDrift:           "ANOMALY_TYPE_LOGIT_DRIFT",
	AnomalyTypeEntropyAnomaly:       "ANOMALY_TYPE_ENTROPY_ANOMALY",
	AnomalyTypeKLDivergence:         "ANOMALY_TYPE_KL_DIVERGENCE",
	AnomalyTypeGradientNormSpike:    "ANOMALY_TYPE_GRADIENT_NORM_SPIKE",
	AnomalyTypeLossSpike:            "ANOMALY_TYPE_LOSS_SPIKE",
	AnomalyTypeCrossRankDivergence:  "ANOMALY_TYPE_CROSS_RANK_DIVERGENCE",
	AnomalyTypeCheckpointDivergence: "ANOMALY_TYPE_CHECKPOINT_DIVERGENCE",
	AnomalyTypeInvariantViolation:   "ANOMALY_TYPE_INVARIANT_VIOLATION",
}

func (a AnomalyType) String() string {
	if name, ok := anomalyTypeNames[a]; ok {
		return name
	}
	return "ANOMALY_TYPE_UNSPECIFIED"
}

// AnomalySource identifies the subsystem that detected an anomaly.
type AnomalySource int32

const (
	AnomalySourceUnspecified      AnomalySource = 0
	AnomalySourceInferenceMonitor AnomalySource = 1
	AnomalySourceTrainingMonitor  AnomalySource = 2
	AnomalySourceInvariantChecker AnomalySource = 3
)

var anomalySourceNames = map[AnomalySource]string{
	AnomalySourceUnspecified:      "ANOMALY_SOURCE_UNSPECIFIED",
	AnomalySourceInferenceMonitor: "ANOMALY_SOURCE_INFERENCE_MONITOR",
	AnomalySourceTrainingMonitor:  "ANOMALY_SOURCE_TRAINING_MONITOR",
	AnomalySourceInvariantChecker: "ANOMALY_SOURCE_INVARIANT_CHECKER",
}

func (a AnomalySource) String() string {
	if name, ok := anomalySourceNames[a]; ok {
		return name
	}
	return "ANOMALY_SOURCE_UNSPECIFIED"
}

// AnomalyEvent represents a single detected anomaly event.
type AnomalyEvent struct {
	EventID           string            `json:"event_id"`
	AnomalyType       AnomalyType       `json:"anomaly_type"`
	Source            AnomalySource     `json:"source"`
	GPU               *GpuIdentifier    `json:"gpu,omitempty"`
	Severity          Severity          `json:"severity"`
	Score             float32           `json:"score"`
	Threshold         float32           `json:"threshold"`
	Details           string            `json:"details"`
	TensorFingerprint []byte            `json:"tensor_fingerprint"`
	Timestamp         *time.Time        `json:"timestamp,omitempty"`
	Metadata          map[string]string `json:"metadata,omitempty"`
	LayerName         string            `json:"layer_name"`
	ModelID           string            `json:"model_id"`
	StepNumber        uint64            `json:"step_number"`
}

// ---------------------------------------------------------------------------
// health.proto
// ---------------------------------------------------------------------------

// GpuHealthState represents lifecycle states for a GPU in the fleet.
type GpuHealthState int32

const (
	GpuHealthStateUnspecified GpuHealthState = 0
	GpuHealthStateHealthy    GpuHealthState = 1
	GpuHealthStateSuspect    GpuHealthState = 2
	GpuHealthStateQuarantined GpuHealthState = 3
	GpuHealthStateDeepTest   GpuHealthState = 4
	GpuHealthStateCondemned  GpuHealthState = 5
)

var gpuHealthStateNames = map[GpuHealthState]string{
	GpuHealthStateUnspecified:  "GPU_HEALTH_STATE_UNSPECIFIED",
	GpuHealthStateHealthy:     "GPU_HEALTH_STATE_HEALTHY",
	GpuHealthStateSuspect:     "GPU_HEALTH_STATE_SUSPECT",
	GpuHealthStateQuarantined: "GPU_HEALTH_STATE_QUARANTINED",
	GpuHealthStateDeepTest:    "GPU_HEALTH_STATE_DEEP_TEST",
	GpuHealthStateCondemned:   "GPU_HEALTH_STATE_CONDEMNED",
}

func (g GpuHealthState) String() string {
	if name, ok := gpuHealthStateNames[g]; ok {
		return name
	}
	return "GPU_HEALTH_STATE_UNSPECIFIED"
}

// SmHealth is the health status for a single Streaming Multiprocessor.
type SmHealth struct {
	SM             *SmIdentifier `json:"sm,omitempty"`
	ReliabilityScore float64    `json:"reliability_score"`
	ProbePassCount uint64       `json:"probe_pass_count"`
	ProbeFailCount uint64       `json:"probe_fail_count"`
	Disabled       bool         `json:"disabled"`
	DisableReason  string       `json:"disable_reason"`
}

// GpuHealth is the health status and Bayesian reliability model for a single GPU.
type GpuHealth struct {
	GPU               *GpuIdentifier `json:"gpu,omitempty"`
	State             GpuHealthState `json:"state"`
	ReliabilityScore  float64        `json:"reliability_score"`
	Alpha             float64        `json:"alpha"`
	Beta              float64        `json:"beta"`
	LastProbeTime     *time.Time     `json:"last_probe_time,omitempty"`
	LastAnomalyTime   *time.Time     `json:"last_anomaly_time,omitempty"`
	ProbePassCount    uint64         `json:"probe_pass_count"`
	ProbeFailCount    uint64         `json:"probe_fail_count"`
	AnomalyCount      uint64         `json:"anomaly_count"`
	StateChangedAt    *time.Time     `json:"state_changed_at,omitempty"`
	StateChangeReason string         `json:"state_change_reason"`
	SmHealth          []*SmHealth    `json:"sm_health,omitempty"`
	AnomalyRate       float64        `json:"anomaly_rate"`
	ProbeFailureRate  float64        `json:"probe_failure_rate"`
}

// FleetHealthSummary is an aggregated health summary for the entire GPU fleet.
type FleetHealthSummary struct {
	TotalGPUs             uint32     `json:"total_gpus"`
	Healthy               uint32     `json:"healthy"`
	Suspect               uint32     `json:"suspect"`
	Quarantined           uint32     `json:"quarantined"`
	DeepTest              uint32     `json:"deep_test"`
	Condemned             uint32     `json:"condemned"`
	OverallSDCRate        float64    `json:"overall_sdc_rate"`
	AverageReliabilityScore float64  `json:"average_reliability_score"`
	SnapshotTime          *time.Time `json:"snapshot_time,omitempty"`
	ActiveAgents          uint32     `json:"active_agents"`
	RateWindowSeconds     uint32     `json:"rate_window_seconds"`
}

// ---------------------------------------------------------------------------
// trust.proto
// ---------------------------------------------------------------------------

// TrustEdge is an edge in the GPU trust graph representing pairwise comparison history.
type TrustEdge struct {
	GpuA              *GpuIdentifier `json:"gpu_a,omitempty"`
	GpuB              *GpuIdentifier `json:"gpu_b,omitempty"`
	AgreementCount    uint64         `json:"agreement_count"`
	DisagreementCount uint64         `json:"disagreement_count"`
	LastComparison    *time.Time     `json:"last_comparison,omitempty"`
	TrustScore        float64        `json:"trust_score"`
}

// TrustGraphSnapshot is a point-in-time snapshot of the entire trust graph.
type TrustGraphSnapshot struct {
	Edges         []*TrustEdge `json:"edges,omitempty"`
	Timestamp     *time.Time   `json:"timestamp,omitempty"`
	CoveragePct   float64      `json:"coverage_pct"`
	TotalGPUs     uint32       `json:"total_gpus"`
	MinTrustScore float64      `json:"min_trust_score"`
	MeanTrustScore float64     `json:"mean_trust_score"`
}

// TmrGpuResult is the result of a single GPU's contribution to a TMR canary.
type TmrGpuResult struct {
	GPU               *GpuIdentifier `json:"gpu,omitempty"`
	OutputFingerprint []byte         `json:"output_fingerprint"`
	ExecutionTimeNs   uint64         `json:"execution_time_ns"`
	Success           bool           `json:"success"`
	ErrorMessage      string         `json:"error_message"`
}

// TmrResult is the aggregated result of a TMR canary run.
type TmrResult struct {
	CanaryID      string          `json:"canary_id"`
	Results       []*TmrGpuResult `json:"results,omitempty"`
	ConsensusHash []byte          `json:"consensus_hash"`
	DissentingGPU *GpuIdentifier  `json:"dissenting_gpu,omitempty"`
	Unanimous     bool            `json:"unanimous"`
	Timestamp     *time.Time      `json:"timestamp,omitempty"`
}

// ---------------------------------------------------------------------------
// quarantine.proto
// ---------------------------------------------------------------------------

// QuarantineAction enumerates actions that can be taken on a GPU's lifecycle.
type QuarantineAction int32

const (
	QuarantineActionUnspecified     QuarantineAction = 0
	QuarantineActionQuarantine     QuarantineAction = 1
	QuarantineActionReinstate      QuarantineAction = 2
	QuarantineActionCondemn        QuarantineAction = 3
	QuarantineActionScheduleDeepTest QuarantineAction = 4
)

var quarantineActionNames = map[QuarantineAction]string{
	QuarantineActionUnspecified:      "QUARANTINE_ACTION_UNSPECIFIED",
	QuarantineActionQuarantine:       "QUARANTINE_ACTION_QUARANTINE",
	QuarantineActionReinstate:        "QUARANTINE_ACTION_REINSTATE",
	QuarantineActionCondemn:          "QUARANTINE_ACTION_CONDEMN",
	QuarantineActionScheduleDeepTest: "QUARANTINE_ACTION_SCHEDULE_DEEP_TEST",
}

func (q QuarantineAction) String() string {
	if name, ok := quarantineActionNames[q]; ok {
		return name
	}
	return "QUARANTINE_ACTION_UNSPECIFIED"
}

// ApprovalStatus tracks approval for directives requiring human sign-off.
type ApprovalStatus struct {
	Approved   bool       `json:"approved"`
	Reviewer   string     `json:"reviewer"`
	ReviewTime *time.Time `json:"review_time,omitempty"`
	Comment    string     `json:"comment"`
}

// QuarantineDirective is a directive to change a GPU's lifecycle state.
type QuarantineDirective struct {
	DirectiveID      string           `json:"directive_id"`
	GPU              *GpuIdentifier   `json:"gpu,omitempty"`
	Action           QuarantineAction `json:"action"`
	Reason           string           `json:"reason"`
	InitiatedBy      string           `json:"initiated_by"`
	Evidence         []string         `json:"evidence,omitempty"`
	Timestamp        *time.Time       `json:"timestamp,omitempty"`
	Priority         uint32           `json:"priority"`
	RequiresApproval bool             `json:"requires_approval"`
	Approval         *ApprovalStatus  `json:"approval,omitempty"`
}

// DirectiveResponse is the response to a directive issuance.
type DirectiveResponse struct {
	DirectiveID     string `json:"directive_id"`
	Accepted        bool   `json:"accepted"`
	RejectionReason string `json:"rejection_reason"`
	ResultingState  string `json:"resulting_state"`
}

// ---------------------------------------------------------------------------
// correlation.proto
// ---------------------------------------------------------------------------

// PatternType enumerates correlation patterns detected across events.
type PatternType int32

const (
	PatternTypeUnspecified       PatternType = 0
	PatternTypeMultiSignal       PatternType = 1
	PatternTypeSMLocalized       PatternType = 2
	PatternTypeEnvironmental     PatternType = 3
	PatternTypeNodeCorrelated    PatternType = 4
	PatternTypeFirmwareCorrelated PatternType = 5
	PatternTypeTMRConfirmed      PatternType = 6
)

var patternTypeNames = map[PatternType]string{
	PatternTypeUnspecified:        "PATTERN_TYPE_UNSPECIFIED",
	PatternTypeMultiSignal:        "PATTERN_TYPE_MULTI_SIGNAL",
	PatternTypeSMLocalized:        "PATTERN_TYPE_SM_LOCALIZED",
	PatternTypeEnvironmental:      "PATTERN_TYPE_ENVIRONMENTAL",
	PatternTypeNodeCorrelated:     "PATTERN_TYPE_NODE_CORRELATED",
	PatternTypeFirmwareCorrelated: "PATTERN_TYPE_FIRMWARE_CORRELATED",
	PatternTypeTMRConfirmed:       "PATTERN_TYPE_TMR_CONFIRMED",
}

func (p PatternType) String() string {
	if name, ok := patternTypeNames[p]; ok {
		return name
	}
	return "PATTERN_TYPE_UNSPECIFIED"
}

// CorrelationEvent links multiple raw events into a higher-level finding.
type CorrelationEvent struct {
	EventID           string         `json:"event_id"`
	EventsCorrelated  []string       `json:"events_correlated,omitempty"`
	PatternType       PatternType    `json:"pattern_type"`
	Confidence        float64        `json:"confidence"`
	AttributedGPU     *GpuIdentifier `json:"attributed_gpu,omitempty"`
	AttributedSM      *SmIdentifier  `json:"attributed_sm,omitempty"`
	Description       string         `json:"description"`
	Timestamp         *time.Time     `json:"timestamp,omitempty"`
	Severity          Severity       `json:"severity"`
	RecommendedAction string         `json:"recommended_action"`
}

// StateTransition is a recorded GPU state transition.
type StateTransition struct {
	FromState   GpuHealthState `json:"from_state"`
	ToState     GpuHealthState `json:"to_state"`
	Timestamp   *time.Time     `json:"timestamp,omitempty"`
	Reason      string         `json:"reason"`
	InitiatedBy string         `json:"initiated_by"`
}

// ReliabilitySample is a point-in-time reliability score sample.
type ReliabilitySample struct {
	Timestamp        *time.Time `json:"timestamp,omitempty"`
	ReliabilityScore float64    `json:"reliability_score"`
	Alpha            float64    `json:"alpha"`
	Beta             float64    `json:"beta"`
}

// GpuHistoryResponse contains historical health data for a GPU.
type GpuHistoryResponse struct {
	StateTransitions   []*StateTransition   `json:"state_transitions,omitempty"`
	Correlations       []*CorrelationEvent  `json:"correlations,omitempty"`
	ReliabilityHistory []*ReliabilitySample `json:"reliability_history,omitempty"`
	NextPageToken      string               `json:"next_page_token"`
}

// ---------------------------------------------------------------------------
// audit.proto
// ---------------------------------------------------------------------------

// AuditEntryType enumerates entry types in the audit ledger.
type AuditEntryType int32

const (
	AuditEntryTypeUnspecified      AuditEntryType = 0
	AuditEntryTypeProbeResult      AuditEntryType = 1
	AuditEntryTypeAnomalyEvent     AuditEntryType = 2
	AuditEntryTypeQuarantineAction AuditEntryType = 3
	AuditEntryTypeConfigChange     AuditEntryType = 4
	AuditEntryTypeTMRResult        AuditEntryType = 5
	AuditEntryTypeSystemEvent      AuditEntryType = 6
)

var auditEntryTypeNames = map[AuditEntryType]string{
	AuditEntryTypeUnspecified:      "AUDIT_ENTRY_TYPE_UNSPECIFIED",
	AuditEntryTypeProbeResult:      "AUDIT_ENTRY_TYPE_PROBE_RESULT",
	AuditEntryTypeAnomalyEvent:     "AUDIT_ENTRY_TYPE_ANOMALY_EVENT",
	AuditEntryTypeQuarantineAction: "AUDIT_ENTRY_TYPE_QUARANTINE_ACTION",
	AuditEntryTypeConfigChange:     "AUDIT_ENTRY_TYPE_CONFIG_CHANGE",
	AuditEntryTypeTMRResult:        "AUDIT_ENTRY_TYPE_TMR_RESULT",
	AuditEntryTypeSystemEvent:      "AUDIT_ENTRY_TYPE_SYSTEM_EVENT",
}

func (a AuditEntryType) String() string {
	if name, ok := auditEntryTypeNames[a]; ok {
		return name
	}
	return "AUDIT_ENTRY_TYPE_UNSPECIFIED"
}

// AuditEntry is a single entry in the tamper-evident audit ledger.
type AuditEntry struct {
	EntryID      uint64         `json:"entry_id"`
	EntryType    AuditEntryType `json:"entry_type"`
	Timestamp    *time.Time     `json:"timestamp,omitempty"`
	GPU          *GpuIdentifier `json:"gpu,omitempty"`
	Data         []byte         `json:"data"`
	PreviousHash []byte         `json:"previous_hash"`
	EntryHash    []byte         `json:"entry_hash"`
	MerkleRoot   []byte         `json:"merkle_root"`
}

// ChainVerificationResult is the response from chain verification.
type ChainVerificationResult struct {
	Valid               bool   `json:"valid"`
	FirstInvalidEntryID uint64 `json:"first_invalid_entry_id"`
	FailureDescription  string `json:"failure_description"`
	EntriesVerified     uint64 `json:"entries_verified"`
	BatchesVerified     uint64 `json:"batches_verified"`
	VerificationTimeMs  uint64 `json:"verification_time_ms"`
}

// AuditQueryFilters contains filters for querying the audit trail.
type AuditQueryFilters struct {
	GPU        *GpuIdentifier `json:"gpu,omitempty"`
	StartTime  *time.Time     `json:"start_time,omitempty"`
	EndTime    *time.Time     `json:"end_time,omitempty"`
	EntryType  AuditEntryType `json:"entry_type"`
	Limit      uint32         `json:"limit"`
	PageToken  string         `json:"page_token"`
	Descending bool           `json:"descending"`
}

// AuditQueryResponse is the response from an audit trail query.
type AuditQueryResponse struct {
	Entries       []*AuditEntry `json:"entries,omitempty"`
	NextPageToken string        `json:"next_page_token"`
	TotalCount    uint64        `json:"total_count"`
}

// ---------------------------------------------------------------------------
// config.proto
// ---------------------------------------------------------------------------

// ProbeScheduleEntry is a single entry in the probe schedule.
type ProbeScheduleEntry struct {
	Type          ProbeType `json:"type"`
	PeriodSeconds uint32   `json:"period_seconds"`
	SmCoverage    float64  `json:"sm_coverage"`
	Priority      uint32   `json:"priority"`
	Enabled       bool     `json:"enabled"`
	TimeoutMs     uint32   `json:"timeout_ms"`
}

// ProbeScheduleUpdate updates the probe execution schedule.
type ProbeScheduleUpdate struct {
	Entries []*ProbeScheduleEntry `json:"entries,omitempty"`
}

// OverheadBudgetUpdate updates the overall overhead budget.
type OverheadBudgetUpdate struct {
	BudgetPct float64 `json:"budget_pct"`
}

// SamplingRateUpdate updates a sampling rate for a specific component.
type SamplingRateUpdate struct {
	Component string  `json:"component"`
	Rate      float64 `json:"rate"`
}

// ThresholdUpdate updates a threshold value for a component parameter.
type ThresholdUpdate struct {
	Component string  `json:"component"`
	Parameter string  `json:"parameter"`
	Value     float64 `json:"value"`
}

// ConfigUpdate is a dynamic configuration update.
// Exactly one of the update fields should be set.
type ConfigUpdate struct {
	UpdateID       string               `json:"update_id"`
	InitiatedBy    string               `json:"initiated_by"`
	Reason         string               `json:"reason"`
	ProbeSchedule  *ProbeScheduleUpdate `json:"probe_schedule,omitempty"`
	OverheadBudget *OverheadBudgetUpdate `json:"overhead_budget,omitempty"`
	SamplingRate   *SamplingRateUpdate  `json:"sampling_rate,omitempty"`
	Threshold      *ThresholdUpdate     `json:"threshold,omitempty"`
}

// ConfigAck is an acknowledgement from a config update recipient.
type ConfigAck struct {
	UpdateID      string `json:"update_id"`
	Applied       bool   `json:"applied"`
	ComponentID   string `json:"component_id"`
	Error         string `json:"error"`
	ConfigVersion uint64 `json:"config_version"`
}

// ---------------------------------------------------------------------------
// telemetry.proto
// ---------------------------------------------------------------------------

// ThermalReading contains thermal sensor readings from a GPU.
type ThermalReading struct {
	GPU                *GpuIdentifier `json:"gpu,omitempty"`
	TemperatureC       float32        `json:"temperature_c"`
	FanSpeedPct        float32        `json:"fan_speed_pct"`
	ThrottleActive     bool           `json:"throttle_active"`
	MemoryTemperatureC float32        `json:"memory_temperature_c"`
	Timestamp          *time.Time     `json:"timestamp,omitempty"`
}

// PowerReading contains electrical power readings from a GPU.
type PowerReading struct {
	GPU                  *GpuIdentifier `json:"gpu,omitempty"`
	PowerW               float32        `json:"power_w"`
	VoltageMv            uint32         `json:"voltage_mv"`
	CurrentMa            uint32         `json:"current_ma"`
	PowerLimitW          float32        `json:"power_limit_w"`
	PowerThrottleActive  bool           `json:"power_throttle_active"`
	Timestamp            *time.Time     `json:"timestamp,omitempty"`
}

// EccCounters contains ECC memory error counters from a GPU.
type EccCounters struct {
	GPU                *GpuIdentifier `json:"gpu,omitempty"`
	SramCorrected      uint64         `json:"sram_corrected"`
	SramUncorrected    uint64         `json:"sram_uncorrected"`
	DramCorrected      uint64         `json:"dram_corrected"`
	DramUncorrected    uint64         `json:"dram_uncorrected"`
	RetiredPages       uint32         `json:"retired_pages"`
	PendingRetiredPages uint32        `json:"pending_retired_pages"`
	ResetRequired      bool           `json:"reset_required"`
	Timestamp          *time.Time     `json:"timestamp,omitempty"`
}

// NvLinkStatus is the status of a single NVLink connection.
type NvLinkStatus struct {
	LinkIndex    uint32 `json:"link_index"`
	Active       bool   `json:"active"`
	CrcErrors    uint64 `json:"crc_errors"`
	ReplayErrors uint64 `json:"replay_errors"`
}

// GpuTelemetryReport is an aggregated telemetry report for a single GPU.
type GpuTelemetryReport struct {
	Thermal             *ThermalReading `json:"thermal,omitempty"`
	Power               *PowerReading   `json:"power,omitempty"`
	Ecc                 *EccCounters    `json:"ecc,omitempty"`
	GpuUtilizationPct   float32         `json:"gpu_utilization_pct"`
	MemoryUtilizationPct float32        `json:"memory_utilization_pct"`
	GpuClockMhz         uint32          `json:"gpu_clock_mhz"`
	MemClockMhz         uint32          `json:"mem_clock_mhz"`
	PcieGen             uint32          `json:"pcie_gen"`
	PcieWidth           uint32          `json:"pcie_width"`
	NvlinkStatus        []*NvLinkStatus `json:"nvlink_status,omitempty"`
}
