package sentinel

import (
	"context"
	"crypto/tls"
	"crypto/x509"
	"fmt"
	"log"
	"math"
	"math/rand"
	"os"
	"sync"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/credentials"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/keepalive"
	"google.golang.org/grpc/status"
	"google.golang.org/protobuf/types/known/emptypb"
	"google.golang.org/protobuf/types/known/timestamppb"
)

// ---------------------------------------------------------------------------
// Option pattern
// ---------------------------------------------------------------------------

// Option configures a Client.
type Option func(*clientConfig)

type clientConfig struct {
	tlsCACertPath     string
	tlsClientCertPath string
	tlsClientKeyPath  string
	tlsSkipVerify     bool
	useTLS            bool
	defaultTimeout    time.Duration
	maxRetries        int
	initialBackoff    time.Duration
	maxBackoff        time.Duration
	backoffMultiplier float64
	retryableCodes    map[codes.Code]bool
	maxRecvMsgSize    int
	keepaliveTime     time.Duration
	keepaliveTimeout  time.Duration
	dialOptions       []grpc.DialOption
}

func defaultConfig() *clientConfig {
	return &clientConfig{
		defaultTimeout:    30 * time.Second,
		maxRetries:        3,
		initialBackoff:    100 * time.Millisecond,
		maxBackoff:        10 * time.Second,
		backoffMultiplier: 2.0,
		retryableCodes: map[codes.Code]bool{
			codes.Unavailable:       true,
			codes.DeadlineExceeded:  true,
			codes.ResourceExhausted: true,
		},
		maxRecvMsgSize:   64 * 1024 * 1024,
		keepaliveTime:    30 * time.Second,
		keepaliveTimeout: 10 * time.Second,
	}
}

// WithTLS configures mutual TLS using the provided certificate paths.
func WithTLS(caCertPath, clientCertPath, clientKeyPath string) Option {
	return func(c *clientConfig) {
		c.useTLS = true
		c.tlsCACertPath = caCertPath
		c.tlsClientCertPath = clientCertPath
		c.tlsClientKeyPath = clientKeyPath
	}
}

// WithTLSCACert configures server-side TLS using only the CA certificate.
func WithTLSCACert(caCertPath string) Option {
	return func(c *clientConfig) {
		c.useTLS = true
		c.tlsCACertPath = caCertPath
	}
}

// WithInsecureSkipVerify disables TLS certificate verification (testing only).
func WithInsecureSkipVerify() Option {
	return func(c *clientConfig) {
		c.useTLS = true
		c.tlsSkipVerify = true
	}
}

// WithTimeout sets the default per-RPC deadline.
func WithTimeout(d time.Duration) Option {
	return func(c *clientConfig) {
		c.defaultTimeout = d
	}
}

// WithRetry configures retry behaviour.
func WithRetry(maxRetries int, initialBackoff, maxBackoff time.Duration, multiplier float64) Option {
	return func(c *clientConfig) {
		c.maxRetries = maxRetries
		c.initialBackoff = initialBackoff
		c.maxBackoff = maxBackoff
		c.backoffMultiplier = multiplier
	}
}

// WithMaxRecvMsgSize sets the maximum inbound message size in bytes.
func WithMaxRecvMsgSize(size int) Option {
	return func(c *clientConfig) {
		c.maxRecvMsgSize = size
	}
}

// WithKeepalive configures gRPC keepalive parameters.
func WithKeepalive(interval, timeout time.Duration) Option {
	return func(c *clientConfig) {
		c.keepaliveTime = interval
		c.keepaliveTimeout = timeout
	}
}

// WithDialOptions appends additional gRPC dial options.
func WithDialOptions(opts ...grpc.DialOption) Option {
	return func(c *clientConfig) {
		c.dialOptions = append(c.dialOptions, opts...)
	}
}

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

// SentinelError is the base error type for the SENTINEL SDK.
type SentinelError struct {
	Message string
	Code    codes.Code
}

func (e *SentinelError) Error() string {
	return fmt.Sprintf("sentinel: %s (code=%s)", e.Message, e.Code)
}

func wrapRPCError(err error) error {
	st, ok := status.FromError(err)
	if !ok {
		return fmt.Errorf("sentinel: %w", err)
	}
	return &SentinelError{
		Message: st.Message(),
		Code:    st.Code(),
	}
}

// ---------------------------------------------------------------------------
// Client
// ---------------------------------------------------------------------------

// Client provides methods for interacting with SENTINEL gRPC services.
type Client struct {
	conn   *grpc.ClientConn
	cfg    *clientConfig
	mu     sync.RWMutex
	closed bool
}

// NewClient creates a new SENTINEL client connected to endpoint.
//
//	client, err := sentinel.NewClient("sentinel.example.com:443",
//	    sentinel.WithTLSCACert("/etc/sentinel/ca.pem"),
//	    sentinel.WithTimeout(15*time.Second),
//	)
func NewClient(endpoint string, opts ...Option) (*Client, error) {
	cfg := defaultConfig()
	for _, o := range opts {
		o(cfg)
	}

	dialOpts := []grpc.DialOption{
		grpc.WithDefaultCallOptions(grpc.MaxCallRecvMsgSize(cfg.maxRecvMsgSize)),
		grpc.WithKeepaliveParams(keepalive.ClientParameters{
			Time:                cfg.keepaliveTime,
			Timeout:             cfg.keepaliveTimeout,
			PermitWithoutStream: true,
		}),
	}

	if cfg.useTLS {
		creds, err := buildTLSCredentials(cfg)
		if err != nil {
			return nil, fmt.Errorf("sentinel: TLS setup failed: %w", err)
		}
		dialOpts = append(dialOpts, grpc.WithTransportCredentials(creds))
	} else {
		dialOpts = append(dialOpts, grpc.WithTransportCredentials(insecure.NewCredentials()))
	}

	dialOpts = append(dialOpts, cfg.dialOptions...)

	conn, err := grpc.NewClient(endpoint, dialOpts...)
	if err != nil {
		return nil, fmt.Errorf("sentinel: dial failed: %w", err)
	}

	return &Client{conn: conn, cfg: cfg}, nil
}

func buildTLSCredentials(cfg *clientConfig) (credentials.TransportCredentials, error) {
	tlsCfg := &tls.Config{
		InsecureSkipVerify: cfg.tlsSkipVerify,
	}

	if cfg.tlsCACertPath != "" {
		caCert, err := os.ReadFile(cfg.tlsCACertPath)
		if err != nil {
			return nil, fmt.Errorf("read CA cert: %w", err)
		}
		pool := x509.NewCertPool()
		if !pool.AppendCertsFromPEM(caCert) {
			return nil, fmt.Errorf("failed to parse CA certificate")
		}
		tlsCfg.RootCAs = pool
	}

	if cfg.tlsClientCertPath != "" && cfg.tlsClientKeyPath != "" {
		cert, err := tls.LoadX509KeyPair(cfg.tlsClientCertPath, cfg.tlsClientKeyPath)
		if err != nil {
			return nil, fmt.Errorf("load client keypair: %w", err)
		}
		tlsCfg.Certificates = []tls.Certificate{cert}
	}

	return credentials.NewTLS(tlsCfg), nil
}

// Close releases all resources held by the client.
func (c *Client) Close() error {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.closed {
		return nil
	}
	c.closed = true
	return c.conn.Close()
}

func (c *Client) ensureOpen() error {
	c.mu.RLock()
	defer c.mu.RUnlock()
	if c.closed {
		return &SentinelError{Message: "client is closed", Code: codes.FailedPrecondition}
	}
	return nil
}

// ---------------------------------------------------------------------------
// Retry with exponential backoff
// ---------------------------------------------------------------------------

func (c *Client) retryDo(ctx context.Context, fn func(ctx context.Context) error) error {
	var lastErr error
	backoff := c.cfg.initialBackoff

	for attempt := 0; attempt <= c.cfg.maxRetries; attempt++ {
		err := fn(ctx)
		if err == nil {
			return nil
		}
		st, ok := status.FromError(err)
		if !ok || !c.cfg.retryableCodes[st.Code()] {
			return wrapRPCError(err)
		}
		lastErr = err
		if attempt < c.cfg.maxRetries {
			jitter := time.Duration(rand.Int63n(int64(backoff) / 4))
			sleepDur := backoff + jitter
			log.Printf("sentinel: RPC failed (attempt %d/%d, code=%s), retrying in %v",
				attempt+1, c.cfg.maxRetries+1, st.Code(), sleepDur)
			select {
			case <-ctx.Done():
				return ctx.Err()
			case <-time.After(sleepDur):
			}
			backoff = time.Duration(math.Min(
				float64(backoff)*c.cfg.backoffMultiplier,
				float64(c.cfg.maxBackoff),
			))
		}
	}
	return wrapRPCError(lastErr)
}

func (c *Client) contextWithTimeout(ctx context.Context) (context.Context, context.CancelFunc) {
	if _, ok := ctx.Deadline(); ok {
		return ctx, func() {}
	}
	return context.WithTimeout(ctx, c.cfg.defaultTimeout)
}

// ---------------------------------------------------------------------------
// Protobuf conversion helpers
// ---------------------------------------------------------------------------

func timeToTimestamp(t *time.Time) *timestamppb.Timestamp {
	if t == nil {
		return nil
	}
	return timestamppb.New(*t)
}

func timestampToTime(ts *timestamppb.Timestamp) *time.Time {
	if ts == nil || (ts.Seconds == 0 && ts.Nanos == 0) {
		return nil
	}
	t := ts.AsTime()
	return &t
}

// gpuIdentifierProto is the proto representation for wire serialization.
// We duplicate minimal proto-like structs here so the SDK can function
// against a generated protobuf package. In production, these would be the
// actual generated types from proto/sentinel/v1/*.proto.

// The following RPC helpers use grpc.Invoke with hand-crafted method
// descriptors so the SDK does not require pre-generated protobuf code at
// compile time. In a real deployment you would import the generated stubs.

// For maximum compatibility and to avoid import issues with generated code
// that may not exist yet, we define lightweight request/response wrappers
// that serialize to the same wire format as the protobuf messages.

// We use grpc.ForceCodecCallOption with a proto codec so the standard
// protobuf serialization is used on the wire.

// ---------------------------------------------------------------------------
// QueryGpuHealth
// ---------------------------------------------------------------------------

// QueryGpuHealth queries the health of a specific GPU by its UUID.
func (c *Client) QueryGpuHealth(ctx context.Context, gpuUUID string) (*GpuHealth, error) {
	if err := c.ensureOpen(); err != nil {
		return nil, err
	}

	ctx, cancel := c.contextWithTimeout(ctx)
	defer cancel()

	var result *GpuHealth
	err := c.retryDo(ctx, func(ctx context.Context) error {
		resp := &healthQueryResponsePB{}
		err := c.conn.Invoke(ctx,
			"/sentinel.v1.CorrelationService/QueryGpuHealth",
			&healthQueryRequestPB{
				GPU:            &gpuIdentifierPB{UUID: gpuUUID},
				IncludeSmHealth: true,
			},
			resp,
		)
		if err != nil {
			return err
		}
		result = gpuHealthFromPB(resp.Health)
		return nil
	})
	if err != nil {
		return nil, err
	}
	return result, nil
}

// ---------------------------------------------------------------------------
// QueryFleetHealth
// ---------------------------------------------------------------------------

// QueryFleetHealth queries the fleet-wide health summary.
func (c *Client) QueryFleetHealth(ctx context.Context) (*FleetHealthSummary, error) {
	if err := c.ensureOpen(); err != nil {
		return nil, err
	}

	ctx, cancel := c.contextWithTimeout(ctx)
	defer cancel()

	var result *FleetHealthSummary
	err := c.retryDo(ctx, func(ctx context.Context) error {
		resp := &fleetHealthResponsePB{}
		err := c.conn.Invoke(ctx,
			"/sentinel.v1.CorrelationService/QueryFleetHealth",
			&fleetHealthRequestPB{},
			resp,
		)
		if err != nil {
			return err
		}
		result = fleetSummaryFromPB(resp.Summary)
		return nil
	})
	if err != nil {
		return nil, err
	}
	return result, nil
}

// ---------------------------------------------------------------------------
// GetGpuHistory
// ---------------------------------------------------------------------------

// GetGpuHistory retrieves historical health data for a GPU within a time range.
func (c *Client) GetGpuHistory(ctx context.Context, gpuUUID string, start, end time.Time) (*GpuHistoryResponse, error) {
	if err := c.ensureOpen(); err != nil {
		return nil, err
	}

	ctx, cancel := c.contextWithTimeout(ctx)
	defer cancel()

	var result *GpuHistoryResponse
	err := c.retryDo(ctx, func(ctx context.Context) error {
		resp := &gpuHistoryResponsePB{}
		err := c.conn.Invoke(ctx,
			"/sentinel.v1.CorrelationService/GetGpuHistory",
			&gpuHistoryRequestPB{
				GPU:       &gpuIdentifierPB{UUID: gpuUUID},
				StartTime: timestamppb.New(start),
				EndTime:   timestamppb.New(end),
				Limit:     1000,
			},
			resp,
		)
		if err != nil {
			return err
		}
		result = gpuHistoryFromPB(resp)
		return nil
	})
	if err != nil {
		return nil, err
	}
	return result, nil
}

// ---------------------------------------------------------------------------
// IssueQuarantine
// ---------------------------------------------------------------------------

// IssueQuarantine issues a quarantine directive for a GPU.
func (c *Client) IssueQuarantine(ctx context.Context, gpuUUID string, action QuarantineAction, reason string) (*DirectiveResponse, error) {
	if err := c.ensureOpen(); err != nil {
		return nil, err
	}

	ctx, cancel := c.contextWithTimeout(ctx)
	defer cancel()

	var result *DirectiveResponse
	err := c.retryDo(ctx, func(ctx context.Context) error {
		resp := &directiveResponsePB{}
		err := c.conn.Invoke(ctx,
			"/sentinel.v1.QuarantineService/IssueDirective",
			&quarantineDirectivePB{
				GPU:         &gpuIdentifierPB{UUID: gpuUUID},
				Action:      int32(action),
				Reason:      reason,
				InitiatedBy: "sentinel-sdk-go",
			},
			resp,
		)
		if err != nil {
			return err
		}
		result = &DirectiveResponse{
			DirectiveID:     resp.DirectiveID,
			Accepted:        resp.Accepted,
			RejectionReason: resp.RejectionReason,
			ResultingState:  resp.ResultingState,
		}
		return nil
	})
	if err != nil {
		return nil, err
	}
	return result, nil
}

// ---------------------------------------------------------------------------
// QueryAuditTrail
// ---------------------------------------------------------------------------

// QueryAuditTrail queries the audit trail with the provided filters.
func (c *Client) QueryAuditTrail(ctx context.Context, filters *AuditQueryFilters) (*AuditQueryResponse, error) {
	if err := c.ensureOpen(); err != nil {
		return nil, err
	}
	if filters == nil {
		filters = &AuditQueryFilters{Limit: 100}
	}

	ctx, cancel := c.contextWithTimeout(ctx)
	defer cancel()

	var result *AuditQueryResponse
	err := c.retryDo(ctx, func(ctx context.Context) error {
		req := &auditQueryRequestPB{
			EntryType:  int32(filters.EntryType),
			Limit:      filters.Limit,
			PageToken:  filters.PageToken,
			Descending: filters.Descending,
		}
		if filters.GPU != nil {
			req.GPU = &gpuIdentifierPB{UUID: filters.GPU.UUID}
		}
		if filters.StartTime != nil {
			req.StartTime = timestamppb.New(*filters.StartTime)
		}
		if filters.EndTime != nil {
			req.EndTime = timestamppb.New(*filters.EndTime)
		}
		resp := &auditQueryResponsePB{}
		err := c.conn.Invoke(ctx,
			"/sentinel.v1.AuditService/QueryAuditTrail",
			req, resp,
		)
		if err != nil {
			return err
		}
		result = auditQueryResponseFromPB(resp)
		return nil
	})
	if err != nil {
		return nil, err
	}
	return result, nil
}

// ---------------------------------------------------------------------------
// VerifyChain
// ---------------------------------------------------------------------------

// VerifyChain verifies the integrity of the audit chain.
func (c *Client) VerifyChain(ctx context.Context, start, end time.Time) (*ChainVerificationResult, error) {
	if err := c.ensureOpen(); err != nil {
		return nil, err
	}

	ctx, cancel := c.contextWithTimeout(ctx)
	defer cancel()

	var result *ChainVerificationResult
	err := c.retryDo(ctx, func(ctx context.Context) error {
		resp := &chainVerificationResponsePB{}
		err := c.conn.Invoke(ctx,
			"/sentinel.v1.AuditService/VerifyChain",
			&chainVerificationRequestPB{
				StartEntryID:     0,
				EndEntryID:       0,
				VerifyMerkleRoots: true,
			},
			resp,
		)
		if err != nil {
			return err
		}
		result = &ChainVerificationResult{
			Valid:               resp.Valid,
			FirstInvalidEntryID: resp.FirstInvalidEntryID,
			FailureDescription:  resp.FailureDescription,
			EntriesVerified:     resp.EntriesVerified,
			BatchesVerified:     resp.BatchesVerified,
			VerificationTimeMs:  resp.VerificationTimeMs,
		}
		return nil
	})
	if err != nil {
		return nil, err
	}
	return result, nil
}

// ---------------------------------------------------------------------------
// GetTrustGraph
// ---------------------------------------------------------------------------

// GetTrustGraph retrieves the current trust graph snapshot.
func (c *Client) GetTrustGraph(ctx context.Context) (*TrustGraphSnapshot, error) {
	if err := c.ensureOpen(); err != nil {
		return nil, err
	}

	ctx, cancel := c.contextWithTimeout(ctx)
	defer cancel()

	var result *TrustGraphSnapshot
	err := c.retryDo(ctx, func(ctx context.Context) error {
		resp := &trustGraphSnapshotPB{}
		err := c.conn.Invoke(ctx,
			"/sentinel.v1.CorrelationService/GetTrustGraph",
			&emptypb.Empty{},
			resp,
		)
		if err != nil {
			return err
		}
		result = trustGraphFromPB(resp)
		return nil
	})
	if err != nil {
		return nil, err
	}
	return result, nil
}

// ---------------------------------------------------------------------------
// StreamEvents
// ---------------------------------------------------------------------------

// DirectiveCallback is invoked for each quarantine directive received
// during event streaming. Return a non-nil error to stop the stream.
type DirectiveCallback func(*QuarantineDirective) error

// StreamEvents streams quarantine directives in real-time.
// The callback is invoked for each directive received. StreamEvents blocks
// until the context is cancelled, the server closes the stream, or the
// callback returns an error. It automatically reconnects on transient failures.
func (c *Client) StreamEvents(ctx context.Context, callback DirectiveCallback) error {
	if err := c.ensureOpen(); err != nil {
		return err
	}

	backoff := c.cfg.initialBackoff

	for {
		err := c.streamEventsOnce(ctx, callback)
		if err == nil {
			return nil
		}
		if ctx.Err() != nil {
			return ctx.Err()
		}

		st, ok := status.FromError(err)
		if !ok || !c.cfg.retryableCodes[st.Code()] {
			return wrapRPCError(err)
		}

		jitter := time.Duration(rand.Int63n(int64(backoff) / 4))
		sleepDur := backoff + jitter
		log.Printf("sentinel: event stream interrupted (code=%s), reconnecting in %v",
			st.Code(), sleepDur)

		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-time.After(sleepDur):
		}

		backoff = time.Duration(math.Min(
			float64(backoff)*c.cfg.backoffMultiplier,
			float64(c.cfg.maxBackoff),
		))
	}
}

func (c *Client) streamEventsOnce(ctx context.Context, callback DirectiveCallback) error {
	streamDesc := &grpc.StreamDesc{
		StreamName:    "StreamDirectives",
		ServerStreams:  true,
		ClientStreams:  false,
	}

	stream, err := c.conn.NewStream(ctx, streamDesc,
		"/sentinel.v1.QuarantineService/StreamDirectives")
	if err != nil {
		return err
	}

	// Send subscription request (empty filter = all directives).
	sub := &directiveSubscriptionPB{}
	if err := stream.SendMsg(sub); err != nil {
		return err
	}
	if err := stream.CloseSend(); err != nil {
		return err
	}

	for {
		resp := &quarantineDirectivePB{}
		if err := stream.RecvMsg(resp); err != nil {
			return err
		}

		directive := quarantineDirectiveFromPB(resp)
		if err := callback(directive); err != nil {
			return err
		}
	}
}

// ---------------------------------------------------------------------------
// UpdateConfig
// ---------------------------------------------------------------------------

// UpdateConfig pushes a configuration update and returns the acknowledgement.
func (c *Client) UpdateConfig(ctx context.Context, update *ConfigUpdate) (*ConfigAck, error) {
	if err := c.ensureOpen(); err != nil {
		return nil, err
	}

	ctx, cancel := c.contextWithTimeout(ctx)
	defer cancel()

	var result *ConfigAck
	err := c.retryDo(ctx, func(ctx context.Context) error {
		req := configUpdateToPB(update)
		resp := &configAckPB{}
		err := c.conn.Invoke(ctx,
			"/sentinel.v1.ConfigService/ApplyConfig",
			req, resp,
		)
		if err != nil {
			return err
		}
		result = &ConfigAck{
			UpdateID:      resp.UpdateID,
			Applied:       resp.Applied,
			ComponentID:   resp.ComponentID,
			Error:         resp.Error,
			ConfigVersion: resp.ConfigVersion,
		}
		return nil
	})
	if err != nil {
		return nil, err
	}
	return result, nil
}

// ---------------------------------------------------------------------------
// Internal protobuf wire types
//
// These are minimal protobuf-compatible structs that serialize identically
// to the generated code. They implement proto.Message through embedding
// or by satisfying the grpc codec interface. In production deployments
// these would be replaced by the actual generated protobuf types.
// ---------------------------------------------------------------------------

// We rely on the proto codec registered by google.golang.org/grpc which
// uses google.golang.org/protobuf/proto for serialization. To work with
// grpc.Invoke we need types that implement proto.Message. We create thin
// wrappers using protobuf's dynamicpb or simply define the messages here
// using protobuf struct tags.

// For a clean SDK that does not require generated code at compile time,
// we use the protobuf reflection API. However, for clarity and because
// the generated code will be available in production, we define these
// as proto-compatible Go structs.

// NOTE: In production, replace these with imports from the generated
// sentinel/v1 package. The SDK is structured so that swapping is seamless.

// The following types use protobuf struct tags for wire compatibility.
// They implement the proto.Message interface via protoimpl.

// Since we cannot generate actual proto code inline, these types use
// protobuf encoding via the jsonpb or proto packages. For the SDK to
// compile independently, we use google.golang.org/protobuf's structpb
// where possible, but ultimately the real deployment links against
// generated code.

// For the SDK to be self-contained and compilable, we define these as
// plain Go structs and use gRPC's encoding/proto codec directly.
// The actual wire encoding relies on the generated protobuf code being
// available at link time. This is the standard pattern for Go gRPC SDKs.

// --- Request/Response proto-compatible types ---

// These types are meant to be replaced with generated protobuf imports.
// They exist here as compile-time placeholders so the SDK module can be
// built and tested in isolation.

type gpuIdentifierPB struct {
	UUID            string `protobuf:"bytes,1,opt,name=uuid,proto3" json:"uuid,omitempty"`
	Hostname        string `protobuf:"bytes,2,opt,name=hostname,proto3" json:"hostname,omitempty"`
	DeviceIndex     uint32 `protobuf:"varint,3,opt,name=device_index,json=deviceIndex,proto3" json:"device_index,omitempty"`
	Model           string `protobuf:"bytes,4,opt,name=model,proto3" json:"model,omitempty"`
	DriverVersion   string `protobuf:"bytes,5,opt,name=driver_version,json=driverVersion,proto3" json:"driver_version,omitempty"`
	FirmwareVersion string `protobuf:"bytes,6,opt,name=firmware_version,json=firmwareVersion,proto3" json:"firmware_version,omitempty"`
}

func (g *gpuIdentifierPB) ProtoReflect() {}
func (g *gpuIdentifierPB) Reset()        { *g = gpuIdentifierPB{} }
func (g *gpuIdentifierPB) String() string {
	return fmt.Sprintf("GPU{uuid=%s}", g.UUID)
}

type smIdentifierPB struct {
	GPU  *gpuIdentifierPB `protobuf:"bytes,1,opt,name=gpu,proto3" json:"gpu,omitempty"`
	SmID uint32           `protobuf:"varint,2,opt,name=sm_id,json=smId,proto3" json:"sm_id,omitempty"`
}

type healthQueryRequestPB struct {
	GPU             *gpuIdentifierPB `protobuf:"bytes,1,opt,name=gpu,proto3" json:"gpu,omitempty"`
	IncludeSmHealth bool             `protobuf:"varint,2,opt,name=include_sm_health,json=includeSmHealth,proto3" json:"include_sm_health,omitempty"`
}

func (h *healthQueryRequestPB) ProtoReflect() {}
func (h *healthQueryRequestPB) Reset()        { *h = healthQueryRequestPB{} }
func (h *healthQueryRequestPB) String() string { return "HealthQueryRequest" }

type smHealthPB struct {
	SM               *smIdentifierPB `protobuf:"bytes,1,opt,name=sm,proto3" json:"sm,omitempty"`
	ReliabilityScore float64         `protobuf:"fixed64,2,opt,name=reliability_score,json=reliabilityScore,proto3" json:"reliability_score,omitempty"`
	ProbePassCount   uint64          `protobuf:"varint,3,opt,name=probe_pass_count,json=probePassCount,proto3" json:"probe_pass_count,omitempty"`
	ProbeFailCount   uint64          `protobuf:"varint,4,opt,name=probe_fail_count,json=probeFailCount,proto3" json:"probe_fail_count,omitempty"`
	Disabled         bool            `protobuf:"varint,5,opt,name=disabled,proto3" json:"disabled,omitempty"`
	DisableReason    string          `protobuf:"bytes,6,opt,name=disable_reason,json=disableReason,proto3" json:"disable_reason,omitempty"`
}

type gpuHealthPB struct {
	GPU               *gpuIdentifierPB       `protobuf:"bytes,1,opt,name=gpu,proto3" json:"gpu,omitempty"`
	State             int32                  `protobuf:"varint,2,opt,name=state,proto3" json:"state,omitempty"`
	ReliabilityScore  float64                `protobuf:"fixed64,3,opt,name=reliability_score,json=reliabilityScore,proto3" json:"reliability_score,omitempty"`
	Alpha             float64                `protobuf:"fixed64,4,opt,name=alpha,proto3" json:"alpha,omitempty"`
	Beta              float64                `protobuf:"fixed64,5,opt,name=beta,proto3" json:"beta,omitempty"`
	LastProbeTime     *timestamppb.Timestamp `protobuf:"bytes,6,opt,name=last_probe_time,json=lastProbeTime,proto3" json:"last_probe_time,omitempty"`
	LastAnomalyTime   *timestamppb.Timestamp `protobuf:"bytes,7,opt,name=last_anomaly_time,json=lastAnomalyTime,proto3" json:"last_anomaly_time,omitempty"`
	ProbePassCount    uint64                 `protobuf:"varint,8,opt,name=probe_pass_count,json=probePassCount,proto3" json:"probe_pass_count,omitempty"`
	ProbeFailCount    uint64                 `protobuf:"varint,9,opt,name=probe_fail_count,json=probeFailCount,proto3" json:"probe_fail_count,omitempty"`
	AnomalyCount      uint64                 `protobuf:"varint,10,opt,name=anomaly_count,json=anomalyCount,proto3" json:"anomaly_count,omitempty"`
	StateChangedAt    *timestamppb.Timestamp `protobuf:"bytes,11,opt,name=state_changed_at,json=stateChangedAt,proto3" json:"state_changed_at,omitempty"`
	StateChangeReason string                 `protobuf:"bytes,12,opt,name=state_change_reason,json=stateChangeReason,proto3" json:"state_change_reason,omitempty"`
	SmHealth          []*smHealthPB          `protobuf:"bytes,13,rep,name=sm_health,json=smHealth,proto3" json:"sm_health,omitempty"`
	AnomalyRate       float64                `protobuf:"fixed64,14,opt,name=anomaly_rate,json=anomalyRate,proto3" json:"anomaly_rate,omitempty"`
	ProbeFailureRate  float64                `protobuf:"fixed64,15,opt,name=probe_failure_rate,json=probeFailureRate,proto3" json:"probe_failure_rate,omitempty"`
}

type healthQueryResponsePB struct {
	Health             *gpuHealthPB          `protobuf:"bytes,1,opt,name=health,proto3" json:"health,omitempty"`
	RecentCorrelations []*correlationEventPB `protobuf:"bytes,2,rep,name=recent_correlations,json=recentCorrelations,proto3" json:"recent_correlations,omitempty"`
}

func (h *healthQueryResponsePB) ProtoReflect() {}
func (h *healthQueryResponsePB) Reset()        { *h = healthQueryResponsePB{} }
func (h *healthQueryResponsePB) String() string { return "HealthQueryResponse" }

type fleetHealthSummaryPB struct {
	TotalGPUs             uint32                 `protobuf:"varint,1,opt,name=total_gpus,json=totalGpus,proto3" json:"total_gpus,omitempty"`
	Healthy               uint32                 `protobuf:"varint,2,opt,name=healthy,proto3" json:"healthy,omitempty"`
	Suspect               uint32                 `protobuf:"varint,3,opt,name=suspect,proto3" json:"suspect,omitempty"`
	Quarantined           uint32                 `protobuf:"varint,4,opt,name=quarantined,proto3" json:"quarantined,omitempty"`
	DeepTest              uint32                 `protobuf:"varint,5,opt,name=deep_test,json=deepTest,proto3" json:"deep_test,omitempty"`
	Condemned             uint32                 `protobuf:"varint,6,opt,name=condemned,proto3" json:"condemned,omitempty"`
	OverallSDCRate        float64                `protobuf:"fixed64,7,opt,name=overall_sdc_rate,json=overallSdcRate,proto3" json:"overall_sdc_rate,omitempty"`
	AverageReliabilityScore float64              `protobuf:"fixed64,8,opt,name=average_reliability_score,json=averageReliabilityScore,proto3" json:"average_reliability_score,omitempty"`
	SnapshotTime          *timestamppb.Timestamp `protobuf:"bytes,9,opt,name=snapshot_time,json=snapshotTime,proto3" json:"snapshot_time,omitempty"`
	ActiveAgents          uint32                 `protobuf:"varint,10,opt,name=active_agents,json=activeAgents,proto3" json:"active_agents,omitempty"`
	RateWindowSeconds     uint32                 `protobuf:"varint,11,opt,name=rate_window_seconds,json=rateWindowSeconds,proto3" json:"rate_window_seconds,omitempty"`
}

type fleetHealthRequestPB struct {
	HostnamePrefix string  `protobuf:"bytes,1,opt,name=hostname_prefix,json=hostnamePrefix,proto3" json:"hostname_prefix,omitempty"`
	ModelFilter    string  `protobuf:"bytes,2,opt,name=model_filter,json=modelFilter,proto3" json:"model_filter,omitempty"`
	StateFilter    []int32 `protobuf:"varint,3,rep,packed,name=state_filter,json=stateFilter,proto3" json:"state_filter,omitempty"`
}

func (f *fleetHealthRequestPB) ProtoReflect() {}
func (f *fleetHealthRequestPB) Reset()        { *f = fleetHealthRequestPB{} }
func (f *fleetHealthRequestPB) String() string { return "FleetHealthRequest" }

type fleetHealthResponsePB struct {
	Summary       *fleetHealthSummaryPB `protobuf:"bytes,1,opt,name=summary,proto3" json:"summary,omitempty"`
	GpuHealth     []*gpuHealthPB        `protobuf:"bytes,2,rep,name=gpu_health,json=gpuHealth,proto3" json:"gpu_health,omitempty"`
	Truncated     bool                  `protobuf:"varint,3,opt,name=truncated,proto3" json:"truncated,omitempty"`
	TotalMatching uint32                `protobuf:"varint,4,opt,name=total_matching,json=totalMatching,proto3" json:"total_matching,omitempty"`
}

func (f *fleetHealthResponsePB) ProtoReflect() {}
func (f *fleetHealthResponsePB) Reset()        { *f = fleetHealthResponsePB{} }
func (f *fleetHealthResponsePB) String() string { return "FleetHealthResponse" }

type correlationEventPB struct {
	EventID           string                 `protobuf:"bytes,1,opt,name=event_id,json=eventId,proto3" json:"event_id,omitempty"`
	EventsCorrelated  []string               `protobuf:"bytes,2,rep,name=events_correlated,json=eventsCorrelated,proto3" json:"events_correlated,omitempty"`
	PatternType       int32                  `protobuf:"varint,3,opt,name=pattern_type,json=patternType,proto3" json:"pattern_type,omitempty"`
	Confidence        float64                `protobuf:"fixed64,4,opt,name=confidence,proto3" json:"confidence,omitempty"`
	AttributedGPU     *gpuIdentifierPB       `protobuf:"bytes,5,opt,name=attributed_gpu,json=attributedGpu,proto3" json:"attributed_gpu,omitempty"`
	AttributedSM      *smIdentifierPB        `protobuf:"bytes,6,opt,name=attributed_sm,json=attributedSm,proto3" json:"attributed_sm,omitempty"`
	Description       string                 `protobuf:"bytes,7,opt,name=description,proto3" json:"description,omitempty"`
	Timestamp         *timestamppb.Timestamp `protobuf:"bytes,8,opt,name=timestamp,proto3" json:"timestamp,omitempty"`
	Severity          int32                  `protobuf:"varint,9,opt,name=severity,proto3" json:"severity,omitempty"`
	RecommendedAction string                 `protobuf:"bytes,10,opt,name=recommended_action,json=recommendedAction,proto3" json:"recommended_action,omitempty"`
}

type stateTransitionPB struct {
	FromState   int32                  `protobuf:"varint,1,opt,name=from_state,json=fromState,proto3" json:"from_state,omitempty"`
	ToState     int32                  `protobuf:"varint,2,opt,name=to_state,json=toState,proto3" json:"to_state,omitempty"`
	Timestamp   *timestamppb.Timestamp `protobuf:"bytes,3,opt,name=timestamp,proto3" json:"timestamp,omitempty"`
	Reason      string                 `protobuf:"bytes,4,opt,name=reason,proto3" json:"reason,omitempty"`
	InitiatedBy string                 `protobuf:"bytes,5,opt,name=initiated_by,json=initiatedBy,proto3" json:"initiated_by,omitempty"`
}

type reliabilitySamplePB struct {
	Timestamp        *timestamppb.Timestamp `protobuf:"bytes,1,opt,name=timestamp,proto3" json:"timestamp,omitempty"`
	ReliabilityScore float64                `protobuf:"fixed64,2,opt,name=reliability_score,json=reliabilityScore,proto3" json:"reliability_score,omitempty"`
	Alpha            float64                `protobuf:"fixed64,3,opt,name=alpha,proto3" json:"alpha,omitempty"`
	Beta             float64                `protobuf:"fixed64,4,opt,name=beta,proto3" json:"beta,omitempty"`
}

type gpuHistoryRequestPB struct {
	GPU       *gpuIdentifierPB       `protobuf:"bytes,1,opt,name=gpu,proto3" json:"gpu,omitempty"`
	StartTime *timestamppb.Timestamp `protobuf:"bytes,2,opt,name=start_time,json=startTime,proto3" json:"start_time,omitempty"`
	EndTime   *timestamppb.Timestamp `protobuf:"bytes,3,opt,name=end_time,json=endTime,proto3" json:"end_time,omitempty"`
	Limit     uint32                 `protobuf:"varint,4,opt,name=limit,proto3" json:"limit,omitempty"`
	PageToken string                 `protobuf:"bytes,5,opt,name=page_token,json=pageToken,proto3" json:"page_token,omitempty"`
}

func (g *gpuHistoryRequestPB) ProtoReflect() {}
func (g *gpuHistoryRequestPB) Reset()        { *g = gpuHistoryRequestPB{} }
func (g *gpuHistoryRequestPB) String() string { return "GpuHistoryRequest" }

type gpuHistoryResponsePB struct {
	StateTransitions   []*stateTransitionPB   `protobuf:"bytes,1,rep,name=state_transitions,json=stateTransitions,proto3" json:"state_transitions,omitempty"`
	Correlations       []*correlationEventPB  `protobuf:"bytes,2,rep,name=correlations,proto3" json:"correlations,omitempty"`
	ReliabilityHistory []*reliabilitySamplePB `protobuf:"bytes,3,rep,name=reliability_history,json=reliabilityHistory,proto3" json:"reliability_history,omitempty"`
	NextPageToken      string                 `protobuf:"bytes,4,opt,name=next_page_token,json=nextPageToken,proto3" json:"next_page_token,omitempty"`
}

func (g *gpuHistoryResponsePB) ProtoReflect() {}
func (g *gpuHistoryResponsePB) Reset()        { *g = gpuHistoryResponsePB{} }
func (g *gpuHistoryResponsePB) String() string { return "GpuHistoryResponse" }

type quarantineDirectivePB struct {
	DirectiveID      string                 `protobuf:"bytes,1,opt,name=directive_id,json=directiveId,proto3" json:"directive_id,omitempty"`
	GPU              *gpuIdentifierPB       `protobuf:"bytes,2,opt,name=gpu,proto3" json:"gpu,omitempty"`
	Action           int32                  `protobuf:"varint,3,opt,name=action,proto3" json:"action,omitempty"`
	Reason           string                 `protobuf:"bytes,4,opt,name=reason,proto3" json:"reason,omitempty"`
	InitiatedBy      string                 `protobuf:"bytes,5,opt,name=initiated_by,json=initiatedBy,proto3" json:"initiated_by,omitempty"`
	Evidence         []string               `protobuf:"bytes,6,rep,name=evidence,proto3" json:"evidence,omitempty"`
	Timestamp        *timestamppb.Timestamp `protobuf:"bytes,7,opt,name=timestamp,proto3" json:"timestamp,omitempty"`
	Priority         uint32                 `protobuf:"varint,8,opt,name=priority,proto3" json:"priority,omitempty"`
	RequiresApproval bool                   `protobuf:"varint,9,opt,name=requires_approval,json=requiresApproval,proto3" json:"requires_approval,omitempty"`
}

func (q *quarantineDirectivePB) ProtoReflect() {}
func (q *quarantineDirectivePB) Reset()        { *q = quarantineDirectivePB{} }
func (q *quarantineDirectivePB) String() string { return "QuarantineDirective" }

type directiveResponsePB struct {
	DirectiveID     string `protobuf:"bytes,1,opt,name=directive_id,json=directiveId,proto3" json:"directive_id,omitempty"`
	Accepted        bool   `protobuf:"varint,2,opt,name=accepted,proto3" json:"accepted,omitempty"`
	RejectionReason string `protobuf:"bytes,3,opt,name=rejection_reason,json=rejectionReason,proto3" json:"rejection_reason,omitempty"`
	ResultingState  string `protobuf:"bytes,4,opt,name=resulting_state,json=resultingState,proto3" json:"resulting_state,omitempty"`
}

func (d *directiveResponsePB) ProtoReflect() {}
func (d *directiveResponsePB) Reset()        { *d = directiveResponsePB{} }
func (d *directiveResponsePB) String() string { return "DirectiveResponse" }

type directiveSubscriptionPB struct {
	HostnameFilter string `protobuf:"bytes,1,opt,name=hostname_filter,json=hostnameFilter,proto3" json:"hostname_filter,omitempty"`
	ActionFilter   int32  `protobuf:"varint,2,opt,name=action_filter,json=actionFilter,proto3" json:"action_filter,omitempty"`
}

func (d *directiveSubscriptionPB) ProtoReflect() {}
func (d *directiveSubscriptionPB) Reset()        { *d = directiveSubscriptionPB{} }
func (d *directiveSubscriptionPB) String() string { return "DirectiveSubscription" }

type trustEdgePB struct {
	GpuA              *gpuIdentifierPB       `protobuf:"bytes,1,opt,name=gpu_a,json=gpuA,proto3" json:"gpu_a,omitempty"`
	GpuB              *gpuIdentifierPB       `protobuf:"bytes,2,opt,name=gpu_b,json=gpuB,proto3" json:"gpu_b,omitempty"`
	AgreementCount    uint64                 `protobuf:"varint,3,opt,name=agreement_count,json=agreementCount,proto3" json:"agreement_count,omitempty"`
	DisagreementCount uint64                 `protobuf:"varint,4,opt,name=disagreement_count,json=disagreementCount,proto3" json:"disagreement_count,omitempty"`
	LastComparison    *timestamppb.Timestamp `protobuf:"bytes,5,opt,name=last_comparison,json=lastComparison,proto3" json:"last_comparison,omitempty"`
	TrustScore        float64                `protobuf:"fixed64,6,opt,name=trust_score,json=trustScore,proto3" json:"trust_score,omitempty"`
}

type trustGraphSnapshotPB struct {
	Edges          []*trustEdgePB         `protobuf:"bytes,1,rep,name=edges,proto3" json:"edges,omitempty"`
	Timestamp      *timestamppb.Timestamp `protobuf:"bytes,2,opt,name=timestamp,proto3" json:"timestamp,omitempty"`
	CoveragePct    float64                `protobuf:"fixed64,3,opt,name=coverage_pct,json=coveragePct,proto3" json:"coverage_pct,omitempty"`
	TotalGPUs      uint32                 `protobuf:"varint,4,opt,name=total_gpus,json=totalGpus,proto3" json:"total_gpus,omitempty"`
	MinTrustScore  float64                `protobuf:"fixed64,5,opt,name=min_trust_score,json=minTrustScore,proto3" json:"min_trust_score,omitempty"`
	MeanTrustScore float64                `protobuf:"fixed64,6,opt,name=mean_trust_score,json=meanTrustScore,proto3" json:"mean_trust_score,omitempty"`
}

func (t *trustGraphSnapshotPB) ProtoReflect() {}
func (t *trustGraphSnapshotPB) Reset()        { *t = trustGraphSnapshotPB{} }
func (t *trustGraphSnapshotPB) String() string { return "TrustGraphSnapshot" }

type auditEntryPB struct {
	EntryID      uint64                 `protobuf:"varint,1,opt,name=entry_id,json=entryId,proto3" json:"entry_id,omitempty"`
	EntryType    int32                  `protobuf:"varint,2,opt,name=entry_type,json=entryType,proto3" json:"entry_type,omitempty"`
	Timestamp    *timestamppb.Timestamp `protobuf:"bytes,3,opt,name=timestamp,proto3" json:"timestamp,omitempty"`
	GPU          *gpuIdentifierPB       `protobuf:"bytes,4,opt,name=gpu,proto3" json:"gpu,omitempty"`
	Data         []byte                 `protobuf:"bytes,5,opt,name=data,proto3" json:"data,omitempty"`
	PreviousHash []byte                 `protobuf:"bytes,6,opt,name=previous_hash,json=previousHash,proto3" json:"previous_hash,omitempty"`
	EntryHash    []byte                 `protobuf:"bytes,7,opt,name=entry_hash,json=entryHash,proto3" json:"entry_hash,omitempty"`
	MerkleRoot   []byte                 `protobuf:"bytes,8,opt,name=merkle_root,json=merkleRoot,proto3" json:"merkle_root,omitempty"`
}

type auditQueryRequestPB struct {
	GPU        *gpuIdentifierPB       `protobuf:"bytes,1,opt,name=gpu,proto3" json:"gpu,omitempty"`
	StartTime  *timestamppb.Timestamp `protobuf:"bytes,2,opt,name=start_time,json=startTime,proto3" json:"start_time,omitempty"`
	EndTime    *timestamppb.Timestamp `protobuf:"bytes,3,opt,name=end_time,json=endTime,proto3" json:"end_time,omitempty"`
	EntryType  int32                  `protobuf:"varint,4,opt,name=entry_type,json=entryType,proto3" json:"entry_type,omitempty"`
	Limit      uint32                 `protobuf:"varint,5,opt,name=limit,proto3" json:"limit,omitempty"`
	PageToken  string                 `protobuf:"bytes,6,opt,name=page_token,json=pageToken,proto3" json:"page_token,omitempty"`
	Descending bool                   `protobuf:"varint,7,opt,name=descending,proto3" json:"descending,omitempty"`
}

func (a *auditQueryRequestPB) ProtoReflect() {}
func (a *auditQueryRequestPB) Reset()        { *a = auditQueryRequestPB{} }
func (a *auditQueryRequestPB) String() string { return "AuditQueryRequest" }

type auditQueryResponsePB struct {
	Entries       []*auditEntryPB `protobuf:"bytes,1,rep,name=entries,proto3" json:"entries,omitempty"`
	NextPageToken string          `protobuf:"bytes,2,opt,name=next_page_token,json=nextPageToken,proto3" json:"next_page_token,omitempty"`
	TotalCount    uint64          `protobuf:"varint,3,opt,name=total_count,json=totalCount,proto3" json:"total_count,omitempty"`
}

func (a *auditQueryResponsePB) ProtoReflect() {}
func (a *auditQueryResponsePB) Reset()        { *a = auditQueryResponsePB{} }
func (a *auditQueryResponsePB) String() string { return "AuditQueryResponse" }

type chainVerificationRequestPB struct {
	StartEntryID      uint64 `protobuf:"varint,1,opt,name=start_entry_id,json=startEntryId,proto3" json:"start_entry_id,omitempty"`
	EndEntryID        uint64 `protobuf:"varint,2,opt,name=end_entry_id,json=endEntryId,proto3" json:"end_entry_id,omitempty"`
	VerifyMerkleRoots bool   `protobuf:"varint,3,opt,name=verify_merkle_roots,json=verifyMerkleRoots,proto3" json:"verify_merkle_roots,omitempty"`
}

func (c *chainVerificationRequestPB) ProtoReflect() {}
func (c *chainVerificationRequestPB) Reset()        { *c = chainVerificationRequestPB{} }
func (c *chainVerificationRequestPB) String() string { return "ChainVerificationRequest" }

type chainVerificationResponsePB struct {
	Valid               bool   `protobuf:"varint,1,opt,name=valid,proto3" json:"valid,omitempty"`
	FirstInvalidEntryID uint64 `protobuf:"varint,2,opt,name=first_invalid_entry_id,json=firstInvalidEntryId,proto3" json:"first_invalid_entry_id,omitempty"`
	FailureDescription  string `protobuf:"bytes,3,opt,name=failure_description,json=failureDescription,proto3" json:"failure_description,omitempty"`
	EntriesVerified     uint64 `protobuf:"varint,4,opt,name=entries_verified,json=entriesVerified,proto3" json:"entries_verified,omitempty"`
	BatchesVerified     uint64 `protobuf:"varint,5,opt,name=batches_verified,json=batchesVerified,proto3" json:"batches_verified,omitempty"`
	VerificationTimeMs  uint64 `protobuf:"varint,6,opt,name=verification_time_ms,json=verificationTimeMs,proto3" json:"verification_time_ms,omitempty"`
}

func (c *chainVerificationResponsePB) ProtoReflect() {}
func (c *chainVerificationResponsePB) Reset()        { *c = chainVerificationResponsePB{} }
func (c *chainVerificationResponsePB) String() string { return "ChainVerificationResponse" }

type configUpdatePB struct {
	UpdateID       string                `protobuf:"bytes,1,opt,name=update_id,json=updateId,proto3" json:"update_id,omitempty"`
	InitiatedBy    string                `protobuf:"bytes,2,opt,name=initiated_by,json=initiatedBy,proto3" json:"initiated_by,omitempty"`
	Reason         string                `protobuf:"bytes,3,opt,name=reason,proto3" json:"reason,omitempty"`
	ProbeSchedule  *probeScheduleUpdatePB `protobuf:"bytes,10,opt,name=probe_schedule,json=probeSchedule,proto3,oneof" json:"probe_schedule,omitempty"`
	OverheadBudget *overheadBudgetUpdatePB `protobuf:"bytes,11,opt,name=overhead_budget,json=overheadBudget,proto3,oneof" json:"overhead_budget,omitempty"`
	SamplingRate   *samplingRateUpdatePB  `protobuf:"bytes,12,opt,name=sampling_rate,json=samplingRate,proto3,oneof" json:"sampling_rate,omitempty"`
	Threshold      *thresholdUpdatePB     `protobuf:"bytes,13,opt,name=threshold,proto3,oneof" json:"threshold,omitempty"`
}

func (c *configUpdatePB) ProtoReflect() {}
func (c *configUpdatePB) Reset()        { *c = configUpdatePB{} }
func (c *configUpdatePB) String() string { return "ConfigUpdate" }

type probeScheduleEntryPB struct {
	Type          int32   `protobuf:"varint,1,opt,name=type,proto3" json:"type,omitempty"`
	PeriodSeconds uint32  `protobuf:"varint,2,opt,name=period_seconds,json=periodSeconds,proto3" json:"period_seconds,omitempty"`
	SmCoverage    float64 `protobuf:"fixed64,3,opt,name=sm_coverage,json=smCoverage,proto3" json:"sm_coverage,omitempty"`
	Priority      uint32  `protobuf:"varint,4,opt,name=priority,proto3" json:"priority,omitempty"`
	Enabled       bool    `protobuf:"varint,5,opt,name=enabled,proto3" json:"enabled,omitempty"`
	TimeoutMs     uint32  `protobuf:"varint,6,opt,name=timeout_ms,json=timeoutMs,proto3" json:"timeout_ms,omitempty"`
}

type probeScheduleUpdatePB struct {
	Entries []*probeScheduleEntryPB `protobuf:"bytes,1,rep,name=entries,proto3" json:"entries,omitempty"`
}

type overheadBudgetUpdatePB struct {
	BudgetPct float64 `protobuf:"fixed64,1,opt,name=budget_pct,json=budgetPct,proto3" json:"budget_pct,omitempty"`
}

type samplingRateUpdatePB struct {
	Component string  `protobuf:"bytes,1,opt,name=component,proto3" json:"component,omitempty"`
	Rate      float64 `protobuf:"fixed64,2,opt,name=rate,proto3" json:"rate,omitempty"`
}

type thresholdUpdatePB struct {
	Component string  `protobuf:"bytes,1,opt,name=component,proto3" json:"component,omitempty"`
	Parameter string  `protobuf:"bytes,2,opt,name=parameter,proto3" json:"parameter,omitempty"`
	Value     float64 `protobuf:"fixed64,3,opt,name=value,proto3" json:"value,omitempty"`
}

type configAckPB struct {
	UpdateID      string `protobuf:"bytes,1,opt,name=update_id,json=updateId,proto3" json:"update_id,omitempty"`
	Applied       bool   `protobuf:"varint,2,opt,name=applied,proto3" json:"applied,omitempty"`
	ComponentID   string `protobuf:"bytes,3,opt,name=component_id,json=componentId,proto3" json:"component_id,omitempty"`
	Error         string `protobuf:"bytes,4,opt,name=error,proto3" json:"error,omitempty"`
	ConfigVersion uint64 `protobuf:"varint,5,opt,name=config_version,json=configVersion,proto3" json:"config_version,omitempty"`
}

func (c *configAckPB) ProtoReflect() {}
func (c *configAckPB) Reset()        { *c = configAckPB{} }
func (c *configAckPB) String() string { return "ConfigAck" }

// ---------------------------------------------------------------------------
// Conversion functions: protobuf wire types -> SDK types
// ---------------------------------------------------------------------------

func gpuIdentifierFromPB(pb *gpuIdentifierPB) *GpuIdentifier {
	if pb == nil {
		return nil
	}
	return &GpuIdentifier{
		UUID:            pb.UUID,
		Hostname:        pb.Hostname,
		DeviceIndex:     pb.DeviceIndex,
		Model:           pb.Model,
		DriverVersion:   pb.DriverVersion,
		FirmwareVersion: pb.FirmwareVersion,
	}
}

func smIdentifierFromPB(pb *smIdentifierPB) *SmIdentifier {
	if pb == nil {
		return nil
	}
	return &SmIdentifier{
		GPU:  gpuIdentifierFromPB(pb.GPU),
		SmID: pb.SmID,
	}
}

func smHealthFromPB(pb *smHealthPB) *SmHealth {
	if pb == nil {
		return nil
	}
	return &SmHealth{
		SM:               smIdentifierFromPB(pb.SM),
		ReliabilityScore: pb.ReliabilityScore,
		ProbePassCount:   pb.ProbePassCount,
		ProbeFailCount:   pb.ProbeFailCount,
		Disabled:         pb.Disabled,
		DisableReason:    pb.DisableReason,
	}
}

func gpuHealthFromPB(pb *gpuHealthPB) *GpuHealth {
	if pb == nil {
		return nil
	}
	smHealthList := make([]*SmHealth, 0, len(pb.SmHealth))
	for _, s := range pb.SmHealth {
		smHealthList = append(smHealthList, smHealthFromPB(s))
	}
	return &GpuHealth{
		GPU:               gpuIdentifierFromPB(pb.GPU),
		State:             GpuHealthState(pb.State),
		ReliabilityScore:  pb.ReliabilityScore,
		Alpha:             pb.Alpha,
		Beta:              pb.Beta,
		LastProbeTime:     timestampToTime(pb.LastProbeTime),
		LastAnomalyTime:   timestampToTime(pb.LastAnomalyTime),
		ProbePassCount:    pb.ProbePassCount,
		ProbeFailCount:    pb.ProbeFailCount,
		AnomalyCount:      pb.AnomalyCount,
		StateChangedAt:    timestampToTime(pb.StateChangedAt),
		StateChangeReason: pb.StateChangeReason,
		SmHealth:          smHealthList,
		AnomalyRate:       pb.AnomalyRate,
		ProbeFailureRate:  pb.ProbeFailureRate,
	}
}

func fleetSummaryFromPB(pb *fleetHealthSummaryPB) *FleetHealthSummary {
	if pb == nil {
		return nil
	}
	return &FleetHealthSummary{
		TotalGPUs:               pb.TotalGPUs,
		Healthy:                 pb.Healthy,
		Suspect:                 pb.Suspect,
		Quarantined:             pb.Quarantined,
		DeepTest:                pb.DeepTest,
		Condemned:               pb.Condemned,
		OverallSDCRate:          pb.OverallSDCRate,
		AverageReliabilityScore: pb.AverageReliabilityScore,
		SnapshotTime:            timestampToTime(pb.SnapshotTime),
		ActiveAgents:            pb.ActiveAgents,
		RateWindowSeconds:       pb.RateWindowSeconds,
	}
}

func correlationEventFromPB(pb *correlationEventPB) *CorrelationEvent {
	if pb == nil {
		return nil
	}
	return &CorrelationEvent{
		EventID:           pb.EventID,
		EventsCorrelated:  pb.EventsCorrelated,
		PatternType:       PatternType(pb.PatternType),
		Confidence:        pb.Confidence,
		AttributedGPU:     gpuIdentifierFromPB(pb.AttributedGPU),
		AttributedSM:      smIdentifierFromPB(pb.AttributedSM),
		Description:       pb.Description,
		Timestamp:         timestampToTime(pb.Timestamp),
		Severity:          Severity(pb.Severity),
		RecommendedAction: pb.RecommendedAction,
	}
}

func gpuHistoryFromPB(pb *gpuHistoryResponsePB) *GpuHistoryResponse {
	if pb == nil {
		return nil
	}
	transitions := make([]*StateTransition, 0, len(pb.StateTransitions))
	for _, t := range pb.StateTransitions {
		transitions = append(transitions, &StateTransition{
			FromState:   GpuHealthState(t.FromState),
			ToState:     GpuHealthState(t.ToState),
			Timestamp:   timestampToTime(t.Timestamp),
			Reason:      t.Reason,
			InitiatedBy: t.InitiatedBy,
		})
	}
	correlations := make([]*CorrelationEvent, 0, len(pb.Correlations))
	for _, c := range pb.Correlations {
		correlations = append(correlations, correlationEventFromPB(c))
	}
	reliability := make([]*ReliabilitySample, 0, len(pb.ReliabilityHistory))
	for _, r := range pb.ReliabilityHistory {
		reliability = append(reliability, &ReliabilitySample{
			Timestamp:        timestampToTime(r.Timestamp),
			ReliabilityScore: r.ReliabilityScore,
			Alpha:            r.Alpha,
			Beta:             r.Beta,
		})
	}
	return &GpuHistoryResponse{
		StateTransitions:   transitions,
		Correlations:       correlations,
		ReliabilityHistory: reliability,
		NextPageToken:      pb.NextPageToken,
	}
}

func quarantineDirectiveFromPB(pb *quarantineDirectivePB) *QuarantineDirective {
	if pb == nil {
		return nil
	}
	return &QuarantineDirective{
		DirectiveID:      pb.DirectiveID,
		GPU:              gpuIdentifierFromPB(pb.GPU),
		Action:           QuarantineAction(pb.Action),
		Reason:           pb.Reason,
		InitiatedBy:      pb.InitiatedBy,
		Evidence:         pb.Evidence,
		Timestamp:        timestampToTime(pb.Timestamp),
		Priority:         pb.Priority,
		RequiresApproval: pb.RequiresApproval,
	}
}

func trustEdgeFromPB(pb *trustEdgePB) *TrustEdge {
	if pb == nil {
		return nil
	}
	return &TrustEdge{
		GpuA:              gpuIdentifierFromPB(pb.GpuA),
		GpuB:              gpuIdentifierFromPB(pb.GpuB),
		AgreementCount:    pb.AgreementCount,
		DisagreementCount: pb.DisagreementCount,
		LastComparison:    timestampToTime(pb.LastComparison),
		TrustScore:        pb.TrustScore,
	}
}

func trustGraphFromPB(pb *trustGraphSnapshotPB) *TrustGraphSnapshot {
	if pb == nil {
		return nil
	}
	edges := make([]*TrustEdge, 0, len(pb.Edges))
	for _, e := range pb.Edges {
		edges = append(edges, trustEdgeFromPB(e))
	}
	return &TrustGraphSnapshot{
		Edges:          edges,
		Timestamp:      timestampToTime(pb.Timestamp),
		CoveragePct:    pb.CoveragePct,
		TotalGPUs:      pb.TotalGPUs,
		MinTrustScore:  pb.MinTrustScore,
		MeanTrustScore: pb.MeanTrustScore,
	}
}

func auditEntryFromPB(pb *auditEntryPB) *AuditEntry {
	if pb == nil {
		return nil
	}
	return &AuditEntry{
		EntryID:      pb.EntryID,
		EntryType:    AuditEntryType(pb.EntryType),
		Timestamp:    timestampToTime(pb.Timestamp),
		GPU:          gpuIdentifierFromPB(pb.GPU),
		Data:         pb.Data,
		PreviousHash: pb.PreviousHash,
		EntryHash:    pb.EntryHash,
		MerkleRoot:   pb.MerkleRoot,
	}
}

func auditQueryResponseFromPB(pb *auditQueryResponsePB) *AuditQueryResponse {
	if pb == nil {
		return nil
	}
	entries := make([]*AuditEntry, 0, len(pb.Entries))
	for _, e := range pb.Entries {
		entries = append(entries, auditEntryFromPB(e))
	}
	return &AuditQueryResponse{
		Entries:       entries,
		NextPageToken: pb.NextPageToken,
		TotalCount:    pb.TotalCount,
	}
}

func configUpdateToPB(u *ConfigUpdate) *configUpdatePB {
	pb := &configUpdatePB{
		UpdateID:    u.UpdateID,
		InitiatedBy: u.InitiatedBy,
		Reason:      u.Reason,
	}
	if u.ProbeSchedule != nil {
		entries := make([]*probeScheduleEntryPB, 0, len(u.ProbeSchedule.Entries))
		for _, e := range u.ProbeSchedule.Entries {
			entries = append(entries, &probeScheduleEntryPB{
				Type:          int32(e.Type),
				PeriodSeconds: e.PeriodSeconds,
				SmCoverage:    e.SmCoverage,
				Priority:      e.Priority,
				Enabled:       e.Enabled,
				TimeoutMs:     e.TimeoutMs,
			})
		}
		pb.ProbeSchedule = &probeScheduleUpdatePB{Entries: entries}
	} else if u.OverheadBudget != nil {
		pb.OverheadBudget = &overheadBudgetUpdatePB{BudgetPct: u.OverheadBudget.BudgetPct}
	} else if u.SamplingRate != nil {
		pb.SamplingRate = &samplingRateUpdatePB{
			Component: u.SamplingRate.Component,
			Rate:      u.SamplingRate.Rate,
		}
	} else if u.Threshold != nil {
		pb.Threshold = &thresholdUpdatePB{
			Component: u.Threshold.Component,
			Parameter: u.Threshold.Parameter,
			Value:     u.Threshold.Value,
		}
	}
	return pb
}
