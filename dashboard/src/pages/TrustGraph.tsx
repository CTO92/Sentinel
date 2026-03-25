import { useRef, useState, useCallback, useMemo, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import ForceGraph2D from "react-force-graph-2d";
import { ZoomIn, ZoomOut, Maximize2, Activity, GitGraph } from "lucide-react";
import { useTrustGraph } from "@/api/hooks";
import { GpuHealthState, type TrustGraphData } from "@/api/types";
import { cn, healthStateColor, formatPercent } from "@/lib/utils";
import { HealthBadge } from "@/components/HealthBadge";

function generateMockTrustGraph(): TrustGraphData {
  const states = Object.values(GpuHealthState);
  const nodeCount = 64;

  const nodes = Array.from({ length: nodeCount }, (_, i) => {
    const stateRoll = Math.random();
    const state =
      stateRoll < 0.75
        ? GpuHealthState.HEALTHY
        : stateRoll < 0.88
          ? GpuHealthState.SUSPECT
          : stateRoll < 0.94
            ? GpuHealthState.QUARANTINED
            : stateRoll < 0.97
              ? GpuHealthState.DEEP_TEST
              : GpuHealthState.CONDEMNED;

    return {
      id: `GPU-${String(i).padStart(4, "0")}`,
      label: `GPU ${i}`,
      state,
      reliability_score:
        state === GpuHealthState.HEALTHY
          ? 0.95 + Math.random() * 0.05
          : 0.3 + Math.random() * 0.5,
      hostname: `gpu-node-${String(Math.floor(i / 8)).padStart(3, "0")}`,
    };
  });

  const edges = [];
  for (let i = 0; i < nodeCount; i++) {
    const numEdges = 2 + Math.floor(Math.random() * 4);
    for (let j = 0; j < numEdges; j++) {
      const target = Math.floor(Math.random() * nodeCount);
      if (target !== i) {
        edges.push({
          source_gpu: nodes[i].id,
          target_gpu: nodes[target].id,
          trust_score: 0.5 + Math.random() * 0.5,
          comparisons: 10 + Math.floor(Math.random() * 100),
          last_compared: new Date(
            Date.now() - Math.random() * 86400000,
          ).toISOString(),
          agreement_rate: 0.9 + Math.random() * 0.1,
        });
      }
    }
  }

  return {
    nodes,
    edges,
    coverage_percent: 94.2,
    total_comparisons: 15847,
  };
}

interface GraphNode {
  id: string;
  label: string;
  state: GpuHealthState;
  reliability_score: number;
  hostname: string;
  x?: number;
  y?: number;
}

interface GraphLink {
  source: string | GraphNode;
  target: string | GraphNode;
  trust_score: number;
  comparisons: number;
  agreement_rate: number;
}

export default function TrustGraph() {
  const navigate = useNavigate();
  const graphRef = useRef<any>(null); // eslint-disable-line @typescript-eslint/no-explicit-any
  const containerRef = useRef<HTMLDivElement>(null);
  const { data: trustData } = useTrustGraph();
  const [hoveredNode, setHoveredNode] = useState<GraphNode | null>(null);
  const [dimensions, setDimensions] = useState({ width: 800, height: 600 });

  const graphData = useMemo(() => trustData ?? generateMockTrustGraph(), [trustData]);

  const forceGraphData = useMemo(() => {
    const nodes: GraphNode[] = graphData.nodes.map((n) => ({ ...n }));
    const links: GraphLink[] = graphData.edges.map((e) => ({
      source: e.source_gpu,
      target: e.target_gpu,
      trust_score: e.trust_score,
      comparisons: e.comparisons,
      agreement_rate: e.agreement_rate,
    }));
    return { nodes, links };
  }, [graphData]);

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;
    const resizeObserver = new ResizeObserver((entries) => {
      for (const entry of entries) {
        setDimensions({
          width: entry.contentRect.width,
          height: Math.max(entry.contentRect.height, 500),
        });
      }
    });
    resizeObserver.observe(container);
    return () => resizeObserver.disconnect();
  }, []);

  const handleNodeClick = useCallback(
    (node: GraphNode) => {
      navigate(`/gpu/${node.id}`);
    },
    [navigate],
  );

  const handleNodeHover = useCallback((node: GraphNode | null) => {
    setHoveredNode(node);
  }, []);

  const drawNode = useCallback(
    (node: GraphNode, ctx: CanvasRenderingContext2D, globalScale: number) => {
      const radius = 6;
      const x = node.x ?? 0;
      const y = node.y ?? 0;

      // Node circle
      ctx.beginPath();
      ctx.arc(x, y, radius, 0, 2 * Math.PI);
      ctx.fillStyle = healthStateColor(node.state);
      ctx.fill();

      // Border for hovered node
      if (hoveredNode && hoveredNode.id === node.id) {
        ctx.strokeStyle = "#ffffff";
        ctx.lineWidth = 2;
        ctx.stroke();
      }

      // Label when zoomed in
      if (globalScale > 1.5) {
        ctx.font = `${10 / globalScale}px monospace`;
        ctx.textAlign = "center";
        ctx.textBaseline = "top";
        ctx.fillStyle = "hsl(210, 40%, 75%)";
        ctx.fillText(node.id.slice(0, 8), x, y + radius + 2);
      }
    },
    [hoveredNode],
  );

  const drawLink = useCallback(
    (link: GraphLink, ctx: CanvasRenderingContext2D) => {
      const source = link.source as GraphNode;
      const target = link.target as GraphNode;
      if (!source.x || !source.y || !target.x || !target.y) return;

      ctx.beginPath();
      ctx.moveTo(source.x, source.y);
      ctx.lineTo(target.x, target.y);
      ctx.strokeStyle = `rgba(100, 150, 255, ${link.trust_score * 0.4})`;
      ctx.lineWidth = Math.max(0.5, link.trust_score * 2);
      ctx.stroke();
    },
    [],
  );

  const handleZoomIn = () => graphRef.current?.zoom(1.5, 300);
  const handleZoomOut = () => graphRef.current?.zoom(0.67, 300);
  const handleFit = () => graphRef.current?.zoomToFit(400, 50);

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold tracking-tight">Trust Graph</h1>
          <p className="text-sm text-muted-foreground mt-1">
            Cross-GPU trust relationships from comparative probes
          </p>
        </div>
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-2 rounded-full bg-primary/10 px-4 py-1.5 text-sm font-medium text-primary border border-primary/30">
            <GitGraph className="h-4 w-4" />
            Coverage: {graphData.coverage_percent.toFixed(1)}%
          </div>
          <div className="text-xs text-muted-foreground">
            {graphData.total_comparisons.toLocaleString()} comparisons
          </div>
        </div>
      </div>

      {/* Graph container */}
      <div className="relative rounded-lg border border-border bg-card overflow-hidden">
        {/* Controls */}
        <div className="absolute top-3 right-3 z-10 flex flex-col gap-1">
          <button
            onClick={handleZoomIn}
            className="rounded-md bg-card/80 border border-border p-2 hover:bg-muted transition-colors backdrop-blur-sm"
            title="Zoom in"
          >
            <ZoomIn className="h-4 w-4" />
          </button>
          <button
            onClick={handleZoomOut}
            className="rounded-md bg-card/80 border border-border p-2 hover:bg-muted transition-colors backdrop-blur-sm"
            title="Zoom out"
          >
            <ZoomOut className="h-4 w-4" />
          </button>
          <button
            onClick={handleFit}
            className="rounded-md bg-card/80 border border-border p-2 hover:bg-muted transition-colors backdrop-blur-sm"
            title="Fit to view"
          >
            <Maximize2 className="h-4 w-4" />
          </button>
        </div>

        {/* Legend */}
        <div className="absolute bottom-3 left-3 z-10 flex items-center gap-3 rounded-md bg-card/80 border border-border px-3 py-2 backdrop-blur-sm">
          {Object.values(GpuHealthState).map((state) => (
            <span
              key={state}
              className="flex items-center gap-1.5 text-[10px] text-muted-foreground"
            >
              <span
                className="h-2.5 w-2.5 rounded-full"
                style={{ backgroundColor: healthStateColor(state) }}
              />
              {state.replace(/_/g, " ")}
            </span>
          ))}
        </div>

        {/* Hover tooltip */}
        {hoveredNode && (
          <div className="absolute top-3 left-3 z-10 rounded-md bg-card/90 border border-border px-4 py-3 backdrop-blur-sm shadow-lg">
            <div className="flex items-center gap-2 mb-2">
              <span className="font-mono text-sm font-medium">
                {hoveredNode.id}
              </span>
              <HealthBadge state={hoveredNode.state} size="sm" />
            </div>
            <div className="space-y-1 text-xs text-muted-foreground">
              <p>Host: {hoveredNode.hostname}</p>
              <p>Reliability: {formatPercent(hoveredNode.reliability_score)}</p>
              <p className="text-[10px] mt-2 text-primary">
                Click to view details
              </p>
            </div>
          </div>
        )}

        {/* Graph */}
        <div ref={containerRef} className="h-[600px] w-full">
          <ForceGraph2D
            ref={graphRef}
            graphData={forceGraphData}
            width={dimensions.width}
            height={dimensions.height}
            nodeCanvasObject={drawNode}
            linkCanvasObject={drawLink}
            onNodeClick={handleNodeClick}
            onNodeHover={handleNodeHover}
            nodeRelSize={6}
            linkDirectionalParticles={0}
            cooldownTicks={100}
            backgroundColor="transparent"
            enableZoomInteraction={true}
            enablePanInteraction={true}
          />
        </div>
      </div>

      {/* Stats table */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div className="rounded-lg border border-border bg-card p-4">
          <h3 className="text-sm font-semibold mb-3">Node Distribution</h3>
          <div className="space-y-2">
            {Object.values(GpuHealthState).map((state) => {
              const count = graphData.nodes.filter(
                (n) => n.state === state,
              ).length;
              const pct = (count / graphData.nodes.length) * 100;
              return (
                <div key={state} className="flex items-center gap-3">
                  <span
                    className="h-2.5 w-2.5 rounded-full flex-shrink-0"
                    style={{ backgroundColor: healthStateColor(state) }}
                  />
                  <span className="text-xs flex-1">
                    {state.replace(/_/g, " ")}
                  </span>
                  <span className="text-xs font-mono">{count}</span>
                  <div className="w-24 h-1.5 rounded-full bg-muted overflow-hidden">
                    <div
                      className="h-full rounded-full"
                      style={{
                        width: `${pct}%`,
                        backgroundColor: healthStateColor(state),
                      }}
                    />
                  </div>
                  <span className="text-[10px] text-muted-foreground w-10 text-right">
                    {pct.toFixed(1)}%
                  </span>
                </div>
              );
            })}
          </div>
        </div>

        <div className="rounded-lg border border-border bg-card p-4">
          <h3 className="text-sm font-semibold mb-3">Trust Statistics</h3>
          <div className="space-y-3 text-sm">
            <div className="flex items-center justify-between">
              <span className="text-muted-foreground">Total Nodes</span>
              <span className="font-mono font-medium">{graphData.nodes.length}</span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-muted-foreground">Total Edges</span>
              <span className="font-mono font-medium">{graphData.edges.length}</span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-muted-foreground">Coverage</span>
              <span className="font-mono font-medium">
                {graphData.coverage_percent.toFixed(1)}%
              </span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-muted-foreground">Avg Trust Score</span>
              <span className="font-mono font-medium">
                {(
                  graphData.edges.reduce((s, e) => s + e.trust_score, 0) /
                  Math.max(graphData.edges.length, 1)
                ).toFixed(3)}
              </span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-muted-foreground">Total Comparisons</span>
              <span className="font-mono font-medium">
                {graphData.total_comparisons.toLocaleString()}
              </span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
