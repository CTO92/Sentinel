import {
  ResponsiveContainer,
  LineChart,
  Line,
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
} from "recharts";
import { format, parseISO } from "date-fns";
import { cn } from "@/lib/utils";

interface SeriesConfig {
  dataKey: string;
  name: string;
  color: string;
  type?: "line" | "area";
  strokeDasharray?: string;
}

interface TimeSeriesChartProps {
  data: Array<Record<string, unknown>>;
  series: SeriesConfig[];
  timeKey?: string;
  height?: number;
  yAxisLabel?: string;
  yDomain?: [number, number];
  formatYTick?: (value: number) => string;
  className?: string;
  chartType?: "line" | "area";
}

function formatTimeTick(value: string): string {
  try {
    return format(parseISO(value), "HH:mm");
  } catch {
    return value;
  }
}

function formatTooltipTime(value: string): string {
  try {
    return format(parseISO(value), "MMM dd, HH:mm:ss");
  } catch {
    return value;
  }
}

export function TimeSeriesChart({
  data,
  series,
  timeKey = "timestamp",
  height = 300,
  yAxisLabel,
  yDomain,
  formatYTick,
  className,
  chartType = "line",
}: TimeSeriesChartProps) {
  const Chart = chartType === "area" ? AreaChart : LineChart;

  return (
    <div className={cn("w-full", className)}>
      <ResponsiveContainer width="100%" height={height}>
        <Chart data={data} margin={{ top: 5, right: 20, bottom: 5, left: 10 }}>
          <CartesianGrid
            strokeDasharray="3 3"
            stroke="hsl(var(--border))"
            opacity={0.5}
          />
          <XAxis
            dataKey={timeKey}
            tickFormatter={formatTimeTick}
            stroke="hsl(var(--muted-foreground))"
            fontSize={11}
            tickLine={false}
          />
          <YAxis
            domain={yDomain}
            tickFormatter={formatYTick}
            label={
              yAxisLabel
                ? {
                    value: yAxisLabel,
                    angle: -90,
                    position: "insideLeft",
                    style: {
                      fill: "hsl(var(--muted-foreground))",
                      fontSize: 11,
                    },
                  }
                : undefined
            }
            stroke="hsl(var(--muted-foreground))"
            fontSize={11}
            tickLine={false}
          />
          <Tooltip
            labelFormatter={formatTooltipTime}
            contentStyle={{
              backgroundColor: "hsl(var(--popover))",
              border: "1px solid hsl(var(--border))",
              borderRadius: "var(--radius)",
              color: "hsl(var(--popover-foreground))",
              fontSize: 12,
            }}
          />
          <Legend
            wrapperStyle={{ fontSize: 12 }}
          />
          {series.map((s) =>
            chartType === "area" || s.type === "area" ? (
              <Area
                key={s.dataKey}
                type="monotone"
                dataKey={s.dataKey}
                name={s.name}
                stroke={s.color}
                fill={s.color}
                fillOpacity={0.1}
                strokeWidth={2}
                dot={false}
                strokeDasharray={s.strokeDasharray}
              />
            ) : (
              <Line
                key={s.dataKey}
                type="monotone"
                dataKey={s.dataKey}
                name={s.name}
                stroke={s.color}
                strokeWidth={2}
                dot={false}
                strokeDasharray={s.strokeDasharray}
              />
            ),
          )}
        </Chart>
      </ResponsiveContainer>
    </div>
  );
}
