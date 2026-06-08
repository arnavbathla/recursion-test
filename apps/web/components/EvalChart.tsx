"use client";

import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { DepthMetrics } from "@/lib/types";

export interface Series {
  name: string;
  depths: Record<string, DepthMetrics>;
}

export function EvalChart({
  series,
  metric = "exact_match",
  title,
}: {
  series: Series[];
  metric?: keyof DepthMetrics;
  title?: string;
}) {
  const depthSet = new Set<number>();
  series.forEach((s) => Object.keys(s.depths).forEach((d) => depthSet.add(Number(d))));
  const depths = Array.from(depthSet).sort((a, b) => a - b);

  const data = depths.map((d) => {
    const row: Record<string, number> = { depth: d };
    series.forEach((s) => {
      const m = s.depths[String(d)];
      if (m) row[s.name] = Number(m[metric]);
    });
    return row;
  });

  const colors = ["#0074d9", "#ff851b", "#2ecc40", "#f012be"];

  return (
    <div className="panel p-4">
      {title && <h3 className="text-sm font-semibold mb-3">{title}</h3>}
      <div style={{ width: "100%", height: 320 }}>
        <ResponsiveContainer>
          <LineChart data={data} margin={{ top: 10, right: 20, bottom: 10, left: 0 }}>
            <CartesianGrid stroke="#26262b" />
            <XAxis dataKey="depth" stroke="#9a9aa3" label={{ value: "recursion depth", position: "insideBottom", offset: -5, fill: "#9a9aa3" }} />
            <YAxis stroke="#9a9aa3" />
            <Tooltip contentStyle={{ background: "#141417", border: "1px solid #26262b" }} />
            <Legend />
            {series.map((s, i) => (
              <Line
                key={s.name}
                type="monotone"
                dataKey={s.name}
                stroke={colors[i % colors.length]}
                strokeWidth={2}
                dot
              />
            ))}
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
