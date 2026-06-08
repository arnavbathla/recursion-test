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
import type { TrainingHistory } from "@/lib/types";

export function TrainingChart({ training }: { training: TrainingHistory }) {
  const history = training.history || [];
  const data = history.map((h) => ({
    epoch: h.epoch,
    train_loss: Number(h.train_loss),
    grad_norm: h.grad_norm != null ? Number(h.grad_norm) : undefined,
    weight_update_l2: h.weight_update_l2 != null ? Number(h.weight_update_l2) : undefined,
  }));

  const changed = training.weights_changed;
  const movement = training.total_weight_movement_l2;

  return (
    <div className="panel p-4">
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-sm font-semibold">Training dynamics</h3>
        <span
          className="text-xs px-2 py-1 rounded"
          style={{
            background: changed ? "rgba(46,204,64,0.15)" : "rgba(255,65,54,0.15)",
            color: changed ? "#2ecc40" : "#ff4136",
            border: `1px solid ${changed ? "#2ecc40" : "#ff4136"}`,
          }}
        >
          {changed ? "weights updated" : "no weight change"}
          {movement != null ? ` · ΣΔ L2 = ${movement.toFixed(4)}` : ""}
        </span>
      </div>

      {data.length === 0 ? (
        <p className="text-sm text-[#9a9aa3]">No training history recorded.</p>
      ) : (
        <>
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            <div>
              <p className="text-xs text-[#9a9aa3] mb-1">Loss &amp; gradient norm per epoch</p>
              <div style={{ width: "100%", height: 240 }}>
                <ResponsiveContainer>
                  <LineChart data={data} margin={{ top: 10, right: 30, bottom: 10, left: 0 }}>
                    <CartesianGrid stroke="#26262b" />
                    <XAxis
                      dataKey="epoch"
                      stroke="#9a9aa3"
                      label={{ value: "epoch", position: "insideBottom", offset: -5, fill: "#9a9aa3" }}
                    />
                    <YAxis yAxisId="left" stroke="#0074d9" />
                    <YAxis yAxisId="right" orientation="right" stroke="#ff851b" />
                    <Tooltip contentStyle={{ background: "#141417", border: "1px solid #26262b" }} />
                    <Legend />
                    <Line
                      yAxisId="left"
                      type="monotone"
                      dataKey="train_loss"
                      name="train loss"
                      stroke="#0074d9"
                      strokeWidth={2}
                      dot
                    />
                    <Line
                      yAxisId="right"
                      type="monotone"
                      dataKey="grad_norm"
                      name="grad norm"
                      stroke="#ff851b"
                      strokeWidth={2}
                      dot
                    />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </div>

            <div>
              <p className="text-xs text-[#9a9aa3] mb-1">
                Weight update magnitude ‖θ&#8202;ₜ − θ&#8202;ₜ₋₁‖₂ per epoch
              </p>
              <div style={{ width: "100%", height: 240 }}>
                <ResponsiveContainer>
                  <LineChart data={data} margin={{ top: 10, right: 30, bottom: 10, left: 0 }}>
                    <CartesianGrid stroke="#26262b" />
                    <XAxis
                      dataKey="epoch"
                      stroke="#9a9aa3"
                      label={{ value: "epoch", position: "insideBottom", offset: -5, fill: "#9a9aa3" }}
                    />
                    <YAxis stroke="#2ecc40" domain={[0, "auto"]} />
                    <Tooltip contentStyle={{ background: "#141417", border: "1px solid #26262b" }} />
                    <Legend />
                    <Line
                      type="monotone"
                      dataKey="weight_update_l2"
                      name="weight Δ (L2)"
                      stroke="#2ecc40"
                      strokeWidth={2}
                      dot
                    />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </div>
          </div>

          <table className="w-full text-xs mt-3">
            <thead className="text-[#9a9aa3]">
              <tr>
                <th className="text-left py-1">epoch</th>
                <th className="text-right py-1">train loss</th>
                <th className="text-right py-1">grad norm</th>
                <th className="text-right py-1">weight Δ (L2)</th>
                <th className="text-right py-1">‖θ‖</th>
              </tr>
            </thead>
            <tbody>
              {history.map((h) => (
                <tr key={h.epoch} className="border-t border-[#26262b]">
                  <td className="py-1">{h.epoch}</td>
                  <td className="text-right py-1">{Number(h.train_loss).toFixed(4)}</td>
                  <td className="text-right py-1">
                    {h.grad_norm != null ? Number(h.grad_norm).toFixed(4) : "—"}
                  </td>
                  <td className="text-right py-1">
                    {h.weight_update_l2 != null ? Number(h.weight_update_l2).toFixed(4) : "—"}
                  </td>
                  <td className="text-right py-1">
                    {h.param_global_norm != null ? Number(h.param_global_norm).toFixed(2) : "—"}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </>
      )}
    </div>
  );
}
