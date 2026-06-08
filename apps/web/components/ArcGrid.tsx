import type { Grid } from "@/lib/types";

// Canonical ARC color palette (index 0-9).
export const ARC_COLORS = [
  "#000000", // 0 black
  "#0074d9", // 1 blue
  "#ff4136", // 2 red
  "#2ecc40", // 3 green
  "#ffdc00", // 4 yellow
  "#aaaaaa", // 5 gray
  "#f012be", // 6 magenta
  "#ff851b", // 7 orange
  "#7fdbff", // 8 cyan
  "#870c25", // 9 maroon
];

export function ArcGrid({
  grid,
  cell = 16,
  label,
}: {
  grid: Grid;
  cell?: number;
  label?: string;
}) {
  if (!grid || grid.length === 0) {
    return <div className="text-xs text-[var(--muted)]">empty</div>;
  }
  const cols = grid[0].length;
  return (
    <div>
      {label && <div className="text-xs text-[var(--muted)] mb-1">{label}</div>}
      <div
        className="inline-grid gap-px bg-[#26262b] p-px rounded"
        style={{ gridTemplateColumns: `repeat(${cols}, ${cell}px)` }}
      >
        {grid.flatMap((row, r) =>
          row.map((v, c) => (
            <div
              key={`${r}-${c}`}
              style={{
                width: cell,
                height: cell,
                background: ARC_COLORS[v] ?? "#000",
              }}
              title={`(${r},${c})=${v}`}
            />
          ))
        )}
      </div>
    </div>
  );
}
