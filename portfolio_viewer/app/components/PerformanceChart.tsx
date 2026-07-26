"use client";

import { selectRange } from "../portfolio-state.mjs";
import type {
  MarketPayload,
  PerformancePoint,
  RangeKey,
} from "../portfolio-types";

const WIDTH = 1000;
const HEIGHT = 360;
const PAD = { top: 24, right: 118, bottom: 38, left: 56 };
const COLORS = ["#4b84ff", "#f2a65a", "#a4a4a4"];

function pathFor(
  points: PerformancePoint[],
  field: string,
  min: number,
  max: number,
) {
  const range = max - min || 1;
  return points
    .map((point, index) => {
      const x =
        PAD.left +
        (index / Math.max(points.length - 1, 1)) *
          (WIDTH - PAD.left - PAD.right);
      const value = Number(point[field]);
      const y =
        PAD.top +
        ((max - value) / range) * (HEIGHT - PAD.top - PAD.bottom);
      return `${index === 0 ? "M" : "L"}${x.toFixed(2)},${y.toFixed(2)}`;
    })
    .join(" ");
}

function lastPosition(
  points: PerformancePoint[],
  field: string,
  min: number,
  max: number,
) {
  const value = Number(points.at(-1)?.[field] ?? 0);
  const range = max - min || 1;
  return {
    value,
    y:
      PAD.top +
      ((max - value) / range) * (HEIGHT - PAD.top - PAD.bottom),
  };
}

export function PerformanceChart({
  market,
  range,
}: {
  market: MarketPayload;
  range: RangeKey;
}) {
  const points = selectRange(market.performance, range, new Date(market.asOf));
  const series = [
    { id: "portfolio", label: "我的组合" },
    ...market.benchmarks,
  ];
  if (points.length < 2) {
    return <div className="empty-chart">所选区间暂无足够行情数据</div>;
  }

  const values = points.flatMap((point: PerformancePoint) =>
    series.map((item) => Number(point[item.id])),
  );
  const rawMin = Math.min(...values);
  const rawMax = Math.max(...values);
  const margin = Math.max((rawMax - rawMin) * 0.1, 1);
  const min = rawMin - margin;
  const max = rawMax + margin;
  const ticks = Array.from({ length: 5 }, (_, index) =>
    max - ((max - min) * index) / 4,
  );

  return (
    <div className="chart-wrap">
      <svg
        className="performance-chart"
        viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
        role="img"
        aria-label="投资组合与基准归一化收益曲线"
      >
        {ticks.map((tick, index) => {
          const y =
            PAD.top +
            (index / 4) * (HEIGHT - PAD.top - PAD.bottom);
          return (
            <g key={tick}>
              <line
                x1={PAD.left}
                y1={y}
                x2={WIDTH - PAD.right}
                y2={y}
                className="chart-grid"
              />
              <text x={PAD.left - 12} y={y + 4} className="axis-label">
                {tick.toFixed(0)}
              </text>
            </g>
          );
        })}
        {series.map((item, index) => {
          const last = lastPosition(points, item.id, min, max);
          return (
            <g key={item.id}>
              <path
                d={pathFor(points, item.id, min, max)}
                fill="none"
                stroke={COLORS[index]}
                strokeWidth={index === 0 ? 3 : 2}
                vectorEffect="non-scaling-stroke"
              />
              <circle
                cx={WIDTH - PAD.right}
                cy={last.y}
                r={index === 0 ? 4 : 3}
                fill={COLORS[index]}
              />
              <text
                x={WIDTH - PAD.right + 10}
                y={last.y + 4}
                fill={COLORS[index]}
                className="series-label"
              >
                {item.label} {last.value.toFixed(1)}
              </text>
            </g>
          );
        })}
        <text x={PAD.left} y={HEIGHT - 12} className="date-label">
          {points[0].date}
        </text>
        <text
          x={WIDTH - PAD.right}
          y={HEIGHT - 12}
          textAnchor="end"
          className="date-label"
        >
          {points.at(-1)?.date}
        </text>
      </svg>
    </div>
  );
}
