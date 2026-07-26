"use client";

import { useEffect, useRef, useState } from "react";

import type { ColumnPreference } from "../portfolio-types";

export function ColumnSettings({
  columns,
  onChange,
  onReset,
}: {
  columns: ColumnPreference[];
  onChange: (columns: ColumnPreference[]) => void;
  onReset: () => void;
}) {
  const [open, setOpen] = useState(false);
  const [dragged, setDragged] = useState<string | null>(null);
  const root = useRef<HTMLDivElement>(null);

  useEffect(() => {
    function close(event: PointerEvent) {
      if (!root.current?.contains(event.target as Node)) setOpen(false);
    }
    window.addEventListener("pointerdown", close);
    return () => window.removeEventListener("pointerdown", close);
  }, []);

  function move(id: string, offset: number) {
    const from = columns.findIndex((column) => column.id === id);
    const to = Math.max(1, Math.min(columns.length - 1, from + offset));
    if (from === to || from === 0) return;
    const next = [...columns];
    const [item] = next.splice(from, 1);
    next.splice(to, 0, item);
    onChange(next);
  }

  function drop(targetId: string) {
    if (!dragged || dragged === targetId || dragged === "symbol") return;
    const next = [...columns];
    const from = next.findIndex((column) => column.id === dragged);
    const to = Math.max(1, next.findIndex((column) => column.id === targetId));
    const [item] = next.splice(from, 1);
    next.splice(to, 0, item);
    onChange(next);
    setDragged(null);
  }

  return (
    <div className="column-settings" ref={root}>
      <button
        type="button"
        className="icon-button"
        aria-label="设置持仓列"
        aria-expanded={open}
        onClick={() => setOpen((value) => !value)}
      >
        <svg viewBox="0 0 24 24" aria-hidden="true">
          <path d="M4 5h10M18 5h2M4 12h2M10 12h10M4 19h7M15 19h5" />
          <circle cx="16" cy="5" r="2" />
          <circle cx="8" cy="12" r="2" />
          <circle cx="13" cy="19" r="2" />
        </svg>
      </button>
      {open && (
        <div className="settings-popover" role="dialog" aria-label="持仓列设置">
          <div className="popover-head">
            <div>
              <strong>持仓列</strong>
              <span>拖拽调整顺序</span>
            </div>
            <button type="button" className="text-button" onClick={onReset}>
              恢复默认
            </button>
          </div>
          <div className="column-list">
            {columns.map((column, index) => (
              <div
                className="column-option"
                key={column.id}
                draggable={!column.locked}
                onDragStart={() => setDragged(column.id)}
                onDragOver={(event) => event.preventDefault()}
                onDrop={() => drop(column.id)}
              >
                <span className="drag-handle" aria-hidden="true">
                  ⋮⋮
                </span>
                <label>
                  <input
                    type="checkbox"
                    checked={column.visible}
                    disabled={column.locked}
                    onChange={(event) =>
                      onChange(
                        columns.map((item) =>
                          item.id === column.id
                            ? { ...item, visible: event.target.checked }
                            : item,
                        ),
                      )
                    }
                  />
                  {column.label}
                </label>
                {!column.locked && (
                  <span className="reorder-buttons">
                    <button
                      type="button"
                      aria-label={`${column.label}上移`}
                      disabled={index <= 1}
                      onClick={() => move(column.id, -1)}
                    >
                      ↑
                    </button>
                    <button
                      type="button"
                      aria-label={`${column.label}下移`}
                      disabled={index === columns.length - 1}
                      onClick={() => move(column.id, 1)}
                    >
                      ↓
                    </button>
                  </span>
                )}
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
