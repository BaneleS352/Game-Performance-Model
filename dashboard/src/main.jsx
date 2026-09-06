import React, { useEffect, useMemo, useState } from "react";
import { createRoot } from "react-dom/client";
import "./styles.css";

const fallbackData = window.DASHBOARD_DATA;

const fallbackWorkbookSheets = [
  "Holdout Comparison",
  "High Performers",
  "Forecasts",
  "Model Accuracy",
  "Classification Metrics",
  "Calibration",
  "future_performance_score_calcul",
];
const hiddenWorkbookSheets = new Set(["Model Accuracy", "Classification Metrics", "Calibration"]);
const workbookSheetLabels = {
  future_performance_score_calcul: "Future Performance Score Calculator",
};

const money = new Intl.NumberFormat("en-ZA", {
  style: "currency",
  currency: "ZAR",
  maximumFractionDigits: 0,
});
const whole = new Intl.NumberFormat("en-ZA", { maximumFractionDigits: 0 });
const pct = new Intl.NumberFormat("en-ZA", { style: "percent", maximumFractionDigits: 0 });
const ledgerRowHeight = 44;
const ledgerOverscan = 10;
const ledgerWindowSize = 34;
const baseLedgerColumns = [
  { key: "Game_ID", label: "Game ID", type: "text" },
  { key: "Lifecycle_Stage", label: "Lifecycle Stage", type: "text" },
  { key: "Horizon", label: "Horizon", type: "number" },
];
const clientLedgerColumns = [
  ...baseLedgerColumns,
  {
    key: "predicted_clients_week_4_range",
    label: "Prediction Clients Week 4 Range",
    type: "number",
    sortKey: "predicted_clients_week_4",
    render: (row) => `${formatValue(row.predicted_clients_week_4_low, "number")} - ${formatValue(row.predicted_clients_week_4_high, "number")}`,
  },
  { key: "confidence_clients", label: "Confidence Clients", type: "percent" },
];
const stakesLedgerColumns = [
  ...baseLedgerColumns,
  {
    key: "predicted_stakes_week_4_range",
    label: "Prediction Stakes Week 4 Range",
    type: "money",
    sortKey: "predicted_stakes_week_4",
    render: (row) => `${formatValue(row.predicted_stakes_week_4_low, "money")} - ${formatValue(row.predicted_stakes_week_4_high, "money")}`,
  },
  { key: "confidence_stakes", label: "Confidence Stakes", type: "percent" },
];
const ledgerModes = {
  top_stakes: {
    label: "Top Stakes",
    sortKey: "predicted_stakes_week_4",
    columns: stakesLedgerColumns,
  },
  top_clients: {
    label: "Top Clients",
    sortKey: "predicted_clients_week_4",
    columns: clientLedgerColumns,
  },
};

function number(value) {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : 0;
}

function formatValue(value, type) {
  if (value === null || value === undefined || value === "") return "-";
  if (type === "money") return money.format(number(value));
  if (type === "percent") return pct.format(number(value));
  if (type === "number") return whole.format(number(value));
  if (type === "boolean") return value ? "Yes" : "No";
  return String(value);
}

function tieBreakSort(sortKey) {
  if (sortKey === "confidence_clients") return "predicted_clients_week_4";
  return "predicted_stakes_week_4";
}

function compareValues(a, b, column) {
  const av = a[column];
  const bv = b[column];
  const an = Number(av);
  const bn = Number(bv);
  if (Number.isFinite(an) && Number.isFinite(bn)) return bn - an;
  return String(bv ?? "").localeCompare(String(av ?? ""));
}

function sheetLabel(name) {
  return workbookSheetLabels[name] ?? name;
}

function Dashboard() {
  const [data, setData] = useState(fallbackData);

  useEffect(() => {
    let active = true;
    fetch("/api/dashboard-data")
      .then(readJson)
      .then((nextData) => {
        if (active) setData(nextData);
      })
      .catch(() => {});
    return () => { active = false; };
  }, []);

  return <DashboardView data={data} onDashboardData={setData} />;
}

function DashboardView({ data, onDashboardData }) {
  if (!data) {
    return (
      <main className="arcade-shell">
        <header className="poster-hero" aria-label="Game Performance Planner">
          <div className="hero-copy">
            <p className="top-pill"><GamepadIcon /> Forecast performance, compare systems, prioritize value.</p>
            <h1><span>Game</span> Performance Planner</h1>
          </div>
        </header>
        <section className="missing-data">
          <h2>Dashboard data is missing</h2>
          <p>Upload and score a workbook to create the first dashboard view.</p>
        </section>
        <UploadPanel onDashboardData={onDashboardData} />
      </main>
    );
  }

  const columns = data.columns?.forecasts ?? Object.keys(data.forecasts?.[0] ?? {}).map((key) => ({
    key,
    label: key.replaceAll("_", " "),
    type: "text",
    defaultVisible: true,
  }));
  const [search, setSearch] = useState("");
  const [system, setSystem] = useState("");
  const [ledgerMode, setLedgerMode] = useState("top_stakes");
  const [sort, setSort] = useState("predicted_stakes_week_4");
  const [activeSheetName, setActiveSheetName] = useState("");
  const [ledgerScrollTop, setLedgerScrollTop] = useState(0);

  useEffect(() => {
    setLedgerScrollTop(0);
  }, [ledgerMode, search, sort, system]);

  const summary = data.summary;
  const systems = useMemo(
    () => [...new Set((data.systems ?? []).map((row) => row.SystemName).filter(Boolean))].sort(),
    [data.systems],
  );

  const rows = useMemo(() => {
    const query = search.trim().toLowerCase();
    const secondarySort = tieBreakSort(sort);
    return [...(data.forecasts ?? [])]
      .filter((row) => !system || row.SystemName === system)
      .filter((row) => {
        if (!query) return true;
        return columns.some((column) => String(row[column.key] ?? "").toLowerCase().includes(query));
      })
      .sort((a, b) => compareValues(a, b, sort) || compareValues(a, b, secondarySort));
  }, [columns, search, sort, system]);

  const topSystems = useMemo(
    () => [...(data.systems ?? [])]
      .sort((a, b) => number(b.predicted_stakes_week_4) - number(a.predicted_stakes_week_4))
      .slice(0, 6),
    [data.systems],
  );
  const topClientSystems = useMemo(
    () => [...(data.systems ?? [])]
      .sort((a, b) => number(b.predicted_clients_week_4) - number(a.predicted_clients_week_4))
      .slice(0, 6),
    [data.systems],
  );
  const topGames = useMemo(
    () => [...(data.top_stakes ?? data.forecasts ?? [])]
      .sort((a, b) => number(b.predicted_stakes_week_4) - number(a.predicted_stakes_week_4))
      .slice(0, 20),
    [data.top_stakes, data.forecasts],
  );
  const workbookSheets = useMemo(() => {
    if (data.workbook_sheets?.length) return data.workbook_sheets;
    return fallbackWorkbookSheets.map((name) => ({
      name,
      totalRows: name === "Forecasts" ? data.forecasts?.length ?? 0 : 0,
      columns: name === "Forecasts" ? data.columns?.forecasts ?? [] : [],
      rows: name === "Forecasts" ? data.forecasts ?? [] : [],
    }));
  }, [data]);
  const activeSheet = workbookSheets.find((sheet) => sheet.name === activeSheetName);
  const selectedColumns = ledgerModes[ledgerMode].columns;
  const ledgerStart = Math.min(
    Math.max(0, Math.floor(ledgerScrollTop / ledgerRowHeight) - ledgerOverscan),
    Math.max(0, rows.length - ledgerWindowSize),
  );
  const ledgerEnd = Math.min(rows.length, ledgerStart + ledgerWindowSize);
  const ledgerRows = rows.slice(ledgerStart, ledgerEnd);
  const ledgerTopPad = ledgerStart * ledgerRowHeight;
  const ledgerBottomPad = Math.max(0, (rows.length - ledgerEnd) * ledgerRowHeight);
  const totalSystemStakes = topSystems.reduce((sum, row) => sum + number(row.predicted_stakes_week_4), 0) || 1;
  const totalSystemClients = topClientSystems.reduce((sum, row) => sum + number(row.predicted_clients_week_4), 0) || 1;
  const healthScore = Math.round((1 - number(summary.holdout_off_by_20_rate)) * 100);
  const xp = Math.round(number(summary.avg_probability_clients) * 420);
  const levelProgress = Math.min(100, Math.max(0, (xp / 420) * 100));
  const donutBackground = makeDonut(topSystems, totalSystemStakes, "predicted_stakes_week_4");
  const clientDonutBackground = makeDonut(topClientSystems, totalSystemClients, "predicted_clients_week_4");

  function openSheet(name) {
    setActiveSheetName(name);
    window.scrollTo({ top: 0, behavior: "smooth" });
  }

  function changeLedgerMode(mode) {
    setLedgerMode(mode);
    setSort(ledgerModes[mode].sortKey);
  }

  return (
    <main className="arcade-shell">
      <header className="poster-hero" aria-label="Game Performance Planner">
        <div className="hero-copy">
          <p className="top-pill"><GamepadIcon /> Forecast performance, compare systems, prioritize value.</p>
          <h1><span>Game</span> Performance Planner</h1>
        </div>
      </header>
      <WorkbookSheetTabs activeSheetName={activeSheetName} onOpenSheet={openSheet} workbookSheets={workbookSheets} />

      {activeSheet ? (
        <SheetPage sheet={activeSheet} />
      ) : (
        <section className="layout-grid" aria-label="Game forecast dashboard">
          <section className="quest-card" aria-label="Forecast Quest dashboard">
          <div className="quest-topbar">
            <div className="quest-title">
              <span className="badge-icon"><GamepadIcon /></span>
              <div>
                <h2>Forecast Quest</h2>
                <p>Level 3 - Portfolio Hero</p>
              </div>
            </div>
          </div>
          <div className="xp-track" aria-label={`${xp} of 420 XP to level 4`}>
            <span>{xp}/420 XP to level 4</span>
            <div><i style={{ width: `${levelProgress}%` }} /></div>
          </div>

          <section className="stat-grid" aria-label="Forecast summary">
            <Kpi icon={<CashIcon />} label="Week 4 Stakes" value={money.format(number(summary.total_predicted_stakes_week_4))} tone="green" />
            <Kpi icon={<UsersIcon />} label="Week 4 Clients" value={whole.format(number(summary.total_predicted_clients_week_4))} tone="pink" />
            <Kpi icon={<ScaleIcon />} label="Client Probability" value={pct.format(number(summary.avg_probability_clients))} tone="mint" />
            <Kpi icon={<PiggyIcon />} label="Stake Probability" value={pct.format(number(summary.avg_probability_stakes))} tone="blue" />
            <Kpi icon={<FlexIcon />} label="Health Score" value={whole.format(healthScore)} tone="lime" />
          </section>

          <section className="mid-grid">
            <article className="widget">
              <h3><GemIcon /> Stakes by System</h3>
              <div className="donut-area">
                <div className="legend">
                  {topSystems.map((row, index) => (
                    <span key={row.SystemName}><i style={{ background: donutColors[index] }} /> {row.SystemName}</span>
                  ))}
                </div>
                <div className="donut" style={{ background: donutBackground }} aria-label="Top systems by predicted stakes" />
              </div>
            </article>

            <article className="widget">
              <h3><UsersIcon /> Clients by System</h3>
              <div className="donut-area">
                <div className="legend">
                  {topClientSystems.map((row, index) => (
                    <span key={row.SystemName}><i style={{ background: donutColors[index] }} /> {row.SystemName}</span>
                  ))}
                </div>
                <div className="donut" style={{ background: clientDonutBackground }} aria-label="Top systems by predicted clients" />
              </div>
            </article>
          </section>

          <section className="top-games">
            <div className="top-games-heading">
              <h3><MedalIcon /> Top 20 Games</h3>
              <span>Ranked by predicted Week 4 stakes</span>
            </div>
            <div className="table-wrap">
              <table>
                <thead>
                  <tr>
                    <th>Rank</th>
                    <th>Game</th>
                    <th>System</th>
                    <th>Predicted Stakes</th>
                    <th>Predicted Clients</th>
                    <th>Confidence</th>
                  </tr>
                </thead>
                <tbody>
                  {topGames.map((row, index) => (
                    <tr key={`${row.Game_ID ?? "top-game"}-${index}`}>
                      <td className="rank">#{index + 1}</td>
                      <td>{row.TournamentName || row.Game_ID || "-"}</td>
                      <td>{row.SystemName || "-"}</td>
                      <td className="num">{money.format(number(row.predicted_stakes_week_4))}</td>
                      <td className="num">{whole.format(number(row.predicted_clients_week_4))}</td>
                      <td className="num">{pct.format(number(row.confidence_stakes))}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </section>

          <section className="ledger">
            <div className="ledger-heading">
              <h3><GamepadIcon /> Game Ledger</h3>
              <div className="ledger-mode" aria-label="Ledger ranking mode">
                {Object.entries(ledgerModes).map(([mode, config]) => (
                  <button
                    aria-pressed={ledgerMode === mode}
                    className={ledgerMode === mode ? "active" : ""}
                    key={mode}
                    onClick={() => changeLedgerMode(mode)}
                    type="button"
                  >
                    {config.label}
                  </button>
                ))}
              </div>
              <label>
                <span className="sr-only">Search all columns</span>
                <input value={search} onChange={(event) => setSearch(event.target.value)} type="search" placeholder="Search games or systems" />
              </label>
              <label>
                <span className="sr-only">Filter by system</span>
                <select value={system} onChange={(event) => setSystem(event.target.value)}>
                  <option value="">All systems</option>
                  {systems.map((name) => <option key={name} value={name}>{name}</option>)}
                </select>
              </label>
            </div>
            <div className="table-wrap" onScroll={(event) => setLedgerScrollTop(event.currentTarget.scrollTop)}>
              <table style={{ minWidth: `${Math.max(860, selectedColumns.length * 150)}px` }}>
                <thead>
                  <tr>
                    {selectedColumns.map((column) => (
                      <th key={column.key}>
                        <button type="button" onClick={() => setSort(column.sortKey ?? column.key)}>
                          {column.label}
                        </button>
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {rows.length ? (
                    <>
                      {ledgerTopPad > 0 && (
                        <tr className="ledger-spacer" aria-hidden="true">
                          <td colSpan={Math.max(selectedColumns.length, 1)} style={{ height: ledgerTopPad }} />
                        </tr>
                      )}
                      {ledgerRows.map((row, index) => (
                    <tr key={`${row.Game_ID ?? "row"}-${ledgerStart + index}`}>
                      {selectedColumns.map((column) => (
                        <td className={column.type === "money" || column.type === "number" || column.type === "percent" ? "num" : ""} key={column.key}>
                          {column.render ? column.render(row) : formatValue(row[column.key], column.type)}
                        </td>
                      ))}
                    </tr>
                      ))}
                      {ledgerBottomPad > 0 && (
                        <tr className="ledger-spacer" aria-hidden="true">
                          <td colSpan={Math.max(selectedColumns.length, 1)} style={{ height: ledgerBottomPad }} />
                        </tr>
                      )}
                    </>
                  ) : (
                    <tr>
                      <td className="empty" colSpan={Math.max(selectedColumns.length, 1)}>No games match the current filters.</td>
                    </tr>
                  )}
                </tbody>
              </table>
            </div>
          </section>
          </section>
        </section>
      )}
      <UploadPanel onDashboardData={onDashboardData} />
    </main>
  );
}

async function readJson(response) {
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(payload.error || "The request could not be completed.");
  return payload;
}

function UploadPanel({ onDashboardData }) {
  const [upload, setUpload] = useState(null);
  const [sheets, setSheets] = useState([]);
  const [sheetName, setSheetName] = useState("");
  const [status, setStatus] = useState("idle");
  const [message, setMessage] = useState("");
  const [reportUrl, setReportUrl] = useState("");

  async function chooseWorkbook(event) {
    const file = event.target.files?.[0];
    if (!file) return;
    setUpload(null);
    setSheets([]);
    setSheetName("");
    setMessage("");
    setReportUrl("");
    setStatus("uploading");
    try {
      const payload = await readJson(await fetch("/api/uploads", {
        method: "POST",
        headers: { "Content-Type": file.type || "application/octet-stream", "X-File-Name": encodeURIComponent(file.name) },
        body: file,
      }));
      setUpload({ id: payload.upload_id, name: file.name });
      setSheets(payload.sheets);
      setSheetName(payload.sheets[0] ?? "");
      setStatus("ready");
      setMessage("Workbook saved. Choose a worksheet to score.");
    } catch (error) {
      setStatus("error");
      setMessage(error.message);
    }
  }

  async function runModel() {
    if (!upload || !sheetName) return;
    setStatus("running");
    setMessage("Scoring the selected worksheet and refreshing the dashboard...");
    setReportUrl("");
    try {
      const result = await readJson(await fetch(`/api/uploads/${upload.id}/run`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ sheet_name: sheetName }),
      }));
      const dashboardData = await readJson(await fetch("/api/dashboard-data"));
      onDashboardData(dashboardData);
      setStatus("success");
      setMessage(`${whole.format(result.summary.games_scored)} games scored from ${upload.name}.`);
      setReportUrl(result.report_url);
    } catch (error) {
      setStatus("error");
      setMessage(error.message);
    }
  }

  const busy = status === "uploading" || status === "running";
  return (
    <section className="upload-panel" aria-labelledby="upload-heading" aria-busy={busy}>
      <div className="upload-heading">
        <span className="badge-icon"><SheetIcon /></span>
        <div>
          <h2 id="upload-heading">New workbook</h2>
          <p>Use the trained model to score an uploaded worksheet.</p>
        </div>
      </div>
      <div className="upload-controls">
        <label className="file-input">
          <span>Workbook (.xlsx)</span>
          <input type="file" accept=".xlsx,application/vnd.openxmlformats-officedocument.spreadsheetml.sheet" onChange={chooseWorkbook} disabled={busy} />
        </label>
        <label>
          <span>Worksheet</span>
          <select value={sheetName} onChange={(event) => setSheetName(event.target.value)} disabled={!upload || busy}>
            <option value="">Choose a worksheet</option>
            {sheets.map((sheet) => <option key={sheet} value={sheet}>{sheet}</option>)}
          </select>
        </label>
        <button className="run-button" type="button" onClick={runModel} disabled={!upload || !sheetName || busy}>
          <ChartIcon /> {status === "running" ? "Scoring..." : "Run model"}
        </button>
      </div>
      {message && (
        <p className={`upload-status ${status}`} role={status === "error" ? "alert" : "status"} aria-live="polite">
          {message} {reportUrl && <a href={reportUrl}>Download report</a>}
        </p>
      )}
    </section>
  );
}

function WorkbookSheetTabs({ workbookSheets, activeSheetName, onOpenSheet }) {
  const visibleSheets = workbookSheets.filter((sheet) => !hiddenWorkbookSheets.has(sheet.name));
  return (
    <nav className="quest-tabs" aria-label="Workbook sheets">
      <button className={!activeSheetName ? "active" : ""} type="button" onClick={() => onOpenSheet("")}>
        <GridIcon /> Dashboard
      </button>
      {visibleSheets.map((sheet) => (
        <button
          className={sheet.name === activeSheetName ? "active" : ""}
          key={sheet.name}
          type="button"
          onClick={() => onOpenSheet(sheet.name)}
        >
          <SheetIcon /> {sheetLabel(sheet.name)}
        </button>
      ))}
    </nav>
  );
}

function SheetPage({ sheet }) {
  const columns = sheet.columns ?? [];
  const rows = (sheet.rows ?? []).slice(0, 250);
  const defaultColumns = columns.filter((column) => column.defaultVisible).slice(0, 10);
  const visibleColumns = defaultColumns.length ? defaultColumns : columns.slice(0, 10);
  return (
    <section className="sheet-page quest-card" aria-label={`${sheetLabel(sheet.name)} sheet`}>
      <div className="sheet-page-heading">
        <span className="badge-icon"><SheetIcon /></span>
        <div>
          <h2>{sheetLabel(sheet.name)}</h2>
          <p>{whole.format(sheet.totalRows ?? sheet.rows?.length ?? 0)} workbook rows</p>
        </div>
      </div>
      {visibleColumns.length ? (
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                {visibleColumns.map((column) => <th key={column.key}>{column.label}</th>)}
              </tr>
            </thead>
            <tbody>
              {rows.map((row, index) => (
                <tr key={`${sheet.name}-${index}`}>
                  {visibleColumns.map((column) => (
                    <td className={column.type === "money" || column.type === "number" || column.type === "percent" ? "num" : ""} key={column.key}>
                      {formatValue(row[column.key], column.type)}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : (
        <p className="sheet-empty">Run the dashboard exporter to load a preview for this workbook sheet.</p>
      )}
    </section>
  );
}

function Kpi({ icon, label, value, tone }) {
  return (
    <article className={`kpi ${tone}`}>
      <span className="kpi-icon" aria-hidden="true">{icon}</span>
      <strong>{value}</strong>
      <span>{label}</span>
    </article>
  );
}

const donutColors = ["#7a1fff", "#b929ff", "#00d4ff", "#00df8c", "#ff8a00", "#ff2e92"];

function makeDonut(rows, total, field) {
  let cursor = 0;
  const stops = rows.map((row, index) => {
    const start = cursor;
    cursor += (number(row[field]) / total) * 100;
    return `${donutColors[index]} ${start}% ${cursor}%`;
  });
  return `conic-gradient(${stops.join(", ")})`;
}

function IconBase({ children }) {
  return <svg viewBox="0 0 48 48" focusable="false" aria-hidden="true">{children}</svg>;
}

function GamepadIcon() {
  return <IconBase><path d="M15 19h18c5 0 8 4 8 9v2c0 4-3 7-6 7-3 0-5-4-7-4h-8c-2 0-4 4-7 4s-6-3-6-7v-2c0-5 3-9 8-9Z" /><path d="M16 28h8M20 24v8M31 27h.2M36 31h.2" /></IconBase>;
}

function GemIcon() {
  return <IconBase><path d="M14 8h20l8 12-18 20L6 20 14 8Z" /><path d="M6 20h36M14 8l10 32L34 8M14 8l10 12L34 8" /></IconBase>;
}

function CoinIcon() {
  return <IconBase><circle cx="24" cy="24" r="16" /><path d="M28 17h-6a4 4 0 0 0 0 8h4a4 4 0 0 1 0 8h-7M24 13v22" /></IconBase>;
}

function FlameIcon() {
  return <IconBase><path d="M25 42c8-2 12-8 12-15 0-7-5-11-9-18-1 6-6 8-6 14-3-2-4-5-4-8-5 5-7 9-7 14 0 8 6 13 14 13Z" /></IconBase>;
}

function CashIcon() {
  return <IconBase><path d="M8 15h32v22H8V15Z" /><circle cx="24" cy="26" r="6" /><path d="M13 20h5M30 32h5" /></IconBase>;
}

function UsersIcon() {
  return <IconBase><circle cx="18" cy="18" r="6" /><circle cx="32" cy="20" r="5" /><path d="M7 39c2-8 7-12 13-12s11 4 13 12M27 30c5 0 9 3 11 9" /></IconBase>;
}

function ScaleIcon() {
  return <IconBase><path d="M24 9v31M13 16h22M14 16 8 29h12l-6-13ZM34 16l-6 13h12l-6-13ZM17 40h14" /></IconBase>;
}

function PiggyIcon() {
  return <IconBase><path d="M10 27c0-7 6-12 16-12 7 0 12 3 14 8v10h-5l-2 5h-6v-4h-8v4h-6l-2-6H7v-5h3Z" /><path d="M18 15l-3-5h8M34 24h.2" /></IconBase>;
}

function FlexIcon() {
  return <IconBase><path d="M12 34c8-2 10-14 16-14 5 0 6 5 10 5M9 29c2-10 8-17 16-17 7 0 12 5 14 14M20 35h18" /></IconBase>;
}

function ChartIcon() {
  return <IconBase><path d="M9 39h30M13 35V24M24 35V15M35 35V9M28 12h7v7" /></IconBase>;
}

function MedalIcon() {
  return <IconBase><circle cx="24" cy="27" r="9" /><path d="m17 5 7 13 7-13M13 5h22M24 22l2 4 4 .5-3 3 .8 4.5-3.8-2.2-3.8 2.2.8-4.5-3-3 4-.5 2-4Z" /></IconBase>;
}

function GridIcon() {
  return <IconBase><path d="M9 9h12v12H9zM27 9h12v12H27zM9 27h12v12H9zM27 27h12v12H27z" /></IconBase>;
}

function SheetIcon() {
  return <IconBase><path d="M12 6h18l8 8v28H12V6Z" /><path d="M30 6v10h8M18 24h14M18 31h14M18 38h9" /></IconBase>;
}

createRoot(document.getElementById("root")).render(<Dashboard />);
