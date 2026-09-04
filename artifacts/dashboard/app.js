(function () {
  const data = window.DASHBOARD_DATA;
  const state = {
    view: "top_stakes",
    search: "",
    system: "",
    sort: "confidence_stakes",
  };

  const money = new Intl.NumberFormat("en-ZA", {
    style: "currency",
    currency: "ZAR",
    maximumFractionDigits: 0,
  });
  const whole = new Intl.NumberFormat("en-ZA", { maximumFractionDigits: 0 });
  const pct = new Intl.NumberFormat("en-ZA", { style: "percent", maximumFractionDigits: 0 });

  const byId = (id) => document.getElementById(id);
  const number = (value) => Number(value || 0);
  const fmtMoney = (value) => money.format(number(value));
  const fmtWhole = (value) => whole.format(number(value));
  const fmtPct = (value) => value === null || value === undefined || value === "" ? "-" : pct.format(number(value));

  function setText(id, value) {
    byId(id).textContent = value;
  }

  function renderSummary() {
    const summary = data.summary;
    setText("asOfDate", summary.as_of_date ? `As of ${summary.as_of_date}` : "Latest forecast");
    setText("totalStakes", fmtMoney(summary.total_predicted_stakes_week_4));
    setText("totalClients", fmtWhole(summary.total_predicted_clients_week_4));
    setText("gamesScored", fmtWhole(summary.games_scored));
    setText("systemsScored", `${fmtWhole(summary.systems_scored)} systems scored`);
    setText("within10", fmtPct(1 - number(summary.holdout_off_by_10_rate)));
    setText("within20", fmtPct(1 - number(summary.holdout_off_by_20_rate)));
    setText("clientProbability", fmtPct(summary.avg_probability_clients));
    setText("stakesProbability", fmtPct(summary.avg_probability_stakes));
  }

  function renderSystemOptions() {
    const select = byId("systemFilter");
    data.systems
      .map((row) => row.SystemName)
      .sort((a, b) => a.localeCompare(b))
      .forEach((system) => {
        const option = document.createElement("option");
        option.value = system;
        option.textContent = system;
        select.appendChild(option);
      });
  }

  function renderFocusBars() {
    const top = [...(data.top_stakes || data.forecasts || [])]
      .sort((a, b) => number(b.predicted_stakes_week_4) - number(a.predicted_stakes_week_4))
      .slice(0, 20);
    const max = Math.max(...top.map((row) => number(row.predicted_stakes_week_4)), 1);
    byId("focusCount").textContent = `Top ${top.length} of ${fmtWhole(data.summary.games_scored)} games by stakes`;
    byId("focusBars").innerHTML = top
      .map((row) => {
        const width = Math.max(3, (number(row.predicted_stakes_week_4) / max) * 100);
        return `
          <div class="focus-row">
            <div>
              <div class="focus-name">${escapeHtml(row.TournamentName || row.Game_ID || "Unknown game")}</div>
              <div class="subtle">${escapeHtml(row.SystemName || "Unknown system")}</div>
            </div>
            <div class="bar-track" aria-hidden="true"><div class="bar-fill" style="width:${width}%"></div></div>
            <div class="num">${fmtMoney(row.predicted_stakes_week_4)}</div>
          </div>`;
      })
      .join("");
  }

  function rowsForView() {
    return data[state.view] || data.forecasts || [];
  }

  function defaultSortForView(view) {
    if (view === "top_stakes") return "confidence_stakes";
    if (view === "top_clients") return "confidence_clients";
    return "predicted_stakes_week_4";
  }

  function tieBreakSort(sortKey) {
    if (sortKey === "confidence_stakes") return "predicted_stakes_week_4";
    if (sortKey === "confidence_clients") return "predicted_clients_week_4";
    return sortKey === "predicted_stakes_week_4" ? "predicted_clients_week_4" : "predicted_stakes_week_4";
  }

  function filteredRows() {
    const query = state.search.trim().toLowerCase();
    const secondarySort = tieBreakSort(state.sort);
    return rowsForView()
      .filter((row) => !state.system || row.SystemName === state.system)
      .filter((row) => {
        if (!query) return true;
        return `${row.TournamentName || ""} ${row.SystemName || ""}`.toLowerCase().includes(query);
      })
      .sort((a, b) => {
        const primary = number(b[state.sort]) - number(a[state.sort]);
        return primary || number(b[secondarySort]) - number(a[secondarySort]);
      });
  }

  function renderRows() {
    const rows = filteredRows();
    byId("rowCount").textContent = `${fmtWhole(rows.length)} games shown`;
    if (!rows.length) {
      byId("gameRows").innerHTML = `
          <tr>
            <td class="empty-row" colspan="6">No games match the current filters.</td>
          </tr>`;
      return;
    }
    byId("gameRows").innerHTML = rows
      .slice(0, 300)
      .map((row) => {
        return `
          <tr>
            <td>
              <div class="game-title">${escapeHtml(row.TournamentName || row.Game_ID || "Unknown game")}</div>
              <div class="subtle">${escapeHtml(row.Game_ID || "")}</div>
            </td>
            <td>${escapeHtml(row.SystemName || "Unknown")}</td>
            <td class="num">${fmtMoney(row.predicted_stakes_week_4)}</td>
            <td class="num">${fmtPct(row.confidence_stakes)}</td>
            <td class="num">${fmtWhole(row.predicted_clients_week_4)}</td>
            <td class="num">${fmtPct(row.confidence_clients)}</td>
          </tr>`;
      })
      .join("");
  }

  function bindControls() {
    document.querySelectorAll(".tab").forEach((button) => {
      button.addEventListener("click", () => {
        document.querySelectorAll(".tab").forEach((tab) => {
          tab.classList.remove("active");
          tab.setAttribute("aria-selected", "false");
        });
        button.classList.add("active");
        button.setAttribute("aria-selected", "true");
        state.view = button.dataset.view;
        state.sort = defaultSortForView(state.view);
        byId("sortSelect").value = state.sort;
        renderRows();
      });
    });
    byId("searchInput").addEventListener("input", (event) => {
      state.search = event.target.value;
      renderRows();
    });
    byId("systemFilter").addEventListener("change", (event) => {
      state.system = event.target.value;
      renderRows();
    });
    byId("sortSelect").addEventListener("change", (event) => {
      state.sort = event.target.value;
      renderRows();
    });
  }

  function escapeHtml(value) {
    return String(value).replace(/[&<>"']/g, (char) => ({
      "&": "&amp;",
      "<": "&lt;",
      ">": "&gt;",
      '"': "&quot;",
      "'": "&#039;",
    })[char]);
  }

  if (!data) {
    document.body.innerHTML = "<main class=\"panel\"><h1>Dashboard data is missing</h1><p>Run the dashboard exporter first.</p></main>";
    return;
  }

  renderSummary();
  renderSystemOptions();
  renderFocusBars();
  byId("sortSelect").value = state.sort;
  bindControls();
  renderRows();
})();
