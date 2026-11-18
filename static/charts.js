// ---------------------------------------------------------------- charts (inline SVG, no libraries)
(function () {
  const NS = "http://www.w3.org/2000/svg";
  const C = { ink: "#0b1c30", soft: "#45464d", teal: "#006a61", tealSoft: "#86f2e4", grid: "#dce9ff", warn: "#ba1a1a" };

  function el(tag, attrs, text) {
    const e = document.createElementNS(NS, tag);
    for (const k in attrs) e.setAttribute(k, attrs[k]);
    if (text != null) e.textContent = text;
    return e;
  }
  function niceStep(range, ticks) {
    const raw = range / ticks, mag = Math.pow(10, Math.floor(Math.log10(raw))), n = raw / mag;
    return (n < 1.5 ? 1 : n < 3.5 ? 2 : n < 7.5 ? 5 : 10) * mag;
  }
  function card(title, note) {
    const box = document.createElement("div");
    box.className = "chart-card";
    const h = document.createElement("div");
    h.className = "chart-title";
    h.textContent = title;
    const p = document.createElement("p");
    p.className = "chart-note";
    p.textContent = note;
    box.append(h, p);
    return box;
  }
  // generic line chart; returns the svg
  function lineChart(o) {
    const W = 640, H = 240, L = 54, R = 14, T = 12, B = 40;
    const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": o.label });
    const x0 = o.xmin, x1 = o.xmax;
    let y0 = o.ymin, y1 = o.ymax;
    const X = (v) => L + ((v - x0) / (x1 - x0)) * (W - L - R);
    const Y = (v) => T + (1 - (v - y0) / (y1 - y0)) * (H - T - B);
    if (o.band) svg.append(el("rect", { x: X(o.band[0]), y: T, width: X(o.band[1]) - X(o.band[0]), height: H - T - B, fill: C.tealSoft, opacity: 0.25 }));
    const ys = niceStep(y1 - y0, 4);
    for (let v = Math.ceil(y0 / ys) * ys; v <= y1 + 1e-9; v += ys) {
      svg.append(el("line", { x1: L, x2: W - R, y1: Y(v), y2: Y(v), stroke: C.grid, "stroke-width": 1 }));
      svg.append(el("text", { x: L - 6, y: Y(v) + 4, "text-anchor": "end", "font-size": 11, fill: C.soft }, +v.toFixed(3)));
    }
    const xs = niceStep(x1 - x0, 7);
    for (let v = Math.ceil(x0 / xs) * xs; v <= x1 + 1e-9; v += xs) {
      svg.append(el("line", { x1: X(v), x2: X(v), y1: T, y2: H - B, stroke: C.grid, "stroke-width": 1 }));
      svg.append(el("text", { x: X(v), y: H - B + 15, "text-anchor": "middle", "font-size": 11, fill: C.soft }, +v.toFixed(2)));
    }
    svg.append(el("text", { x: (L + W - R) / 2, y: H - 6, "text-anchor": "middle", "font-size": 12, fill: C.soft }, o.xlabel));
    svg.append(el("text", { x: 13, y: (T + H - B) / 2, "text-anchor": "middle", "font-size": 12, fill: C.soft, transform: `rotate(-90 13 ${(T + H - B) / 2})` }, o.ylabel));
    let dpath = "";
    o.x.forEach((xv, i) => { if (o.y[i] != null && xv != null) dpath += (dpath ? "L" : "M") + X(xv).toFixed(1) + " " + Y(o.y[i]).toFixed(1); });
    svg.append(el("path", { d: dpath, fill: "none", stroke: C.teal, "stroke-width": 1.8, "stroke-linejoin": "round" }));
    if (o.mark) {
      const mx = X(o.mark.x);
      svg.append(el("line", { x1: mx, x2: mx, y1: T, y2: H - B, stroke: o.mark.color, "stroke-width": 1.6, "stroke-dasharray": o.mark.dash ? "5 4" : "0" }));
      const right = mx > W * 0.7;
      svg.append(el("text", { x: right ? mx - 6 : mx + 6, y: T + 13, "text-anchor": right ? "end" : "start", "font-size": 12, "font-weight": 700, fill: o.mark.color }, o.mark.text));
    }
    return svg;
  }

  window.drawCharts = function (d) {
    const host = document.getElementById("resCharts");
    host.replaceChildren();
    const wt = d.waveform && d.waveform.t, wy = d.waveform && d.waveform.y;
    const sf = d.spectrum && d.spectrum.f, sp = d.spectrum && d.spectrum.p;
    if (!wt || !wt.length || !sf || !sf.length) {
      const m = document.createElement("div");
      m.className = "chart-card";
      m.textContent = "No chart data for this recording.";
      host.append(m);
      return;
    }
    // waveform: image-height units -> estimated cm (same scale the RMS uses)
    const k = d.rms && d.rms_cm ? d.rms_cm / d.rms : null;
    const wv = wy.map((v) => (v == null ? null : (k ? v * k : v)));
    const wmax = Math.max(0.05, ...wv.filter((v) => v != null).map(Math.abs)) * 1.15;
    const c1 = card("Hand movement (2-12 Hz band)", "Tremor-band motion of the wrist along its main direction. A steady hand stays flat; a tremor looks like a regular wave.");
    c1.append(lineChart({
      label: "Waveform", x: wt, y: wv, xmin: 0, xmax: wt[wt.length - 1], ymin: -wmax, ymax: wmax,
      xlabel: "Time (s)", ylabel: k ? "Displacement (cm, est.)" : "Displacement (image units)"
    }));
    // spectrum: normalised power so the peak is easy to compare
    const pmax = Math.max(...sp.filter((v) => v != null)) || 1;
    const pn = sp.map((v) => (v == null ? null : v / pmax));
    const c2 = card("Frequency spectrum", "Where the movement energy sits. The shaded band (2-12 Hz) is where we look for tremor; a tall narrow peak means a regular rhythm.");
    const mark = d.reliable
      ? { x: d.frequency_hz, color: C.teal, text: d.frequency_hz.toFixed(2) + " Hz", dash: false }
      : (d.strongest_bin_hz ? { x: d.strongest_bin_hz, color: C.soft, text: "strongest bin " + d.strongest_bin_hz.toFixed(2) + " Hz (not reliable)", dash: true } : null);
    c2.append(lineChart({
      label: "Spectrum", x: sf, y: pn, xmin: 0, xmax: Math.min(15, sf[sf.length - 1]), ymin: 0, ymax: 1.05,
      band: [2, 12], mark, xlabel: "Frequency (Hz)", ylabel: "Relative power"
    }));
    host.append(c1, c2);
  };
})();