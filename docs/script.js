(() => {
  "use strict";
  const prefersReduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  /* ---------- Training reel ---------- */
  const reel = document.querySelector(".reel");
  if (reel) {
    const slides = [...reel.querySelectorAll(".reel-slide")];
    const tabs = [...reel.querySelectorAll(".reel-tabs button")];
    const bars = tabs.map((t) => t.querySelector(".reel-bar i"));
    const phaseEl = reel.querySelector(".reel-phase");
    const lineEl = reel.querySelector(".reel-line");
    const toggle = reel.querySelector(".reel-toggle");
    const CHAPTER_MS = 9000;

    let current = 0;
    let elapsed = 0;
    let lastTick = null;
    let rafId = null;
    let manualPaused = prefersReduced; // reduced motion: wait for an explicit play
    let offscreen = false;

    const videoOf = (i) => slides[i].querySelector("video");

    function ensureSrc(i) {
      const v = videoOf(i);
      const s = v.querySelector("source[data-src]");
      if (s) {
        s.src = s.dataset.src;
        s.removeAttribute("data-src");
        v.load();
      }
    }

    const isPaused = () => manualPaused || offscreen || document.hidden;

    function stopLoop() {
      if (rafId) cancelAnimationFrame(rafId);
      rafId = null;
    }

    function tick(now) {
      if (lastTick == null) lastTick = now;
      elapsed += now - lastTick;
      lastTick = now;
      if (elapsed >= CHAPTER_MS) {
        setChapter(current + 1);
        return;
      }
      bars[current].style.transform = `scaleX(${(elapsed / CHAPTER_MS).toFixed(4)})`;
      rafId = requestAnimationFrame(tick);
    }

    function startLoop() {
      stopLoop();
      lastTick = null;
      rafId = requestAnimationFrame(tick);
    }

    function syncPlayback() {
      slides.forEach((_, i) => {
        const v = videoOf(i);
        if (i === current && !isPaused()) {
          ensureSrc(i);
          v.play().catch(() => {});
        } else {
          v.pause();
        }
      });
      reel.classList.toggle("is-paused", manualPaused);
      toggle.setAttribute("aria-label", manualPaused ? "Play reel" : "Pause reel");
      if (isPaused()) stopLoop();
      else startLoop();
    }

    function setChapter(i) {
      current = (i + slides.length) % slides.length;
      elapsed = 0;
      slides.forEach((s, k) => s.classList.toggle("is-active", k === current));
      tabs.forEach((t, k) => t.setAttribute("aria-selected", String(k === current)));
      bars.forEach((b, k) => {
        b.style.transform = k < current ? "scaleX(1)" : "scaleX(0)";
      });
      phaseEl.textContent = tabs[current].dataset.phase;
      lineEl.textContent = tabs[current].dataset.line;
      if (!isPaused()) ensureSrc((current + 1) % slides.length); // warm the next chapter
      syncPlayback();
    }

    tabs.forEach((t, k) => t.addEventListener("click", () => setChapter(k)));
    toggle.addEventListener("click", () => {
      manualPaused = !manualPaused;
      syncPlayback();
    });
    document.addEventListener("visibilitychange", syncPlayback);
    if ("IntersectionObserver" in window) {
      new IntersectionObserver(
        ([e]) => {
          offscreen = !e.isIntersecting;
          syncPlayback();
        },
        { threshold: 0.15 }
      ).observe(reel);
    }

    setChapter(0);
  }

  /* ---------- Curriculum stage explorer ---------- */
  const explorer = document.querySelector(".stage-explorer");
  if (explorer) {
    const chips = [...explorer.querySelectorAll(".stage-nav button")];
    const panels = [...explorer.querySelectorAll(".stage-panel")];
    const rail = explorer.querySelector(".stage-rail");
    const TOTAL_STAGES = 23;
    if (rail) {
      for (let i = 0; i < TOTAL_STAGES; i++) rail.appendChild(document.createElement("i"));
    }
    const ticks = rail ? [...rail.children] : [];

    const parseStages = (str) => {
      const on = new Set();
      str.split(",").forEach((part) => {
        const [a, b] = part.split("-").map(Number);
        for (let s = a; s <= (b || a); s++) on.add(s);
      });
      return on;
    };

    const select = (k) => {
      chips.forEach((c, i) => c.setAttribute("aria-selected", String(i === k)));
      panels.forEach((p, i) => p.classList.toggle("is-active", i === k));
      const on = parseStages(chips[k].dataset.stages);
      ticks.forEach((t, i) => t.classList.toggle("on", on.has(i + 1)));
    };

    chips.forEach((c, k) => {
      c.addEventListener("click", () => select(k));
      c.addEventListener("mouseenter", () => select(k));
      c.addEventListener("focus", () => select(k));
    });
    select(0);
  }

  /* ---------- Architecture expand-on-hover ---------- */
  const arch = document.querySelector(".arch");
  if (arch) {
    const summary = arch.querySelector(".arch-summary");
    let pinned = false;
    const set = (open) => {
      arch.classList.toggle("is-open", open);
      summary.setAttribute("aria-expanded", String(open));
    };
    set(false); // ships expanded for no-JS; condensed once JS is running
    summary.addEventListener("click", () => {
      pinned = !pinned;
      set(pinned);
    });
    arch.addEventListener("mouseenter", () => set(true));
    arch.addEventListener("mouseleave", () => {
      if (!pinned) set(false);
    });
    arch.addEventListener("focusout", (e) => {
      if (!pinned && !arch.contains(e.relatedTarget)) set(false);
    });
    summary.addEventListener("focus", () => set(true));
  }

  /* ---------- Seed lab: in-browser port of the generator's layout rules ---------- */
  const lab = document.querySelector(".seedlab");
  if (lab) {
    const input = lab.querySelector("#seed-input");
    const dice = lab.querySelector(".seedlab-dice");
    const svg = lab.querySelector("svg");
    const readout = lab.querySelector(".seedlab-readout");

    // configs/procedural_course.v1.json + ProceduralCourseGenerator.lua (v0.7)
    const CFG = {
      stageCount: 8, platW: 12, platL: 12,
      gapMin: 3.5, gapMax: 7, offMin: -5, offMax: 5,
      beamWMin: 1.5, beamWMax: 3, beamLMin: 8, beamLMax: 16,
      stairMin: 3, stairMax: 6, stairRun: 3, stairRise: 1,
      maxJumpGap: 8, maxLateralOffset: 6, maxCourseRise: 16, maxAttempts: 32,
    };
    const KINDS = ["gap", "offset", "beam", "stairs"];
    const COLORS = { platform: "#4891d2", beam: "#a86d0a", stair: "#5b3d99" };

    const mulberry32 = (a) => () => {
      a |= 0; a = (a + 0x6d2b79f5) | 0;
      let t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };

    function candidate(genSeed) {
      const rnd = mulberry32(genSeed);
      const num = (a, b) => a + rnd() * (b - a);
      const int = (a, b) => a + Math.floor(rnd() * (b - a + 1));
      let cur = { x: 0, y: 4, z: 0 };
      const parts = [{ kind: "platform", x: 0, y: 4, z: 0, w: CFG.platW, l: CFG.platL, seg: 0 }];
      const segs = [];
      for (let i = 1; i <= CFG.stageCount; i++) {
        const kind = KINDS[int(0, KINDS.length - 1)];
        if (kind === "gap" || kind === "offset") {
          const gap = num(CFG.gapMin, CFG.gapMax);
          let off = 0, angle = 0;
          if (kind === "offset") {
            off = num(CFG.offMin, CFG.offMax);
            angle = (Math.atan(off / (CFG.platL + gap)) * 180) / Math.PI;
          }
          cur = { x: cur.x + off, y: cur.y, z: cur.z - (CFG.platL + gap) };
          parts.push({ kind: "platform", ...cur, w: CFG.platW, l: CFG.platL, seg: i });
          segs.push({ i, kind, gap, off, angle, exit: { ...cur } });
        } else if (kind === "beam") {
          const len = num(CFG.beamLMin, CFG.beamLMax);
          const width = num(CFG.beamWMin, CFG.beamWMax);
          const edge = cur.z - CFG.platL / 2;
          parts.push({ kind: "beam", x: cur.x, y: cur.y, z: edge - len / 2, w: width, l: len, seg: i });
          cur = { x: cur.x, y: cur.y, z: edge - len - CFG.platL / 2 };
          parts.push({ kind: "platform", ...cur, w: CFG.platW, l: CFG.platL, seg: i });
          segs.push({ i, kind, len, width, exit: { ...cur } });
        } else {
          const count = int(CFG.stairMin, CFG.stairMax);
          const edge = cur.z - CFG.platL / 2;
          for (let s = 1; s <= count; s++) {
            parts.push({
              kind: "stair", x: cur.x, y: cur.y + s * CFG.stairRise,
              z: edge - (s - 0.5) * CFG.stairRun, w: CFG.platW, l: CFG.stairRun,
              seg: i, step: s, steps: count,
            });
          }
          cur = { x: cur.x, y: cur.y + count * CFG.stairRise, z: edge - count * CFG.stairRun - CFG.platL / 2 };
          parts.push({ kind: "platform", ...cur, w: CFG.platW, l: CFG.platL, seg: i });
          segs.push({ i, kind, count, exit: { ...cur } });
        }
      }
      return { parts, segs, end: cur };
    }

    function rejectReason(c) {
      for (const s of c.segs) {
        if ((s.kind === "gap" || s.kind === "offset") && s.gap > CFG.maxJumpGap) return "jump gap";
        if (s.kind === "offset" && Math.abs(s.off) > CFG.maxLateralOffset) return "lateral offset";
      }
      if (c.end.y - 4 > CFG.maxCourseRise) return "course rise";
      return null;
    }

    function generate(seed) {
      const rejections = [];
      for (let attempt = 0; attempt < CFG.maxAttempts; attempt++) {
        const c = candidate((seed + attempt * 104729) % 2147483647);
        const reason = rejectReason(c);
        if (!reason) return { ...c, attempt, rejections };
        rejections.push(reason);
      }
      return null;
    }

    const esc = (s) => String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;");

    function partTitle(p, segs) {
      if (p.kind === "beam") {
        const s = segs[p.seg - 1];
        return `segment ${p.seg} · beam — ${s.len.toFixed(1)} studs long, ${s.width.toFixed(1)} wide`;
      }
      if (p.kind === "stair") return `segment ${p.seg} · stairs — step ${p.step} of ${p.steps}`;
      if (p.seg === 0) return "start platform";
      const s = segs[p.seg - 1];
      let d = `segment ${p.seg} · landing platform`;
      if (s.kind === "gap") d += ` — after a ${s.gap.toFixed(1)} stud gap`;
      if (s.kind === "offset") d += ` — after a ${s.gap.toFixed(1)} stud gap at ${s.angle.toFixed(1)}°`;
      if (p.y > 4) d += ` · elevation +${(p.y - 4).toFixed(0)}`;
      return d;
    }

    function render(seed) {
      const course = generate(seed);
      if (!course) return;
      const pad = 5;
      let maxFx = 0, minFy = 0, maxFy = 0;
      course.parts.forEach((p) => {
        maxFx = Math.max(maxFx, -p.z + p.l / 2);
        minFy = Math.min(minFy, p.x - p.w / 2);
        maxFy = Math.max(maxFy, p.x + p.w / 2);
      });
      const vx = -CFG.platL / 2 - pad;
      const vy = minFy - pad - 3;
      const vw = maxFx - vx + pad;
      const vh = maxFy - vy + pad;
      const bits = [];
      course.parts.forEach((p) => {
        const x = -p.z - p.l / 2, y = p.x - p.w / 2;
        const fill = p.kind === "stair" ? COLORS.stair : COLORS[p.kind];
        const op = p.kind === "stair" ? 0.45 + 0.55 * (p.step / p.steps) : 1;
        bits.push(
          `<rect x="${x.toFixed(2)}" y="${y.toFixed(2)}" width="${p.l.toFixed(2)}" height="${p.w.toFixed(2)}" rx="1" fill="${fill}" fill-opacity="${op.toFixed(2)}"><title>${esc(partTitle(p, course.segs))}</title></rect>`
        );
        if (p.kind === "platform" && p.y > 4) {
          bits.push(
            `<text x="${(-p.z).toFixed(2)}" y="${(p.x + 1.2).toFixed(2)}" text-anchor="middle" font-size="3.4" fill="#fff" style="pointer-events:none">+${(p.y - 4).toFixed(0)}</text>`
          );
        }
      });
      const pts = [[0, 0], ...course.segs.map((s) => [-s.exit.z, s.exit.x])];
      bits.push(
        `<polyline points="${pts.map(([a, b]) => `${a.toFixed(1)},${b.toFixed(1)}`).join(" ")}" fill="none" stroke="#566477" stroke-width="0.6" stroke-dasharray="1.8 2.4" stroke-opacity="0.7"/>`
      );
      bits.push(`<circle cx="0" cy="0" r="1.8" fill="none" stroke="#0b1628" stroke-width="0.7"><title>spawn</title></circle>`);
      bits.push(`<circle cx="${(-course.end.z).toFixed(1)}" cy="${course.end.x.toFixed(1)}" r="1.8" fill="#0b1628"><title>finish</title></circle>`);
      svg.setAttribute("viewBox", `${vx.toFixed(1)} ${vy.toFixed(1)} ${vw.toFixed(1)} ${vh.toFixed(1)}`);
      svg.innerHTML = bits.join("");
      const order = course.segs.map((s) => s.kind).join(" → ");
      let extra = ` · ${Math.round(maxFx)} studs`;
      if (course.attempt > 0) extra += ` · layout ${course.attempt + 1} — ${course.rejections.length} rejected (${course.rejections[0]} limit)`;
      readout.innerHTML = `<b>seed ${esc(seed)}</b> · ${esc(order)}${esc(extra)}`;
    }

    const clampSeed = () => Math.min(2147483646, Math.max(0, Math.floor(Number(input.value) || 0)));
    input.addEventListener("input", () => render(clampSeed()));
    dice.addEventListener("click", () => {
      input.value = String(Math.floor(Math.random() * 100000));
      render(clampSeed());
    });
    lab.hidden = false;
    render(clampSeed());
  }

  /* ---------- Episode trace chart ---------- */
  const trace = document.querySelector(".trace");
  if (trace && window.fetch) {
    fetch("media/episode-trace.json")
      .then((r) => (r.ok ? r.json() : null))
      .then((data) => {
        if (!data || !Array.isArray(data.steps) || data.steps.length < 2) return;
        trace.hidden = false;
        initTrace(trace, data);
      })
      .catch(() => {});
  }

  function initTrace(fig, data) {
    const svg = fig.querySelector("svg");
    const tip = fig.querySelector(".trace-tip");
    const metaEl = fig.querySelector(".trace-meta");
    const plot = fig.querySelector(".trace-plot");
    const steps = data.steps;
    const meta = data.meta || {};
    const parts = [];
    if (meta.model) parts.push(String(meta.model).split("/").slice(-2).join("/"));
    if (meta.seed != null) parts.push(`seed ${meta.seed}`);
    if (meta.curriculum_stage != null) parts.push(`stage ${meta.curriculum_stage}`);
    parts.push(meta.completed ? `completed in ${steps.length - 1} decisions` : `${steps.length - 1} decisions, not completed`);
    if (meta.generated) parts.push(`recorded ${meta.generated}`);
    metaEl.textContent = parts.join(" · ") + ".";

    const M = { l: 46, r: 14, t: 20, b: 34 };
    let geom = null;

    function draw() {
      const W = Math.max(320, plot.clientWidth - 22);
      const H = 280;
      const iw = W - M.l - M.r, ih = H - M.t - M.b;
      const maxD = Math.max(...steps.map((s) => s.d)) * 1.08;
      const X = (i) => M.l + (i / (steps.length - 1)) * iw;
      const Y = (d) => M.t + ih - (d / maxD) * ih;
      geom = { W, X, Y };
      const b = [];
      const gridN = 4;
      for (let g = 0; g <= gridN; g++) {
        const v = (maxD * g) / gridN;
        const y = Y(v);
        b.push(`<line x1="${M.l}" y1="${y}" x2="${W - M.r}" y2="${y}" stroke="#d8e2ed" stroke-width="1"/>`);
        b.push(`<text x="${M.l - 8}" y="${y + 4}" text-anchor="end" font-size="11" fill="#566477">${Math.round(v)}</text>`);
      }
      const xStep = steps.length > 60 ? 20 : steps.length > 25 ? 10 : 5;
      for (let i = 0; i < steps.length; i += xStep) {
        b.push(`<text x="${X(i)}" y="${H - 12}" text-anchor="middle" font-size="11" fill="#566477">${i}</text>`);
      }
      b.push(`<text x="${M.l}" y="${11}" font-size="11" fill="#566477">distance to checkpoint (studs)</text>`);
      b.push(`<text x="${W - M.r}" y="${H - 12}" text-anchor="end" font-size="11" fill="#566477">decision step</text>`);
      steps.forEach((s, i) => {
        if (i > 0 && s.cp > steps[i - 1].cp) {
          b.push(`<line x1="${X(i)}" y1="${M.t}" x2="${X(i)}" y2="${M.t + ih}" stroke="#566477" stroke-width="1" stroke-dasharray="3 4" stroke-opacity=".55"/>`);
          const flip = X(i) > W - M.r - 40;
          b.push(`<text x="${X(i) + (flip ? -5 : 5)}" y="${M.t + 11}" text-anchor="${flip ? "end" : "start"}" font-size="11" fill="#566477">CP${s.cp}</text>`);
        }
        if (s.jump) {
          b.push(`<path d="M ${X(i) - 4} ${M.t + ih + 9} l 4 -7 l 4 7 z" fill="#b8790f"><title>jump intent · step ${i}</title></path>`);
        }
      });
      const path = steps.map((s, i) => `${i ? "L" : "M"} ${X(i).toFixed(1)} ${Y(s.d).toFixed(1)}`).join(" ");
      b.push(`<path d="${path}" fill="none" stroke="#2367c9" stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/>`);
      steps.forEach((s, i) => {
        if (i > 0 && s.cp > steps[i - 1].cp) {
          b.push(`<path d="M ${X(i)} ${Y(s.d) - 5} l 5 5 l -5 5 l -5 -5 z" fill="#2367c9" stroke="#fff" stroke-width="2"/>`);
        }
      });
      b.push(`<line class="trace-cross" x1="0" y1="${M.t}" x2="0" y2="${M.t + ih}" stroke="#0b1628" stroke-width="1" stroke-opacity="0" />`);
      svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
      svg.setAttribute("width", W);
      svg.setAttribute("height", H);
      svg.innerHTML = b.join("");
    }

    draw();
    let resizeT;
    window.addEventListener("resize", () => {
      clearTimeout(resizeT);
      resizeT = setTimeout(draw, 150);
    });

    plot.addEventListener("pointermove", (e) => {
      if (!geom) return;
      const rect = svg.getBoundingClientRect();
      const px = ((e.clientX - rect.left) / rect.width) * geom.W;
      const i = Math.max(0, Math.min(steps.length - 1, Math.round(((px - M.l) / (geom.W - M.l - M.r)) * (steps.length - 1))));
      const s = steps[i];
      const cross = svg.querySelector(".trace-cross");
      if (cross) {
        cross.setAttribute("x1", geom.X(i));
        cross.setAttribute("x2", geom.X(i));
        cross.setAttribute("stroke-opacity", "0.35");
      }
      tip.hidden = false;
      tip.textContent = `step ${i} · ${s.d.toFixed(1)} studs · CP${s.cp}${s.jump ? " · jump" : ""}`;
      tip.style.left = `${(geom.X(i) / geom.W) * rect.width + 10}px`;
      tip.style.top = `${(geom.Y(s.d) / 280) * rect.height + 14}px`;
    });
    plot.addEventListener("pointerleave", () => {
      tip.hidden = true;
      const cross = svg.querySelector(".trace-cross");
      if (cross) cross.setAttribute("stroke-opacity", "0");
    });
  }

  /* ---------- Scroll reveals ---------- */
  if (!prefersReduced && "IntersectionObserver" in window) {
    const selectors = [
      ".hero .kicker", ".hero h1", ".hero .abstract", ".hero .hero-links",
      ".section-heading", ".two-col p", "main .media-frame", ".arch",
      ".specs article", ".recipe", ".headline-result", ".ablation-table .table-row",
      ".reference-list a", ".stage-explorer",
    ];
    const els = new Set();
    selectors.forEach((sel) => document.querySelectorAll(sel).forEach((el) => els.add(el)));

    const byParent = new Map();
    els.forEach((el) => {
      const sibs = byParent.get(el.parentElement) || [];
      sibs.push(el);
      byParent.set(el.parentElement, sibs);
    });
    byParent.forEach((sibs) =>
      sibs.forEach((el, i) => {
        el.classList.add("will-reveal");
        el.style.setProperty("--reveal-delay", `${Math.min(i * 70, 420)}ms`);
      })
    );

    const io = new IntersectionObserver(
      (entries) => {
        entries.forEach((e) => {
          if (e.isIntersecting) {
            e.target.classList.add("is-visible");
            io.unobserve(e.target);
          }
        });
      },
      { rootMargin: "0px 0px -60px 0px", threshold: 0.08 }
    );
    els.forEach((el) => io.observe(el));
  }

  /* ---------- Stat count-ups ---------- */
  if (!prefersReduced && "IntersectionObserver" in window) {
    const io = new IntersectionObserver(
      (entries) => {
        entries.forEach((e) => {
          if (!e.isIntersecting) return;
          io.unobserve(e.target);
          const el = e.target;
          const m = el.textContent.trim().match(/^([\d.]+)(.*)$/);
          if (!m) return;
          const target = parseFloat(m[1]);
          const suffix = m[2] || "";
          const decimals = (m[1].split(".")[1] || "").length;
          const dur = 1100;
          const t0 = performance.now();
          const step = (now) => {
            const p = Math.min((now - t0) / dur, 1);
            const eased = 1 - Math.pow(1 - p, 3);
            el.textContent = (target * eased).toFixed(decimals) + suffix;
            if (p < 1) requestAnimationFrame(step);
          };
          requestAnimationFrame(step);
        });
      },
      { threshold: 0.6 }
    );
    document.querySelectorAll("[data-count]").forEach((el) => io.observe(el));
  }
})();
