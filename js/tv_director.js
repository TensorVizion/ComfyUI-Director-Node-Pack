/**
 * TV Director — custom timeline widget.
 *
 * Replaces the hidden `timeline_json` STRING widget on TVDirectorTimeline
 * nodes with a drag/drop shot-card timeline + a per-shot editing panel.
 * On any edit, serializes state back into the widget's value as JSON so
 * it round-trips through the normal ComfyUI prompt/workflow save/load
 * and reaches the Python node's `timeline_json` input unchanged.
 *
 * Design intentionally mirrors the LTX Director UX (horizontal shot
 * strip, click-to-select, inline duration/prompt editing) but stores
 * data in the backend-agnostic DIRECTOR_TIMELINE schema instead of
 * LTX-specific guide objects.
 */

import { app } from "../../scripts/app.js";

const SCHEMA_VERSION = 1;

function emptyTimeline() {
  return {
    schema_version: SCHEMA_VERSION,
    global: {
      fps: 24,
      width: 768,
      height: 512,
      seed: -1,
      global_prompt_prefix: "",
      global_negative_prompt: "",
    },
    audio_tracks: [],
    shots: [],
  };
}

function newShot(id) {
  return {
    id,
    order: 0,
    prompt: "",
    negative_prompt: "",
    duration_frames: 97,
    image_ref: null,
    image_role: "first",
    transition_in: "cut",
    transition_out: "cut",
    audio_track_id: null,
    strength: 1.0,
    camera_hint: "",
  };
}

function uid() {
  return "shot_" + Math.random().toString(36).slice(2, 10);
}

app.registerExtension({
  name: "TensorVizion.Director.Timeline",

  async beforeRegisterNodeDef(nodeType, nodeData, appInstance) {
    if (nodeData.name !== "TVDirectorTimeline") return;

    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      onNodeCreated?.apply(this, arguments);

      const node = this;

      // Locate the raw JSON widget. It stays in node.widgets so ComfyUI's
      // normal save path still serializes its `.value` into
      // widgets_values and sends it to Python on execution — we just
      // make it invisible.
      //
      // The garbled overlapping text in the previous version happened
      // because this is a multiline STRING widget, which ComfyUI backs
      // with a real <textarea> DOM element (`.inputEl`) positioned over
      // the canvas independently of the widget's computeSize/type. That
      // textarea kept rendering behind our DOM widget no matter what
      // computeSize returned. Fix: hide the textarea itself, zero its
      // layout footprint, and no-op its canvas draw call so nothing about
      // it can render, while leaving `.value` fully intact and writable.
      const jsonWidget = node.widgets?.find((w) => w.name === "timeline_json");
      if (jsonWidget) {
        jsonWidget.computeSize = () => [0, -4];
        if (jsonWidget.inputEl) {
          jsonWidget.inputEl.style.display = "none";
        }
        jsonWidget.draw = () => {};
      }


      let state;
      try {
        state = jsonWidget?.value ? JSON.parse(jsonWidget.value) : emptyTimeline();
      } catch (e) {
        state = emptyTimeline();
      }
      if (!state.shots) state.shots = [];
      if (!state.global) state.global = emptyTimeline().global;

      node._tvState = state;
      let selectedShotId = state.shots[0]?.id ?? null;

      function persist() {
        // Re-derive order from current array index every time.
        state.shots.forEach((s, i) => (s.order = i));
        if (jsonWidget) {
          jsonWidget.value = JSON.stringify(state);
        }
        node.setDirtyCanvas(true, true);
      }

      function addShot() {
        const shot = newShot(uid());
        shot.order = state.shots.length;
        state.shots.push(shot);
        selectedShotId = shot.id;
        persist();
        render();
      }

      function removeShot(id) {
        state.shots = state.shots.filter((s) => s.id !== id);
        if (selectedShotId === id) {
          selectedShotId = state.shots[0]?.id ?? null;
        }
        persist();
        render();
      }

      function duplicateShot(id) {
        const src = state.shots.find((s) => s.id === id);
        if (!src) return;
        const copy = { ...src, id: uid() };
        const idx = state.shots.findIndex((s) => s.id === id);
        state.shots.splice(idx + 1, 0, copy);
        selectedShotId = copy.id;
        persist();
        render();
      }

      function moveShot(id, delta) {
        const idx = state.shots.findIndex((s) => s.id === id);
        const newIdx = idx + delta;
        if (newIdx < 0 || newIdx >= state.shots.length) return;
        const [item] = state.shots.splice(idx, 1);
        state.shots.splice(newIdx, 0, item);
        persist();
        render();
      }

      // --- DOM construction -------------------------------------------------
      const container = document.createElement("div");
      container.style.cssText = `
        display: flex;
        flex-direction: column;
        gap: 6px;
        width: 100%;
        font-family: sans-serif;
        font-size: 11px;
        color: #ddd;
        padding: 4px;
        box-sizing: border-box;
      `;

      const toolbar = document.createElement("div");
      toolbar.style.cssText = "display:flex; gap:6px; align-items:center;";
      const addBtn = document.createElement("button");
      addBtn.textContent = "+ Add Shot";
      styleButton(addBtn, "#C6FF3A", "#111");
      addBtn.onclick = (e) => {
        e.stopPropagation();
        addShot();
      };
      const summary = document.createElement("span");
      summary.style.cssText = "color:#888; margin-left:auto;";
      toolbar.appendChild(addBtn);
      toolbar.appendChild(summary);

      const strip = document.createElement("div");
      strip.style.cssText = `
        display:flex;
        gap:4px;
        overflow-x:auto;
        min-height:64px;
        padding:4px;
        background:#181818;
        border-radius:6px;
        border:1px solid #2a2a2a;
      `;

      const editor = document.createElement("div");
      editor.style.cssText = `
        display:flex;
        flex-direction:column;
        gap:4px;
        padding:6px;
        background:#151515;
        border-radius:6px;
        border:1px solid #2a2a2a;
        min-height: 180px;
      `;

      const globalPanel = document.createElement("div");
      globalPanel.style.cssText = `
        display:flex; flex-wrap:wrap; gap:4px; padding:6px;
        background:#101010; border-radius:6px; border:1px solid #262626;
      `;

      container.appendChild(toolbar);
      container.appendChild(strip);
      container.appendChild(editor);
      container.appendChild(globalPanel);

      function styleButton(btn, bg, fg) {
        btn.style.cssText = `
          background:${bg}; color:${fg}; border:none; border-radius:4px;
          padding:3px 8px; font-size:11px; cursor:pointer; font-weight:600;
        `;
      }

      function labeledInput(labelText, value, onChange, opts = {}) {
        const wrap = document.createElement("label");
        wrap.style.cssText = "display:flex; flex-direction:column; gap:2px; flex:1; min-width:90px;";
        const lab = document.createElement("span");
        lab.textContent = labelText;
        lab.style.cssText = "color:#999; font-size:10px;";
        let input;
        if (opts.textarea) {
          input = document.createElement("textarea");
          input.rows = opts.rows || 2;
        } else if (opts.select) {
          input = document.createElement("select");
          opts.select.forEach((optVal) => {
            const o = document.createElement("option");
            o.value = optVal;
            o.textContent = optVal;
            if (optVal === value) o.selected = true;
            input.appendChild(o);
          });
        } else {
          input = document.createElement("input");
          input.type = opts.number ? "number" : "text";
        }
        if (!opts.select) input.value = value ?? "";
        input.style.cssText = `
          background:#0d0d0d; color:#eee; border:1px solid #333; border-radius:4px;
          padding:3px 5px; font-size:11px; width:100%; box-sizing:border-box;
        `;
        input.onclick = (e) => e.stopPropagation();
        input.onchange = (e) => {
          const v = opts.number ? parseFloat(e.target.value) : e.target.value;
          onChange(v);
        };
        wrap.appendChild(lab);
        wrap.appendChild(input);
        return wrap;
      }

      function renderGlobalPanel() {
        globalPanel.innerHTML = "";
        const g = state.global;
        globalPanel.appendChild(labeledInput("FPS", g.fps, (v) => { g.fps = v; persist(); render(); }, { number: true }));
        globalPanel.appendChild(labeledInput("Width", g.width, (v) => { g.width = v; persist(); }, { number: true }));
        globalPanel.appendChild(labeledInput("Height", g.height, (v) => { g.height = v; persist(); }, { number: true }));
        globalPanel.appendChild(labeledInput("Seed", g.seed, (v) => { g.seed = v; persist(); }, { number: true }));
        globalPanel.appendChild(labeledInput("Global prompt prefix", g.global_prompt_prefix, (v) => { g.global_prompt_prefix = v; persist(); }));
        globalPanel.appendChild(labeledInput("Global negative", g.global_negative_prompt, (v) => { g.global_negative_prompt = v; persist(); }));
      }

      function renderStrip() {
        strip.innerHTML = "";
        state.shots.forEach((shot, i) => {
          const card = document.createElement("div");
          const isSelected = shot.id === selectedShotId;
          card.draggable = true;
          card.style.cssText = `
            min-width:96px; max-width:96px; height:56px;
            background:${isSelected ? "#2a3a12" : "#222"};
            border:1px solid ${isSelected ? "#C6FF3A" : "#3a3a3a"};
            border-radius:5px; padding:4px; cursor:pointer;
            display:flex; flex-direction:column; justify-content:space-between;
            flex-shrink:0;
          `;
          const title = document.createElement("div");
          title.textContent = `#${i + 1} ${shot.image_role}`;
          title.style.cssText = "font-size:10px; font-weight:600; color:#C6FF3A; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;";
          const promptPreview = document.createElement("div");
          promptPreview.textContent = shot.prompt || "(no prompt)";
          promptPreview.style.cssText = "font-size:9px; color:#aaa; overflow:hidden; text-overflow:ellipsis; white-space:nowrap;";
          const dur = document.createElement("div");
          dur.textContent = `${shot.duration_frames}f`;
          dur.style.cssText = "font-size:9px; color:#777;";

          card.appendChild(title);
          card.appendChild(promptPreview);
          card.appendChild(dur);

          card.onclick = (e) => {
            e.stopPropagation();
            selectedShotId = shot.id;
            render();
          };

          card.ondragstart = (e) => {
            e.dataTransfer.setData("text/plain", String(i));
          };
          card.ondragover = (e) => e.preventDefault();
          card.ondrop = (e) => {
            e.preventDefault();
            const fromIdx = parseInt(e.dataTransfer.getData("text/plain"), 10);
            const toIdx = i;
            if (fromIdx === toIdx || isNaN(fromIdx)) return;
            const [item] = state.shots.splice(fromIdx, 1);
            state.shots.splice(toIdx, 0, item);
            persist();
            render();
          };

          strip.appendChild(card);
        });

        if (state.shots.length === 0) {
          const hint = document.createElement("div");
          hint.textContent = "No shots yet — click \"+ Add Shot\" to start your timeline.";
          hint.style.cssText = "color:#666; padding:8px; font-size:11px;";
          strip.appendChild(hint);
        }
      }

      function renderEditor() {
        editor.innerHTML = "";
        const shot = state.shots.find((s) => s.id === selectedShotId);
        if (!shot) {
          editor.textContent = "Select a shot above to edit it.";
          editor.style.color = "#666";
          return;
        }

        const row1 = document.createElement("div");
        row1.style.cssText = "display:flex; gap:6px;";
        row1.appendChild(labeledInput("Prompt", shot.prompt, (v) => { shot.prompt = v; persist(); renderStrip(); }, { textarea: true, rows: 2 }));
        row1.appendChild(labeledInput("Negative prompt", shot.negative_prompt, (v) => { shot.negative_prompt = v; persist(); }, { textarea: true, rows: 2 }));

        const row2 = document.createElement("div");
        row2.style.cssText = "display:flex; gap:6px; flex-wrap:wrap;";
        row2.appendChild(labeledInput("Duration (frames)", shot.duration_frames, (v) => { shot.duration_frames = v; persist(); renderStrip(); updateSummary(); }, { number: true }));
        row2.appendChild(labeledInput("Image role", shot.image_role, (v) => { shot.image_role = v; persist(); renderStrip(); }, { select: ["first", "last", "middle", "reference"] }));
        row2.appendChild(labeledInput("Image ref index", shot.image_ref ?? "", (v) => { shot.image_ref = v === "" ? null : parseInt(v, 10); persist(); }, { number: true }));
        row2.appendChild(labeledInput("Strength", shot.strength, (v) => { shot.strength = v; persist(); }, { number: true }));

        const row3 = document.createElement("div");
        row3.style.cssText = "display:flex; gap:6px; flex-wrap:wrap;";
        row3.appendChild(labeledInput("Transition in", shot.transition_in, (v) => { shot.transition_in = v; persist(); }, { select: ["cut", "crossfade", "hold", "morph"] }));
        row3.appendChild(labeledInput("Transition out", shot.transition_out, (v) => { shot.transition_out = v; persist(); }, { select: ["cut", "crossfade", "hold", "morph"] }));
        row3.appendChild(labeledInput("Camera hint", shot.camera_hint, (v) => { shot.camera_hint = v; persist(); }));
        row3.appendChild(labeledInput("Audio track id", shot.audio_track_id ?? "", (v) => { shot.audio_track_id = v || null; persist(); }));

        const actions = document.createElement("div");
        actions.style.cssText = "display:flex; gap:6px; margin-top:4px;";
        const mkBtn = (label, fn, bg = "#333", fg = "#eee") => {
          const b = document.createElement("button");
          b.textContent = label;
          styleButton(b, bg, fg);
          b.onclick = (e) => { e.stopPropagation(); fn(); };
          return b;
        };
        actions.appendChild(mkBtn("◀ Move left", () => moveShot(shot.id, -1)));
        actions.appendChild(mkBtn("Move right ▶", () => moveShot(shot.id, 1)));
        actions.appendChild(mkBtn("Duplicate", () => duplicateShot(shot.id)));
        actions.appendChild(mkBtn("Delete", () => removeShot(shot.id), "#5a1a1a", "#ffaaaa"));

        editor.appendChild(row1);
        editor.appendChild(row2);
        editor.appendChild(row3);
        editor.appendChild(actions);
      }

      function updateSummary() {
        const totalFrames = state.shots.reduce((sum, s) => sum + (s.duration_frames || 0), 0);
        const fps = state.global.fps || 24;
        const seconds = (totalFrames / fps).toFixed(1);
        summary.textContent = `${state.shots.length} shots · ${totalFrames}f · ~${seconds}s @ ${fps}fps`;
      }

      function render() {
        renderStrip();
        renderEditor();
        renderGlobalPanel();
        updateSummary();
        persist();
      }

      render();

      // Mount as a DOM widget on the node.
      node.addDOMWidget("tvdirector_ui", "TV Director Timeline", container, {
        serialize: false,
        hideOnZoom: false,
      });

      node.setSize([Math.max(node.size[0], 480), Math.max(node.size[1], 420)]);
    };
  },
});
