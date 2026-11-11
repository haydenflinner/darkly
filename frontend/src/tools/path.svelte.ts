/**
 * Path tool: draws an S9 interpolating spline (Aurimas Gasiulis's
 * curvature-law spline — solved by the `spline` crate, MIT,
 * <https://gasiulis.name/new-spline/>). Knots are the editable geometry:
 * the committed vector object keeps them (`ObjectSource::Spline`), so a
 * path stays re-editable after commit.
 *
 * Interaction follows the polygon-select convention
 * (krita/libs/ui/tool/kis_tool_polyline_base.cpp):
 *   - Click:                  place a knot.
 *   - Press on a knot:        drag it (re-solves live).
 *   - Click on first knot:    close the path and commit.
 *   - Double-click / Enter:   commit the path as drawn.
 *   - Alt+drag on an end knot of an open path: pull out an end-tangent
 *                             control (S9's ends).
 *   - C:                      toggle closed on the in-progress path.
 *   - Backspace:              remove the last placed knot.
 *   - Escape:                 cancel the in-progress path / exit editing.
 *   - Click on a committed spline object: re-enter knot editing on it
 *     (writes through `set_spline_knots`, one coalesced undo lane).
 */
import { ToolBase, type ToolDescriptor } from './registry';
import { getActiveInstance, type DarklyInstance } from '../state/app.svelte';
import { OverlayBuilder } from '../canvas/gpu_overlay';
import { actions } from '../actions/registry';
import PathOptions from '../ui/PathOptions.svelte';

/** Hit radius in buffer px — Krita's 10px constant, kept constant on screen
 *  by dividing by zoom. */
const HIT_RADIUS_BUFFER_PX = 10;

const DBLCLICK_MS = 400;
const DBLCLICK_RADIUS_BUFFER_PX = 6;

interface Editing {
    layerId: number;
    objectId: number;
    /** The object's non-printing flag — the Guide toggle's bound state. */
    guide: boolean;
}

/** Creation defaults for the path tool: an app-global artist preference,
 *  written by the options strip and read when a path commits. */
class PathSession {
    /** Stroke width in canvas px. */
    strokeWidth = $state(2);
    /** Selection-combine mode the shape ops use when no modifier keys are
     *  held — `replace` is the lasso default. */
    selectionMode = $state<'replace' | 'add' | 'subtract' | 'intersect'>('replace');
    /** Pen-guide engagement — `wall` refuses crossings, `snap` rides the
     *  nearest guide within radius. Engine-side session state, mirrored
     *  here so the options strip can bind it. */
    guideMode = $state<'off' | 'wall' | 'snap'>('off');
}

export const pathSession = new PathSession();

/** Which handle a press grabbed: a knot index, or an end-tangent index
 *  (0 = start control, 1 = end control). */
type Grab = { kind: 'knot'; i: number } | { kind: 'end'; i: 0 | 1 };

class PathTool extends ToolBase {
    private knots: [number, number][] = [];
    private ends: [[number, number], [number, number]] | null = null;
    private closed = false;
    private cursor: [number, number] | null = null;
    /** The committed object being knot-edited — reactive so the options
     *  strip enables its shape ops only while one is live. */
    editing = $state<Editing | null>(null);

    /** The press owning the current drag (knot/end/new-knot), by pointerId.
     *  A second contact never re-opens it — the single-owner rule the
     *  gesture fix established for declarative drags. */
    private owner: { pointerId: number; grab: Grab } | null = null;
    private lastClickTime = 0;
    private lastClickPos: [number, number] | null = null;

    /** Async preview ordering: the latest knots solve wins; an earlier
     *  `splineSolve` resolving late must not paint over it. */
    private previewSeq = 0;
    /** Knot-drag write coalescing: posts are throttled to one per frame so
     *  a fast drag doesn't queue a document write per move event. */
    private writeQueued = false;

    private hitRadius(): number {
        return HIT_RADIUS_BUFFER_PX / this.inst.zoom;
    }

    private knotAt(p: [number, number]): Grab | null {
        const r = this.hitRadius();
        const r2 = r * r;
        // Ends handles take precedence over their anchor knots when both
        // are under the pointer (an end control sits near its knot).
        if (this.ends) {
            for (const i of [0, 1] as const) {
                const [x, y] = this.ends[i];
                const dx = p[0] - x;
                const dy = p[1] - y;
                if (dx * dx + dy * dy <= r2) return { kind: 'end', i };
            }
        }
        for (let i = this.knots.length - 1; i >= 0; i--) {
            const dx = p[0] - this.knots[i][0];
            const dy = p[1] - this.knots[i][1];
            if (dx * dx + dy * dy <= r2) return { kind: 'knot', i };
        }
        return null;
    }

    private cursorOnFirstKnot(): boolean {
        if (this.editing || this.knots.length < 3 || !this.cursor) return false;
        const [fx, fy] = this.knots[0];
        const dx = this.cursor[0] - fx;
        const dy = this.cursor[1] - fy;
        const r = this.hitRadius();
        return dx * dx + dy * dy <= r * r;
    }

    /** Re-solve the current knots and push the overlay. The engine returns
     *  the flattened polyline (what the renderer will draw), so the preview
     *  is exactly the committed shape — no JS-side curve math. */
    private async refreshPreview(): Promise<void> {
        const engine = this.engine;
        const canvasEl = this.canvasEl;
        if (!engine || !canvasEl) return;
        const seq = ++this.previewSeq;
        const solved = this.knots.length >= 2
            ? await engine.api.splineSolve({
                  knots: this.knots,
                  closed: this.closed,
                  ends: this.closed ? undefined : (this.ends ?? undefined),
              })
            : null;
        if (seq !== this.previewSeq) return; // a newer solve is in flight
        const o = new OverlayBuilder(canvasEl);
        const accent = { color: '#4af' as const, thickness: 2 };
        if (solved) {
            const s = solved.samples;
            for (let i = 1; i < s.length; i++) o.line(s[i - 1], s[i], accent);
            if (this.closed && s.length > 1) o.line(s[s.length - 1], s[0], accent);
        } else {
            // <2 knots: draw the raw chain so a single segment still shows.
            for (let i = 1; i < this.knots.length; i++) {
                o.line(this.knots[i - 1], this.knots[i], accent);
            }
        }
        // Rubber-band from the last knot to the cursor while drawing.
        if (!this.editing && !this.closed && this.cursor && this.knots.length > 0) {
            o.line(this.knots[this.knots.length - 1], this.cursor, {
                color: '#4af', thickness: 1, dash: 4,
            });
        }
        // End-tangent controls: a handle line from the anchor knot to the
        // control point (open paths only — they have no meaning closed).
        if (!this.closed && this.ends && this.knots.length >= 2) {
            const first = this.knots[0];
            const last = this.knots[this.knots.length - 1];
            o.line(first, this.ends[0], { color: '#fa4', thickness: 1, dash: 4 });
            o.line(last, this.ends[1], { color: '#fa4', thickness: 1, dash: 4 });
            o.handle(this.ends[0], { fill: '#fff', stroke: '#fa4' });
            o.handle(this.ends[1], { fill: '#fff', stroke: '#fa4' });
        }
        for (const k of this.knots) o.handle(k, { fill: '#fff', stroke: '#4af' });
        // Close indicator on the first knot when the cursor is in its zone.
        if (this.cursorOnFirstKnot()) {
            const r = this.hitRadius();
            o.line(
                [this.knots[0][0] - r, this.knots[0][1]],
                [this.knots[0][0] + r, this.knots[0][1]],
                { color: '#4af', thickness: 1 },
            );
            o.line(
                [this.knots[0][0], this.knots[0][1] - r],
                [this.knots[0][0], this.knots[0][1] + r],
                { color: '#4af', thickness: 1 },
            );
        }
        o.push(engine);
        this.inst.requestFrame();
    }

    private clearOverlay(): void {
        this.previewSeq++;
        this.inst.engine?.api.clearOverlay();
        this.inst.requestFrame();
    }

    private clearState(): void {
        this.knots = [];
        this.ends = null;
        this.closed = false;
        this.cursor = null;
        this.editing = null;
        this.owner = null;
        this.writeQueued = false;
        this.clearOverlay();
    }

    /** Write the current edit set back to the committed object (drag-time
     *  path). Throttled to one post per frame; every post lands on the same
     *  undo lane, so a whole knot drag coalesces to one step. */
    private queueKnotWrite(): void {
        if (!this.editing || this.writeQueued) return;
        this.writeQueued = true;
        requestAnimationFrame(() => {
            this.writeQueued = false;
            const e = this.editing;
            const engine = this.inst.engine;
            if (!e || !engine) return;
            engine.api.setSplineKnots({
                id: e.layerId,
                object: e.objectId,
                knots: this.knots,
                closed: this.closed,
                ends: this.ends ?? undefined,
            });
        });
    }

    /** Write the edit set back immediately — used at drag release and for
     *  discrete edits (close toggle, knot delete) that don't get a second
     *  chance to flush. */
    private writeNow(): void {
        const e = this.editing;
        const engine = this.inst.engine;
        if (!e || !engine) return;
        this.writeQueued = false;
        engine.api.setSplineKnots({
            id: e.layerId,
            object: e.objectId,
            knots: this.knots,
            closed: this.closed,
            ends: this.closed ? undefined : (this.ends ?? undefined),
        });
    }

    /** Commit the in-progress path: onto the active vector layer when one is
     *  selected, else a new vector layer above the active node. */
    private async commit(): Promise<void> {
        const engine = this.engine;
        if (!engine) return;
        if (this.knots.length < 2 || (this.closed && this.knots.length < 3)) {
            this.clearState();
            return;
        }
        const knots = this.knots;
        const closed = this.closed;
        const ends = this.ends;
        const c = this.inst.consumeForeground();
        const color: [number, number, number, number] = [c.r, c.g, c.b, c.a];
        const stroke_w = pathSession.strokeWidth;
        const active = this.inst.activeNode;
        const target = active?.type === 'vector' ? active.id : null;
        const endsArg = closed ? undefined : (ends ?? undefined);

        let layerId: number;
        let objectId: number;
        if (target !== null) {
            const res = await engine.api.addSplineObject({
                id: target, knots, closed,
                ends: endsArg, color, stroke_w,
            });
            if (!res || res.object < 0) return;
            layerId = target;
            objectId = res.object;
        } else {
            const res = await engine.api.addSpline({
                knots, closed, ends: endsArg, color, stroke_w,
                anchor: this.inst.activeLayerId ?? -1,
            });
            if (!res) return;
            layerId = res.id;
            objectId = res.object;
            this.inst.selectLayer(layerId);
        }
        // Stay in knot-editing on the committed object: placing the last
        // knot is rarely the final shape, and re-entering is one click.
        this.editing = { layerId, objectId, guide: false };
        this.inst.requestFrame();
    }

    /** Flip the edited object's guide flag: it stops painting (the layer
     *  scene skips it), draws as a dashed overlay, and — while a guide
     *  mode is engaged — constrains brush strokes. Undoable via the
     *  object's own undo lane. */
    setGuide(on: boolean): void {
        const e = this.editing;
        if (!e) return;
        this.inst.engine?.api.setObjectGuide({ id: e.layerId, object: e.objectId, guide: on });
        this.editing = { ...e, guide: on };
    }

    /** Engage the pen guide for all strokes: `wall` refuses crossings of
     *  every guide object, `snap` pulls the tip onto the nearest one.
     *  Session-level, not per-object. */
    setGuideMode(mode: 'off' | 'wall' | 'snap'): void {
        pathSession.guideMode = mode;
        this.inst.engine?.api.setGuideMode({ mode });
    }

    /** Turn the edited object's shape into a selection: the shared first
     *  step of the shape ops (cutout and mask compose on it). The polygon
     *  derives from the object's own layer+affine; `false` when nothing is
     *  being edited or the engine rejects the shape. */
    async toSelection(): Promise<boolean> {
        const engine = this.inst.engine;
        const e = this.editing;
        if (!engine || !e) return false;
        const res = await engine.api.objectToSelection({
            id: e.layerId,
            object: e.objectId,
            mode: pathSession.selectionMode,
            antialias: true,
        });
        return res === true;
    }

    /** Cut the region the edited path bounds out of the active layer —
     *  "cutout the path": shape → selection, then the normal clear. */
    async cutOut(): Promise<void> {
        if (!(await this.toSelection())) return;
        actions.dispatch('clearSelectionContents');
    }

    /** Use the edited path's region as the mask on the active layer —
     *  shape → selection → `selection_to_mask` (mask created if absent). */
    async toMask(): Promise<void> {
        if (!(await this.toSelection())) return;
        const host = this.inst.activeLayerId;
        if (host === null) return;
        this.inst.engine?.api.selectionToMask({ id: host });
    }

    /** Re-enter editing on a committed spline object under the pointer. */
    private async tryEditExisting(cx: number, cy: number): Promise<boolean> {
        const engine = this.engine;
        const layerId = this.inst.activeLayerId;
        if (!engine || layerId === null || this.inst.activeNode?.type !== 'vector') {
            return false;
        }
        const hit = await engine.api.hitTestVectorObject({ id: layerId, x: cx, y: cy });
        if (!hit || hit.object < 0) return false;
        const res = await engine.api.splineObjects({ id: layerId });
        const found = res?.objects.find((o) => o.object === hit.object);
        if (!found) return false;
        this.editing = { layerId, objectId: found.object, guide: found.guide };
        this.knots = found.knots.map((k) => [k[0], k[1]] as [number, number]);
        this.closed = found.closed;
        this.ends = found.ends
            ? [[found.ends[0][0], found.ends[0][1]], [found.ends[1][0], found.ends[1][1]]]
            : null;
        await this.refreshPreview();
        return true;
    }

    onActivate(): void {
        // Vello's one-time pipeline compile (>1s) overlaps tool selection
        // instead of stalling the first commit frame.
        this.inst.engine?.api.warmVectorRenderer();
    }

    onDeactivate(): void {
        this.clearState();
    }

    claimsPointer(): boolean {
        // The tool owns the canvas while active: knot drags and click
        // placement must not be preempted by global drag chords.
        return true;
    }

    async onPointerDown(e: PointerEvent, cx: number, cy: number): Promise<void> {
        if (e.button !== 0 || this.owner) return;
        this.cursor = [cx, cy];
        const p: [number, number] = [cx, cy];

        // A committed-object edit already owns the knots: a press near a
        // handle drags; a press elsewhere adds nothing (the object is
        // fixed-size geometry until the artist re-enters drawing).
        const grab = this.knotAt(p);
        if (grab) {
            this.owner = { pointerId: e.pointerId, grab };
            // Alt+press on an end knot of an open path pulls out an
            // end-tangent control instead of moving the knot.
            if (e.altKey && !this.closed && grab.kind === 'knot' && this.knots.length >= 2) {
                const isEnd = grab.i === 0 || grab.i === this.knots.length - 1;
                if (isEnd) {
                    const i: 0 | 1 = grab.i === 0 ? 0 : 1;
                    const anchor = this.knots[grab.i];
                    const dir: [number, number] = i === 0
                        ? [anchor[0] - this.knots[1][0], anchor[1] - this.knots[1][1]]
                        : [anchor[0] - this.knots[this.knots.length - 2][0],
                           anchor[1] - this.knots[this.knots.length - 2][1]];
                    const len = Math.hypot(dir[0], dir[1]) || 1;
                    const k = 0.4;
                    const ends = this.ends ?? [[...anchor], [...anchor]];
                    ends[i] = [anchor[0] + (dir[0] / len) * len * k, anchor[1] + (dir[1] / len) * len * k];
                    this.ends = ends as [[number, number], [number, number]];
                    this.owner = { pointerId: e.pointerId, grab: { kind: 'end', i } };
                }
            }
            return;
        }

        if (this.editing) {
            // Clicking outside a committed object's handles ends the edit
            // and falls through to a fresh path on the next press.
            this.editing = null;
            this.clearState();
            return;
        }

        // Not drawing and nothing grabbed: a press on a committed spline
        // object re-enters knot editing on it — and if the press lands on
        // one of its handles, the same press already drags it.
        if (this.knots.length === 0 && (await this.tryEditExisting(cx, cy))) {
            const g = this.knotAt(p);
            if (g) this.owner = { pointerId: e.pointerId, grab: g };
            return;
        }

        // Close on first-knot click, double-click commits as drawn.
        if (this.cursorOnFirstKnot()) {
            this.closed = true;
            await this.commit();
            return;
        }
        const now = e.timeStamp;
        if (this.knots.length >= 2 && this.isDoubleClick(cx, cy, now)) {
            await this.commit();
            return;
        }

        this.knots.push(p);
        this.lastClickTime = now;
        this.lastClickPos = p;
        await this.refreshPreview();
    }

    private isDoubleClick(cx: number, cy: number, now: number): boolean {
        if (this.lastClickPos === null) return false;
        if (now - this.lastClickTime > DBLCLICK_MS) return false;
        const dx = cx - this.lastClickPos[0];
        const dy = cy - this.lastClickPos[1];
        const r = DBLCLICK_RADIUS_BUFFER_PX / this.inst.zoom;
        return dx * dx + dy * dy <= r * r;
    }

    onPointerMove(e: PointerEvent, cx: number, cy: number): void {
        this.cursor = [cx, cy];
        if (this.owner && e.pointerId !== this.owner.pointerId) return;
        if (this.owner) {
            const g = this.owner.grab;
            if (g.kind === 'knot') {
                this.knots[g.i] = [cx, cy];
            } else if (this.ends) {
                this.ends[g.i] = [cx, cy];
            }
            void this.refreshPreview();
            if (this.editing) this.queueKnotWrite();
            return;
        }
        if (this.knots.length > 0) void this.refreshPreview();
    }

    onPointerUp(e: PointerEvent): void {
        if (!this.owner || e.pointerId !== this.owner.pointerId) return;
        this.owner = null;
        // Flush the final drag position: the frame throttle may still hold
        // one pending write, and this last point is the one that matters.
        if (this.editing) this.writeNow();
    }

    onPointerLeave(): void {
        this.cursor = null;
        if (this.knots.length > 0) void this.refreshPreview();
    }

    onKeyDown(e: KeyboardEvent): boolean {
        if (e.key === 'Enter') {
            void this.commit();
            return true;
        }
        if (e.key === 'c' || e.key === 'C') {
            // Toggle open/closed. In edit mode the write goes through the
            // same coalesced lane as a knot drag; ends controls become
            // meaningless on a closed path and are dropped.
            if (this.knots.length >= 3) {
                this.closed = !this.closed;
                if (this.closed) this.ends = null;
                void this.refreshPreview();
                if (this.editing) this.writeNow();
            }
            return true;
        }
        if (e.key === 'Backspace') {
            if (this.knots.length === 0) return false;
            // A committed object can't drop below its solvable minimum —
            // deleting the object is the layer panel's job.
            const min = this.editing ? (this.closed ? 3 : 2) : 0;
            if (this.knots.length <= min) return true;
            this.knots.pop();
            if (this.knots.length === 0) {
                this.clearState();
            } else {
                void this.refreshPreview();
                if (this.editing) this.writeNow();
            }
            return true;
        }
        if (e.key === 'Escape') {
            if (this.editing || this.knots.length > 0) {
                this.clearState();
                return true;
            }
            return false;
        }
        return false;
    }

    dismissOverlay(): void {
        this.clearOverlay();
    }
}

/** The focused instance's path tool, if the path tool is the active tool.
 *  The options strip reads `editing` (reactive) and calls the shape ops
 *  through this — the `focusedTextTool` pattern. */
export function focusedPathTool(): PathTool | null {
    const inst = getActiveInstance();
    if (!inst) return null;
    const t = inst.tool('path');
    return t instanceof PathTool ? t : null;
}

export const pathTool: ToolDescriptor = {
    id: 'path',
    group: 'paint',
    optionsComponent: PathOptions,
    create: (inst: DarklyInstance) => new PathTool(inst),
};
