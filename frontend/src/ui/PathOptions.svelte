<script lang="ts">
    import { focusedPathTool, pathSession } from '../tools/path.svelte';
    import Scrub from './Scrub.svelte';
    import ToolBarLayout from './ToolBarLayout.svelte';

    let tool = $derived(focusedPathTool());
    let editing = $derived(tool?.editing ?? null);
</script>

<ToolBarLayout>
    {#snippet center()}
        <Scrub
            mode="drag"
            label="Width"
            value={pathSession.strokeWidth}
            min={0.5}
            max={64}
            default={2}
            formatValue={(v) => v.toFixed(1)}
            onChange={(v) => { pathSession.strokeWidth = Math.max(0.5, v); }}
            title="Stroke width applied when the path commits, in canvas pixels."
        />
        <button
            type="button"
            class="bar-control"
            class:on={editing?.guide}
            disabled={!editing}
            title="Non-printing guide: the path stops painting, draws dashed, and can constrain the pen."
            onclick={() => tool?.setGuide(!editing?.guide)}
        >Guide</button>
        <div class="mode-group" title="Pen guide mode: how the brush relates to guide paths.">
            {#each ['off', 'wall', 'snap'] as const as mode}
                <button
                    type="button"
                    class="bar-control"
                    class:on={pathSession.guideMode === mode}
                    onclick={() => tool?.setGuideMode(mode)}
                >{mode}</button>
            {/each}
        </div>
        <button
            type="button"
            class="bar-control"
            disabled={!editing}
            title="Convert the edited path's region into the active selection."
            onclick={() => void tool?.toSelection()}
        >Selection</button>
        <button
            type="button"
            class="bar-control"
            disabled={!editing}
            title="Cut the region the edited path bounds out of the active layer."
            onclick={() => void tool?.cutOut()}
        >Cut out</button>
        <button
            type="button"
            class="bar-control"
            disabled={!editing}
            title="Use the edited path's region as the mask on the active layer."
            onclick={() => void tool?.toMask()}
        >Mask</button>
    {/snippet}
</ToolBarLayout>

<style>
    .mode-group {
        display: flex;
        gap: 0;
        border: 1px solid var(--bg-hover);
        border-radius: 4px;
        overflow: hidden;
    }
    .mode-group .bar-control {
        border-radius: 0;
        border: none;
    }
    .mode-group .bar-control.on {
        background: var(--accent);
        color: #ffffff;
    }
    .bar-control.on {
        background: var(--accent);
        color: #ffffff;
    }
</style>
