//! Explicit frees for transient GPU resources.
//!
//! On the web, wgpu's `Drop` for a texture or a buffer is a no-op (see
//! wgpu's `backend/webgpu.rs`): the `GPUTexture` JS object stays alive,
//! and so does its GPU memory, until the JS garbage collector finds the
//! wrapper. V8 does not see that memory, so it is in no hurry. A stroke
//! replay drops one scratch and one pre-stroke snapshot per stroke, and
//! those dead textures piled up into gigabytes in Chrome's GPU process
//! before a GC ran (gpu-commons-oom: 2.8 GB of them on a 1517-stroke
//! commons). Native wgpu frees on drop by itself.
//!
//! [`OwnedTexture`] and [`OwnedBuffer`] hold a resource that has exactly
//! one owner. When the owner drops one, it goes on a per-thread list, and
//! [`destroy_retired`] calls `destroy()` on everything listed. A dropped
//! texture cannot be destroyed at once: the encoder that copied it into
//! its grown successor (`Scratch::grow_write`) is not yet submitted, and
//! WebGPU rejects a submit that names a destroyed texture. The host calls
//! [`destroy_retired`] where no encoder is open: after a request drain and
//! after a frame. Work already submitted is safe; WebGPU frees the memory
//! when that work finishes.
//!
//! Native builds drop at once by default (wgpu-core defers the free
//! itself). [`set_deferred`] turns the web behaviour on for one thread so
//! a native test can prove that no destroyed resource reaches a later
//! submit: wgpu's validation then panics.

use std::cell::{Cell, RefCell};
use std::ops::Deref;

enum Retired {
    Texture(wgpu::Texture),
    Buffer(wgpu::Buffer),
}

thread_local! {
    static DEFERRED: Cell<bool> = const { Cell::new(cfg!(target_arch = "wasm32")) };
    static RETIRED: RefCell<Vec<Retired>> = const { RefCell::new(Vec::new()) };
}

/// Turn deferred destruction on or off for this thread. On by default on
/// wasm and off on native. A native test turns it on to run the web
/// path; it must then call [`destroy_retired`] between engine calls.
pub fn set_deferred(on: bool) {
    DEFERRED.with(|d| d.set(on));
    if !on {
        RETIRED.with(|r| r.borrow_mut().clear());
    }
}

fn retire(item: Retired) {
    if DEFERRED.with(Cell::get) {
        RETIRED.with(|r| r.borrow_mut().push(item));
    }
    // Otherwise `item` drops here, the plain native path.
}

/// Destroy every resource retired on this thread since the last call.
/// Call it only where no command encoder is open. Returns how many
/// resources it destroyed.
pub fn destroy_retired() -> usize {
    let items = RETIRED.with(|r| std::mem::take(&mut *r.borrow_mut()));
    for item in &items {
        match item {
            Retired::Texture(t) => t.destroy(),
            Retired::Buffer(b) => b.destroy(),
        }
    }
    items.len()
}

/// A texture with exactly one owner. Dropping it retires the texture, and
/// the next [`destroy_retired`] frees its GPU memory. Not `Clone`: a
/// second handle would outlive the destroy. Reads go through `Deref`,
/// so a `&OwnedTexture` passes where a `&wgpu::Texture` is wanted.
pub struct OwnedTexture(Option<wgpu::Texture>);

impl OwnedTexture {
    pub fn new(texture: wgpu::Texture) -> Self {
        Self(Some(texture))
    }
}

impl From<wgpu::Texture> for OwnedTexture {
    fn from(texture: wgpu::Texture) -> Self {
        Self::new(texture)
    }
}

impl Deref for OwnedTexture {
    type Target = wgpu::Texture;
    fn deref(&self) -> &wgpu::Texture {
        self.0
            .as_ref()
            .expect("an OwnedTexture holds its texture until drop")
    }
}

// r[impl ink.replay.frees]
impl Drop for OwnedTexture {
    fn drop(&mut self) {
        if let Some(t) = self.0.take() {
            retire(Retired::Texture(t));
        }
    }
}

/// A buffer with exactly one owner, retired on drop like [`OwnedTexture`].
/// A pending `map_async` on a destroyed buffer fails, and the dropped
/// owner never reads that result.
pub struct OwnedBuffer(Option<wgpu::Buffer>);

impl OwnedBuffer {
    pub fn new(buffer: wgpu::Buffer) -> Self {
        Self(Some(buffer))
    }
}

impl From<wgpu::Buffer> for OwnedBuffer {
    fn from(buffer: wgpu::Buffer) -> Self {
        Self::new(buffer)
    }
}

impl Deref for OwnedBuffer {
    type Target = wgpu::Buffer;
    fn deref(&self) -> &wgpu::Buffer {
        self.0
            .as_ref()
            .expect("an OwnedBuffer holds its buffer until drop")
    }
}

impl Drop for OwnedBuffer {
    fn drop(&mut self) {
        if let Some(b) = self.0.take() {
            retire(Retired::Buffer(b));
        }
    }
}
