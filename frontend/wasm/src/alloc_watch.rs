//! Dev-only allocation watchdog: logs a JS stack trace for any single
//! Rust allocation over 64 MB. Debug builds only — release wasm keeps the
//! system allocator directly. Used to hunt wasm-heap OOMs (`rust_oom` →
//! `unreachable`), where the failing alloc's stack is never the culprit.

#[cfg(debug_assertions)]
mod imp {
    use std::alloc::{GlobalAlloc, Layout, System};

    pub struct Watch;

    unsafe impl GlobalAlloc for Watch {
        unsafe fn alloc(&self, l: Layout) -> *mut u8 {
            if l.size() > 64 * 1024 * 1024 {
                let e = js_sys::Error::new("bigalloc");
                let msg = format!("BIGALLOC {}MB align={}", l.size() / (1024 * 1024), l.align());
                let stack = js_sys::Reflect::get(&e, &"stack".into()).unwrap_or_default();
                web_sys::console::warn_2(&msg.into(), &stack);
            }
            unsafe { System.alloc(l) }
        }
        unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
            unsafe { System.dealloc(p, l) }
        }
    }

    #[global_allocator]
    static A: Watch = Watch;
}

#[cfg(debug_assertions)]
#[allow(unused_imports)]
pub use imp::*;
