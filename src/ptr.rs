//! Fortran-style array views used by the transcribed routines.
//!
//! Reference-LAPACK passes array sections (`A( I, I+1 )`) that alias the same
//! storage in disjoint read/write patterns a borrow checker cannot express
//! (`DTRMM( .., T( L+1, L+1 ), LDT, T( 1, L+1 ), LDT )` reads and writes the
//! same columns in different rows). The transcriptions therefore work on a
//! `Copy` pointer view, exactly as the Fortran does. Every public entry point
//! checks slice lengths up front; debug builds additionally bounds-check every
//! element access against the length of the slice the view was created from.
//!
//! Indices passed to [`V::at`]/[`V::set`] are 0-based offsets. The
//! transcriptions keep Fortran's 1-based loop variables and convert with
//! [`ix`] (`A( I, J )` is `a.at(ix(i, j, lda))`).

#[derive(Clone, Copy)]
pub struct V<T: Copy> {
    ptr: *mut T,
    #[cfg(debug_assertions)]
    len: usize,
}

// SAFETY: `V` is a raw view. The routines only share it across rayon tasks
// that write disjoint elements (distinct rows or columns of an element-wise
// kernel), which is the Fortran aliasing contract the transcription relies on.
unsafe impl<T: Copy> Send for V<T> {}
unsafe impl<T: Copy> Sync for V<T> {}

impl<T: Copy> V<T> {
    #[inline(always)]
    pub fn new(s: &mut [T]) -> Self {
        V {
            ptr: s.as_mut_ptr(),
            #[cfg(debug_assertions)]
            len: s.len(),
        }
    }

    /// The view shifted by `off` elements (Fortran `A( I, J )` as an argument).
    #[inline(always)]
    pub(crate) fn off(self, off: usize) -> Self {
        #[cfg(debug_assertions)]
        {
            // A zero-length section at the very end is legal in Fortran
            // (e.g. `A( MIN( I+2, N ), I )` with N-I = 1 and no elements read).
            debug_assert!(off <= self.len, "view offset {off} beyond length {}", self.len);
            V { ptr: self.ptr.wrapping_add(off), len: self.len - off }
        }
        #[cfg(not(debug_assertions))]
        V { ptr: self.ptr.wrapping_add(off) }
    }

    /// Read element `i`.
    ///
    /// # Safety
    /// `i` must be inside the slice this view was created from.
    #[inline(always)]
    pub(crate) unsafe fn at(self, i: usize) -> T {
        #[cfg(debug_assertions)]
        debug_assert!(i < self.len, "read {i} beyond length {}", self.len);
        // SAFETY: caller guarantees `i` is in bounds.
        unsafe { *self.ptr.add(i) }
    }

    /// Write element `i`.
    ///
    /// # Safety
    /// `i` must be inside the slice this view was created from, and no other
    /// task may be reading or writing the same element concurrently.
    #[inline(always)]
    pub(crate) unsafe fn set(self, i: usize, v: T) {
        #[cfg(debug_assertions)]
        debug_assert!(i < self.len, "write {i} beyond length {}", self.len);
        // SAFETY: caller guarantees `i` is in bounds and unaliased for writes.
        unsafe { *self.ptr.add(i) = v }
    }
}

impl<T: Copy> V<T> {
    /// The next `len` elements as a shared slice.
    ///
    /// # Safety
    /// In bounds, and no element of the range is written through another
    /// view while the slice lives (Fortran's no-aliasing contract).
    #[inline(always)]
    pub(crate) unsafe fn slice<'a>(self, len: usize) -> &'a [T] {
        #[cfg(debug_assertions)]
        debug_assert!(len <= self.len, "slice {len} beyond length {}", self.len);
        // SAFETY: caller guarantees bounds and aliasing.
        unsafe { std::slice::from_raw_parts(self.ptr, len) }
    }

    /// The next `len` elements as a mutable slice.
    ///
    /// # Safety
    /// In bounds, and the range is not read or written through any other
    /// view or slice while this slice lives.
    #[inline(always)]
    #[allow(clippy::mut_from_ref)]
    pub(crate) unsafe fn slice_mut<'a>(self, len: usize) -> &'a mut [T] {
        #[cfg(debug_assertions)]
        debug_assert!(len <= self.len, "slice {len} beyond length {}", self.len);
        // SAFETY: caller guarantees bounds and exclusivity.
        unsafe { std::slice::from_raw_parts_mut(self.ptr, len) }
    }
}

/// 0-based offset of the 1-based column-major element `( i, j )`.
#[inline(always)]
pub(crate) const fn ix(i: usize, j: usize, lda: usize) -> usize {
    (i - 1) + (j - 1) * lda
}

pub(crate) type D = V<f64>;
pub(crate) type I = V<usize>;
/// LAPACK `INTEGER` workspace: the MRRR routines store `-1` sentinels
/// (`dlarrb.f:307`, `dlarrj.f:242`), so these views are signed.
pub(crate) type J = V<i32>;
