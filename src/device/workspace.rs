//! Device buffers, one-copy staging, sub-range reads and traffic counters.

use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering::Relaxed};
use std::time::{Duration, Instant};

use cubecl::prelude::*;
use cubecl::server::Handle;

/// Upload a host slice with ONE host copy (`create`, not `create_from_slice`,
/// which copies twice in cubecl 0.10.0).
pub fn upload<R: Runtime, T: bytemuck::Pod + Send + Sync>(client: &ComputeClient<R>, data: &[T]) -> Handle {
    client.create(cubecl::bytes::Bytes::from_elems(data.to_vec()))
}

/// Upload an owned vector without any host copy.
pub fn upload_vec<R: Runtime, T: bytemuck::Pod + Send + Sync>(client: &ComputeClient<R>, data: Vec<T>) -> Handle {
    client.create(cubecl::bytes::Bytes::from_elems(data))
}

/// Upload/read/launch counters for the traffic gates.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct TransferCounters {
    pub uploads: usize,
    pub upload_bytes: usize,
    pub reads: usize,
    pub read_bytes: usize,
    pub launches: usize,
    /// Peak device `bytes_in_use` seen at the phase sample points (0 unless
    /// [`set_memory_sampling`] is on).
    pub peak_device_bytes: usize,
}

/// Interior-mutable counters owned by a workspace.
#[derive(Debug, Default)]
pub struct Counters {
    uploads: AtomicUsize,
    upload_bytes: AtomicUsize,
    reads: AtomicUsize,
    read_bytes: AtomicUsize,
    launches: AtomicUsize,
    peak_device_bytes: AtomicUsize,
    phase_ns: [AtomicU64; PHASES],
    phase_wait_ns: [AtomicU64; PHASES],
    wait_mark_ns: AtomicU64,
    span_ns: [AtomicU64; SPANS],
}

/// Host-side sections inside the phases, timed without syncing (device work
/// they enqueue is charged to whichever later read waits for it).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Span {
    /// `DSYTRD` panel update: `[V|W]` packing, upload and the K2 launch.
    TrdUpdate = 0,
    /// `DSTEDC` leaves: host `DSTEQR`, upload and scatter.
    StedcLeaf = 1,
    /// `DSTEDC` merge back-transform: permutations and staged U packing,
    /// uploads and the K3m launches.
    StedcMerge = 2,
    /// `DSTEDC` gathers and rotations (reorder, deflation copies).
    StedcGather = 3,
    /// `DORMTR` block: host `T`/`V` packing, uploads and the K3 launches.
    OrmBlock = 4,
}

const SPANS: usize = 5;

/// Driver phases timed by [`set_phase_profiling`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Phase {
    /// Scaling and staging the input as `M`.
    Stage = 0,
    /// Tridiagonalisation (`DSYTRD`/`ZHETRD`).
    Trd = 1,
    /// Tridiagonal eigensolver (`DSTEDC`, or `DSTERF` without vectors).
    Stedc = 2,
    /// Back-transformation (`DORMTR`/`ZUNMTR`).
    Orm = 3,
    /// Final read of the eigenvectors.
    Readback = 4,
}

const PHASES: usize = 5;

/// Wall time per phase, each closed by a device sync, plus the time spent
/// blocked in sub-range reads (device wait + copy) during the whole call.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PhaseTimes {
    pub stage: Duration,
    pub trd: Duration,
    pub stedc: Duration,
    pub orm: Duration,
    pub readback: Duration,
    pub read_wait: Duration,
    /// Read wait inside each phase (`[stage, trd, stedc, orm, readback]`).
    pub phase_wait: [Duration; PHASES],
    /// Host time per [`Span`], indexed by `Span as usize`.
    pub spans: [Duration; SPANS],
}

impl PhaseTimes {
    pub fn total(&self) -> Duration {
        self.stage + self.trd + self.stedc + self.orm + self.readback
    }
}

static MEMORY_SAMPLING: AtomicBool = AtomicBool::new(false);

static PHASE_PROFILING: AtomicBool = AtomicBool::new(false);
static READ_WAIT_NS: AtomicU64 = AtomicU64::new(0);

/// Close each driver phase with a device sync and record its wall time
/// (profiling only: the extra syncs change nothing numerically but add
/// latency). Read with `DeviceWorkspace::phase_times`.
///
/// The switch and the read-wait accumulator are process-global, so the
/// `read_wait` / `phase_wait` figures are only meaningful while one workspace
/// at a time is solving. The per-phase and per-span totals live in the
/// workspace's own counters.
pub fn set_phase_profiling(on: bool) {
    PHASE_PROFILING.store(on, Relaxed);
    crate::stedc::DLAED3_TIMING.store(on, Relaxed);
}

/// Host time in `DLAED3`'s secular roots, `W` products and eigenvector
/// columns since the last call (collected while phase profiling is on).
pub fn take_dlaed3_times() -> [Duration; 3] {
    core::array::from_fn(|i| Duration::from_nanos(crate::stedc::DLAED3_NS[i].swap(0, Relaxed)))
}

/// Whether [`set_phase_profiling`] is on.
pub fn phase_profiling() -> bool {
    PHASE_PROFILING.load(Relaxed)
}

/// Sample `client.memory_usage()` at each phase's peak point (a blocking
/// server round trip; diagnostics and the memory gate only).
#[doc(hidden)]
pub fn set_memory_sampling(on: bool) {
    MEMORY_SAMPLING.store(on, Relaxed);
}

impl Counters {
    pub fn upload(&self, bytes: usize) {
        self.uploads.fetch_add(1, Relaxed);
        self.upload_bytes.fetch_add(bytes, Relaxed);
    }
    pub fn read(&self, bytes: usize) {
        self.reads.fetch_add(1, Relaxed);
        self.read_bytes.fetch_add(bytes, Relaxed);
    }
    pub fn launch(&self) {
        self.launches.fetch_add(1, Relaxed);
    }
    pub fn snapshot(&self) -> TransferCounters {
        TransferCounters {
            uploads: self.uploads.load(Relaxed),
            upload_bytes: self.upload_bytes.load(Relaxed),
            reads: self.reads.load(Relaxed),
            read_bytes: self.read_bytes.load(Relaxed),
            launches: self.launches.load(Relaxed),
            peak_device_bytes: self.peak_device_bytes.load(Relaxed),
        }
    }
    /// Record the live device bytes if sampling is on. Call while a phase's
    /// transients are still alive.
    pub fn sample<R: Runtime>(&self, client: &ComputeClient<R>) {
        if MEMORY_SAMPLING.load(Relaxed)
            && let Ok(u) = client.memory_usage()
        {
            self.peak_device_bytes.fetch_max(u.bytes_in_use as usize, Relaxed);
        }
    }
    pub fn reset(&self) {
        for c in
            [&self.uploads, &self.upload_bytes, &self.reads, &self.read_bytes, &self.launches, &self.peak_device_bytes]
        {
            c.store(0, Relaxed);
        }
        for p in self.phase_ns.iter().chain(&self.phase_wait_ns).chain(&self.span_ns) {
            p.store(0, Relaxed);
        }
        READ_WAIT_NS.store(0, Relaxed);
        self.wait_mark_ns.store(0, Relaxed);
    }

    /// End `phase` (started at `*t`) when phase profiling is on: sync the
    /// device, add the elapsed time and restart `*t`.
    pub fn mark<R: Runtime>(&self, client: &ComputeClient<R>, phase: Phase, t: &mut Instant) {
        if !phase_profiling() {
            return;
        }
        let _ = cubecl::future::block_on(client.sync());
        self.phase_ns[phase as usize].fetch_add(t.elapsed().as_nanos() as u64, Relaxed);
        let wait = READ_WAIT_NS.load(Relaxed);
        let prev = self.wait_mark_ns.swap(wait, Relaxed);
        self.phase_wait_ns[phase as usize].fetch_add(wait.saturating_sub(prev), Relaxed);
        *t = Instant::now();
    }

    /// Start a host span when phase profiling is on.
    pub fn span_start(&self) -> Option<Instant> {
        phase_profiling().then(Instant::now)
    }

    /// End a span started by [`Self::span_start`].
    pub fn span_end(&self, span: Span, t: Option<Instant>) {
        if let Some(t) = t {
            self.span_ns[span as usize].fetch_add(t.elapsed().as_nanos() as u64, Relaxed);
        }
    }

    /// Accumulated phase times (see [`set_phase_profiling`]).
    pub fn phase_times(&self) -> PhaseTimes {
        let d = |p: Phase| Duration::from_nanos(self.phase_ns[p as usize].load(Relaxed));
        PhaseTimes {
            stage: d(Phase::Stage),
            trd: d(Phase::Trd),
            stedc: d(Phase::Stedc),
            orm: d(Phase::Orm),
            readback: d(Phase::Readback),
            read_wait: Duration::from_nanos(READ_WAIT_NS.load(Relaxed)),
            phase_wait: core::array::from_fn(|i| Duration::from_nanos(self.phase_wait_ns[i].load(Relaxed))),
            spans: core::array::from_fn(|i| Duration::from_nanos(self.span_ns[i].load(Relaxed))),
        }
    }
}

/// Read `len` elements of `T` starting at element `start` of `handle` through
/// an offset handle (a sub-range read; the whole buffer is never read to get
/// part of it). `buf_elems` is the buffer's logical length in elements, as
/// created: offsets apply to the logical byte length, while
/// `Handle::size()` may report a padded page (ROCm/HIP pads to its
/// alignment), so it cannot be used to derive the end trim.
pub fn read_range<R: Runtime, T: bytemuck::Pod>(
    client: &ComputeClient<R>,
    handle: &Handle,
    buf_elems: usize,
    start: usize,
    len: usize,
    out: &mut [T],
) {
    let esz = core::mem::size_of::<T>();
    assert!(start + len <= buf_elems);
    let total = buf_elems * esz;
    let begin = (start * esz) as u64;
    let end_trim = (total - (start + len) * esz) as u64;
    let sub = handle.clone().offset_start(begin).offset_end(end_trim);
    let t = phase_profiling().then(Instant::now);
    let bytes = client.read_one_unchecked(sub);
    if let Some(t) = t {
        READ_WAIT_NS.fetch_add(t.elapsed().as_nanos() as u64, Relaxed);
    }
    let src: &[T] = bytemuck::cast_slice(&bytes[..len * esz]);
    out[..len].copy_from_slice(src);
}
