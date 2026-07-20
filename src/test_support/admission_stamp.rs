//! Link-retained, no-exec resource-admission metadata.
//!
//! Nextest's wrapper process needs to acquire ktstr's exact CPU/LLC claim
//! before it starts a heavyweight test executable.  These records project the
//! admission-relevant portion of every [`KtstrTestEntry`] and canned topology
//! preset into a strict, versioned ELF wire format.  They intentionally live
//! beside, rather than inside, the scheduler-manifest records: changing the
//! admission contract must never silently change that established layout.

use std::collections::{BTreeMap, BTreeSet};
use std::mem::{offset_of, size_of};
use std::path::Path;

use linkme::distributed_slice;

use super::scheduler_manifest_stamp::ElfStampReader;
use super::{KtstrTestEntry, Topology};

const ADMISSION_MAGIC: [u8; 8] = *b"KTSTRAD2";
const ADMISSION_VERSION: u16 = 2;
const ADMISSION_KIND_SENTINEL: u16 = 0;
const ADMISSION_KIND_TEST: u16 = 1;
const ADMISSION_KIND_PRESET: u16 = 2;
const ADMISSION_KIND_TEST_KEY: u16 = 3;
const ADMISSION_KIND_PRESET_KEY: u16 = 4;

const TEST_SECTION: &str = "linkme_KTSTR_ADMISSION_TESTS_V2";
const PRESET_SECTION: &str = "linkme_KTSTR_ADMISSION_PRESETS_V2";
const TEST_KEY_SECTION: &str = "linkme_KTSTR_ADMISSION_TEST_KEYS_V2";
const PRESET_KEY_SECTION: &str = "linkme_KTSTR_ADMISSION_PRESET_KEYS_V2";
const LEGACY_TEST_SECTION: &str = "linkme_KTSTR_TESTS";

// KVM's topology surface is capped below 256 vCPUs.  Fixed-width arrays make
// the stamp self-contained: explicit NUMA and sparse-LLC declarations do not
// leak the Rust layout of `NumaNode` or depend on nested pointer relocations.
const MAX_TOPOLOGY_COMPONENTS: usize = 255;

#[repr(C)]
#[derive(Clone, Copy)]
struct AdmissionHeaderV2 {
    magic: [u8; 8],
    version: u16,
    kind: u16,
    record_size: u32,
}

impl AdmissionHeaderV2 {
    const fn new(kind: u16, record_size: usize) -> Self {
        Self {
            magic: ADMISSION_MAGIC,
            version: ADMISSION_VERSION,
            kind,
            record_size: record_size as u32,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
struct AdmissionStrV2 {
    ptr: *const u8,
    len: u64,
}

// Every emitted pointer refers to an immutable static string.
unsafe impl Sync for AdmissionStrV2 {}

impl AdmissionStrV2 {
    const fn new(value: &'static str) -> Self {
        Self {
            ptr: value.as_ptr(),
            len: value.len() as u64,
        }
    }

    const fn empty() -> Self {
        Self {
            ptr: std::ptr::null(),
            len: 0,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
struct AdmissionOptionalU32V2 {
    value: u32,
    present: u8,
    reserved: [u8; 3],
}

impl AdmissionOptionalU32V2 {
    const fn new(value: Option<u32>) -> Self {
        match value {
            Some(value) => Self {
                value,
                present: 1,
                reserved: [0; 3],
            },
            None => Self {
                value: 0,
                present: 0,
                reserved: [0; 3],
            },
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
struct AdmissionTopologyStampV2 {
    numa_nodes: u32,
    llcs: u32,
    cores_per_llc: u32,
    threads_per_core: u32,
    explicit_nodes: u8,
    llc_cores_present: u8,
    node_llcs_len: u16,
    llc_cores_len: u16,
    reserved: [u8; 2],
    node_llcs: [u32; MAX_TOPOLOGY_COMPONENTS],
    llc_cores: [u32; MAX_TOPOLOGY_COMPONENTS],
}

impl AdmissionTopologyStampV2 {
    const fn new(topology: Topology) -> Self {
        let mut node_llcs = [0; MAX_TOPOLOGY_COMPONENTS];
        let (explicit_nodes, node_llcs_len) = match topology.nodes {
            Some(nodes) => {
                assert!(
                    nodes.len() <= MAX_TOPOLOGY_COMPONENTS,
                    "ktstr admission stamp has too many explicit NUMA nodes"
                );
                let mut index = 0;
                while index < nodes.len() {
                    node_llcs[index] = nodes[index].llcs;
                    index += 1;
                }
                (1, nodes.len() as u16)
            }
            None => (0, 0),
        };

        let mut llc_cores = [0; MAX_TOPOLOGY_COMPONENTS];
        let (llc_cores_present, llc_cores_len) = match topology.llc_cores {
            Some(cores) => {
                assert!(
                    cores.len() <= MAX_TOPOLOGY_COMPONENTS,
                    "ktstr admission stamp has too many LLC core-count entries"
                );
                let mut index = 0;
                while index < cores.len() {
                    llc_cores[index] = cores[index];
                    index += 1;
                }
                (1, cores.len() as u16)
            }
            None => (0, 0),
        };

        Self {
            numa_nodes: topology.numa_nodes,
            llcs: topology.llcs,
            cores_per_llc: topology.cores_per_llc,
            threads_per_core: topology.threads_per_core,
            explicit_nodes,
            llc_cores_present,
            node_llcs_len,
            llc_cores_len,
            reserved: [0; 2],
            node_llcs,
            llc_cores,
        }
    }

    const fn sentinel() -> Self {
        Self {
            numa_nodes: 0,
            llcs: 0,
            cores_per_llc: 0,
            threads_per_core: 0,
            explicit_nodes: 0,
            llc_cores_present: 0,
            node_llcs_len: 0,
            llc_cores_len: 0,
            reserved: [0; 2],
            node_llcs: [0; MAX_TOPOLOGY_COMPONENTS],
            llc_cores: [0; MAX_TOPOLOGY_COMPONENTS],
        }
    }
}

/// Version-2 admission record emitted for every `KtstrTestEntry`.
///
/// Public only because proc-macro expansions in downstream crates construct
/// it.  It is not a source-level extension point.
#[doc(hidden)]
#[repr(C)]
#[derive(Clone, Copy)]
pub struct AdmissionTestStampV2 {
    header: AdmissionHeaderV2,
    name: AdmissionStrV2,
    topology: AdmissionTopologyStampV2,
    cpu_budget: AdmissionOptionalU32V2,
    host_only: u8,
    performance_mode: u8,
    no_perf_mode: u8,
    expect_auto_repro: u8,
    reserved: [u8; 4],
    legacy_entry: *const KtstrTestEntry,
}

unsafe impl Sync for AdmissionTestStampV2 {}

impl AdmissionTestStampV2 {
    /// Project one const-evaluated test entry into the v2 admission wire shape.
    #[doc(hidden)]
    pub const fn new(entry: &'static KtstrTestEntry) -> Self {
        Self {
            header: AdmissionHeaderV2::new(ADMISSION_KIND_TEST, size_of::<AdmissionTestStampV2>()),
            name: AdmissionStrV2::new(entry.name),
            topology: AdmissionTopologyStampV2::new(entry.topology),
            cpu_budget: AdmissionOptionalU32V2::new(entry.cpu_budget),
            host_only: entry.host_only as u8,
            performance_mode: entry.performance_mode as u8,
            no_perf_mode: entry.no_perf_mode as u8,
            expect_auto_repro: entry.expect_auto_repro as u8,
            reserved: [0; 4],
            legacy_entry: entry,
        }
    }

    const fn sentinel() -> Self {
        Self {
            header: AdmissionHeaderV2::new(
                ADMISSION_KIND_SENTINEL,
                size_of::<AdmissionTestStampV2>(),
            ),
            name: AdmissionStrV2::empty(),
            topology: AdmissionTopologyStampV2::sentinel(),
            cpu_budget: AdmissionOptionalU32V2::new(None),
            host_only: 0,
            performance_mode: 0,
            no_perf_mode: 0,
            expect_auto_repro: 0,
            reserved: [0; 4],
            legacy_entry: std::ptr::null(),
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
struct AdmissionPresetStampV2 {
    header: AdmissionHeaderV2,
    name: AdmissionStrV2,
    topology: AdmissionTopologyStampV2,
    forced_cpu_budget: AdmissionOptionalU32V2,
    sentinel_expected_count: u32,
    reserved: [u8; 4],
}

unsafe impl Sync for AdmissionPresetStampV2 {}

impl AdmissionPresetStampV2 {
    const fn new(name: &'static str, topology: Topology, forced_cpu_budget: Option<u32>) -> Self {
        Self {
            header: AdmissionHeaderV2::new(
                ADMISSION_KIND_PRESET,
                size_of::<AdmissionPresetStampV2>(),
            ),
            name: AdmissionStrV2::new(name),
            topology: AdmissionTopologyStampV2::new(topology),
            forced_cpu_budget: AdmissionOptionalU32V2::new(forced_cpu_budget),
            sentinel_expected_count: 0,
            reserved: [0; 4],
        }
    }

    const fn sentinel(expected_count: u32) -> Self {
        Self {
            header: AdmissionHeaderV2::new(
                ADMISSION_KIND_SENTINEL,
                size_of::<AdmissionPresetStampV2>(),
            ),
            name: AdmissionStrV2::empty(),
            topology: AdmissionTopologyStampV2::sentinel(),
            forced_cpu_budget: AdmissionOptionalU32V2::new(None),
            sentinel_expected_count: expected_count,
            reserved: [0; 4],
        }
    }
}

/// Stable, deliberately non-cryptographic hash used by the compact admission
/// lookup sections. A full string comparison is authoritative, so collisions
/// merely add one candidate parse and can never alias two test cells.
const fn admission_name_hash(value: &str) -> u64 {
    let bytes = value.as_bytes();
    let mut hash = 0xcbf29ce484222325u64;
    let mut index = 0usize;
    while index < bytes.len() {
        hash ^= bytes[index] as u64;
        hash = hash.wrapping_mul(0x100000001b3);
        index += 1;
    }
    hash
}

/// Compact lookup key for one test admission record.
///
/// Public only because proc-macro expansions in downstream crates emit it.
#[doc(hidden)]
#[repr(C)]
#[derive(Clone, Copy)]
pub struct AdmissionTestKeyV2 {
    header: AdmissionHeaderV2,
    name_hash: u64,
    name_len: u64,
    record: *const AdmissionTestStampV2,
}

unsafe impl Sync for AdmissionTestKeyV2 {}

impl AdmissionTestKeyV2 {
    #[doc(hidden)]
    pub const fn new(
        entry: &'static KtstrTestEntry,
        record: &'static AdmissionTestStampV2,
    ) -> Self {
        Self {
            header: AdmissionHeaderV2::new(
                ADMISSION_KIND_TEST_KEY,
                size_of::<AdmissionTestKeyV2>(),
            ),
            name_hash: admission_name_hash(entry.name),
            name_len: entry.name.len() as u64,
            record,
        }
    }

    const fn sentinel() -> Self {
        Self {
            header: AdmissionHeaderV2::new(
                ADMISSION_KIND_SENTINEL,
                size_of::<AdmissionTestKeyV2>(),
            ),
            name_hash: 0,
            name_len: 0,
            record: std::ptr::null(),
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
struct AdmissionPresetKeyV2 {
    header: AdmissionHeaderV2,
    name_hash: u64,
    name_len: u64,
    record: *const AdmissionPresetStampV2,
}

unsafe impl Sync for AdmissionPresetKeyV2 {}

impl AdmissionPresetKeyV2 {
    const fn new(name: &'static str, record: &'static AdmissionPresetStampV2) -> Self {
        Self {
            header: AdmissionHeaderV2::new(
                ADMISSION_KIND_PRESET_KEY,
                size_of::<AdmissionPresetKeyV2>(),
            ),
            name_hash: admission_name_hash(name),
            name_len: name.len() as u64,
            record,
        }
    }

    const fn sentinel() -> Self {
        Self {
            header: AdmissionHeaderV2::new(
                ADMISSION_KIND_SENTINEL,
                size_of::<AdmissionPresetKeyV2>(),
            ),
            name_hash: 0,
            name_len: 0,
            record: std::ptr::null(),
        }
    }
}

/// Link-retained v2 test-admission records.
#[doc(hidden)]
#[distributed_slice]
pub static KTSTR_ADMISSION_TESTS_V2: [AdmissionTestStampV2];

#[distributed_slice]
static KTSTR_ADMISSION_PRESETS_V2: [AdmissionPresetStampV2];

/// Compact link-retained lookup keys for v2 test-admission records.
#[doc(hidden)]
#[distributed_slice]
pub static KTSTR_ADMISSION_TEST_KEYS_V2: [AdmissionTestKeyV2];

#[distributed_slice]
static KTSTR_ADMISSION_PRESET_KEYS_V2: [AdmissionPresetKeyV2];

#[distributed_slice(KTSTR_ADMISSION_TESTS_V2)]
static KTSTR_ADMISSION_TESTS_V2_SENTINEL: AdmissionTestStampV2 = AdmissionTestStampV2::sentinel();

#[distributed_slice(KTSTR_ADMISSION_TEST_KEYS_V2)]
static KTSTR_ADMISSION_TEST_KEYS_V2_SENTINEL: AdmissionTestKeyV2 = AdmissionTestKeyV2::sentinel();

#[cfg(target_arch = "aarch64")]
const CANNED_PRESET_COUNT: u32 = 14;
#[cfg(not(target_arch = "aarch64"))]
const CANNED_PRESET_COUNT: u32 = 26;

#[distributed_slice(KTSTR_ADMISSION_PRESETS_V2)]
static KTSTR_ADMISSION_PRESETS_V2_SENTINEL: AdmissionPresetStampV2 =
    AdmissionPresetStampV2::sentinel(CANNED_PRESET_COUNT);

#[distributed_slice(KTSTR_ADMISSION_PRESET_KEYS_V2)]
static KTSTR_ADMISSION_PRESET_KEYS_V2_SENTINEL: AdmissionPresetKeyV2 =
    AdmissionPresetKeyV2::sentinel();

macro_rules! admission_preset {
    (
        $(#[$meta:meta])*
        $static_name:ident, $name:literal,
        $numa_nodes:expr, $llcs:expr, $cores:expr, $threads:expr,
        $llc_cores:expr, $forced_cpu_budget:expr
    ) => {
        $(#[$meta])*
        #[distributed_slice(KTSTR_ADMISSION_PRESETS_V2)]
        static $static_name: AdmissionPresetStampV2 = AdmissionPresetStampV2::new(
            $name,
            Topology {
                numa_nodes: $numa_nodes,
                llcs: $llcs,
                cores_per_llc: $cores,
                threads_per_core: $threads,
                nodes: None,
                distances: None,
                llc_cores: $llc_cores,
            },
            $forced_cpu_budget,
        );

        $(#[$meta])*
        #[allow(non_snake_case)]
        mod $static_name {
            use super::*;

            #[distributed_slice(KTSTR_ADMISSION_PRESET_KEYS_V2)]
            static KEY: AdmissionPresetKeyV2 =
                AdmissionPresetKeyV2::new($name, &super::$static_name);
        }
    };
}

admission_preset!(
    PRESET_4C_1L_NOSMT,
    "4cpu-1llc-nosmt",
    1,
    1,
    4,
    1,
    None,
    None
);
admission_preset!(
    PRESET_4C_2L_NOSMT,
    "4cpu-2llc-nosmt",
    1,
    2,
    2,
    1,
    None,
    None
);
admission_preset!(
    PRESET_9C_3L_NOSMT,
    "9cpu-3llc-nosmt",
    1,
    3,
    3,
    1,
    None,
    None
);
admission_preset!(
    PRESET_15C_5L_NOSMT,
    "15cpu-5llc-nosmt",
    1,
    5,
    3,
    1,
    None,
    None
);
admission_preset!(
    PRESET_14C_7L_NOSMT,
    "14cpu-7llc-nosmt",
    1,
    7,
    2,
    1,
    None,
    None
);
admission_preset!(
    #[cfg(not(target_arch = "aarch64"))]
    PRESET_8C_2L_SMT,
    "8cpu-2llc-smt",
    1,
    2,
    2,
    2,
    None,
    None
);
admission_preset!(
    #[cfg(not(target_arch = "aarch64"))]
    PRESET_12C_3L_SMT,
    "12cpu-3llc-smt",
    1,
    3,
    2,
    2,
    None,
    None
);
admission_preset!(
    #[cfg(not(target_arch = "aarch64"))]
    PRESET_32C_4L_SMT,
    "32cpu-4llc-smt",
    1,
    4,
    4,
    2,
    None,
    None
);
admission_preset!(
    #[cfg(not(target_arch = "aarch64"))]
    PRESET_64C_8L_SMT,
    "64cpu-8llc-smt",
    1,
    8,
    4,
    2,
    None,
    None
);
admission_preset!(
    #[cfg(not(target_arch = "aarch64"))]
    PRESET_128C_4L_SMT,
    "128cpu-4llc-smt",
    1,
    4,
    16,
    2,
    None,
    None
);
admission_preset!(
    #[cfg(not(target_arch = "aarch64"))]
    PRESET_128C_8L_SMT,
    "128cpu-8llc-smt",
    1,
    8,
    8,
    2,
    None,
    None
);
admission_preset!(
    #[cfg(not(target_arch = "aarch64"))]
    PRESET_240C_15L_SMT,
    "240cpu-15llc-smt",
    1,
    15,
    8,
    2,
    None,
    None
);
admission_preset!(
    #[cfg(not(target_arch = "aarch64"))]
    PRESET_252C_14L_SMT,
    "252cpu-14llc-smt",
    1,
    14,
    9,
    2,
    None,
    None
);
admission_preset!(
    PRESET_32C_4L_NOSMT,
    "32cpu-4llc-nosmt",
    1,
    4,
    8,
    1,
    None,
    None
);
admission_preset!(
    PRESET_64C_8L_NOSMT,
    "64cpu-8llc-nosmt",
    1,
    8,
    8,
    1,
    None,
    None
);
admission_preset!(
    PRESET_128C_4L_NOSMT,
    "128cpu-4llc-nosmt",
    1,
    4,
    32,
    1,
    None,
    None
);
admission_preset!(
    PRESET_128C_8L_NOSMT,
    "128cpu-8llc-nosmt",
    1,
    8,
    16,
    1,
    None,
    None
);
admission_preset!(
    PRESET_240C_15L_NOSMT,
    "240cpu-15llc-nosmt",
    1,
    15,
    16,
    1,
    None,
    None
);
admission_preset!(
    PRESET_252C_14L_NOSMT,
    "252cpu-14llc-nosmt",
    1,
    14,
    18,
    1,
    None,
    None
);
admission_preset!(
    #[cfg(not(target_arch = "aarch64"))]
    PRESET_2N_32C_2L_SMT,
    "2numa-32cpu-2llc-smt",
    2,
    2,
    8,
    2,
    None,
    None
);
admission_preset!(
    PRESET_2N_16C_4L_NOSMT,
    "2numa-16cpu-4llc-nosmt",
    2,
    4,
    4,
    1,
    None,
    None
);
admission_preset!(
    #[cfg(not(target_arch = "aarch64"))]
    PRESET_2N_128C_8L_SMT,
    "2numa-128cpu-8llc-smt",
    2,
    8,
    8,
    2,
    None,
    None
);
admission_preset!(
    PRESET_2N_128C_8L_NOSMT,
    "2numa-128cpu-8llc-nosmt",
    2,
    8,
    16,
    1,
    None,
    None
);
admission_preset!(
    PRESET_4N_32C_8L_NOSMT,
    "4numa-32cpu-8llc-nosmt",
    4,
    8,
    4,
    1,
    None,
    None
);
admission_preset!(
    #[cfg(not(target_arch = "aarch64"))]
    PRESET_4N_192C_12L_SMT,
    "4numa-192cpu-12llc-smt",
    4,
    12,
    8,
    2,
    None,
    None
);

#[cfg(not(target_arch = "aarch64"))]
static UNEVEN_11LLC_CORES: [u32; 11] = [9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 6];
admission_preset!(
    #[cfg(not(target_arch = "aarch64"))]
    PRESET_192C_11L_UNEVEN_SMT,
    "192cpu-11llc-smt",
    1,
    11,
    9,
    2,
    Some(&UNEVEN_11LLC_CORES),
    Some(96)
);

/// Which generated ktstr test family an admission descriptor belongs to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AdmissionCellKind {
    Host,
    Ktstr,
    Gauntlet,
    Verifier,
}

/// Admission policy declared by a test or imposed by a generated cell.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AdmissionMode {
    Default,
    Performance,
    NoPerf,
}

/// Owned topology projection used by the pre-exec admission wrapper.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct AdmissionTopologyDescriptor {
    pub numa_nodes: u32,
    pub llcs: u32,
    pub cores_per_llc: u32,
    pub threads_per_core: u32,
    /// Present only when the declaration used explicit `Topology::nodes`.
    pub node_llcs: Option<Vec<u32>>,
    /// Present only for a non-uniform LLC topology.
    pub llc_cores: Option<Vec<u32>>,
}

impl AdmissionTopologyDescriptor {
    /// Total guest vCPUs represented by this topology.
    pub fn total_cpus(&self) -> u32 {
        let cores = self
            .llc_cores
            .as_ref()
            .map(|cores| cores.iter().copied().sum())
            .unwrap_or_else(|| self.llcs * self.cores_per_llc);
        cores * self.threads_per_core
    }

    fn checked_total_cpus(&self) -> Option<u32> {
        let cores = match &self.llc_cores {
            Some(cores) => cores
                .iter()
                .try_fold(0u32, |sum, cores| sum.checked_add(*cores))?,
            None => self.llcs.checked_mul(self.cores_per_llc)?,
        };
        cores.checked_mul(self.threads_per_core)
    }
}

/// Complete pre-exec admission description for one exact generated test name.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct AdmissionCellDescriptor {
    pub exact_name: String,
    pub kind: AdmissionCellKind,
    pub entry_name: Option<String>,
    pub preset_name: Option<String>,
    pub scheduler_name: Option<String>,
    pub kernel: Option<String>,
    pub topology: AdmissionTopologyDescriptor,
    pub cpu_budget: Option<u32>,
    pub mode: AdmissionMode,
    pub host_only: bool,
    pub performance_mode: bool,
    pub no_perf_mode: bool,
    pub expect_auto_repro: bool,
}

#[derive(Debug, Clone)]
struct ParsedAdmissionTest {
    record_base: u64,
    name: String,
    topology: AdmissionTopologyDescriptor,
    cpu_budget: Option<u32>,
    host_only: bool,
    performance_mode: bool,
    no_perf_mode: bool,
    expect_auto_repro: bool,
    legacy_entry: u64,
}

#[derive(Debug, Clone)]
struct ParsedAdmissionPreset {
    record_base: u64,
    name: String,
    topology: AdmissionTopologyDescriptor,
    forced_cpu_budget: Option<u32>,
}

struct AdmissionIndex {
    tests: Vec<ParsedAdmissionTest>,
    presets: Vec<ParsedAdmissionPreset>,
}

#[derive(Clone, Copy)]
struct AdmissionKeySlot {
    base: u64,
    index: usize,
    kind: u16,
}

struct AdmissionKeyTable {
    section: &'static str,
    expected_kind: u16,
    name_hash_offset: usize,
    name_len_offset: usize,
    record_offset: usize,
    slots: Vec<AdmissionKeySlot>,
}

fn header(
    reader: &ElfStampReader<'_>,
    base: u64,
    record_size: usize,
    section: &str,
    index: usize,
) -> Result<u16, String> {
    let what = format!("{section} record {index}");
    let magic = reader.bytes_at_va(
        base + offset_of!(AdmissionHeaderV2, magic) as u64,
        ADMISSION_MAGIC.len(),
        &what,
    )?;
    if magic != ADMISSION_MAGIC {
        return Err(format!(
            "corrupt admission stamp in {}: {what} has invalid magic {:?}",
            reader.source.display(),
            magic,
        ));
    }
    let version = reader.u16(base, offset_of!(AdmissionHeaderV2, version), &what)?;
    if version != ADMISSION_VERSION {
        return Err(format!(
            "unsupported admission stamp version {version} in {} ({what}); expected version \
             {ADMISSION_VERSION}",
            reader.source.display(),
        ));
    }
    let encoded_size = reader.u32(base, offset_of!(AdmissionHeaderV2, record_size), &what)?;
    if encoded_size as usize != record_size {
        return Err(format!(
            "corrupt admission stamp in {}: {what} declares record size {encoded_size}, \
             expected {record_size}",
            reader.source.display(),
        ));
    }
    reader.u16(base, offset_of!(AdmissionHeaderV2, kind), &what)
}

fn string(reader: &ElfStampReader<'_>, base: u64, what: &str) -> Result<String, String> {
    let len = reader.raw_u64(base + offset_of!(AdmissionStrV2, len) as u64, what)?;
    let len = usize::try_from(len).map_err(|_| {
        format!(
            "{what} in admission ELF {} has an unrepresentable string length",
            reader.source.display(),
        )
    })?;
    if len == 0 {
        return Ok(String::new());
    }
    let pointer = reader.pointer(base, offset_of!(AdmissionStrV2, ptr), what)?;
    if pointer == 0 {
        return Err(format!(
            "{what} in admission ELF {} has a null pointer for a non-empty string",
            reader.source.display(),
        ));
    }
    let bytes = reader.bytes_at_va(pointer, len, what)?;
    std::str::from_utf8(bytes)
        .map(str::to_owned)
        .map_err(|error| {
            format!(
                "{what} in admission ELF {} is not UTF-8: {error}",
                reader.source.display(),
            )
        })
}

fn optional_u32(reader: &ElfStampReader<'_>, base: u64, what: &str) -> Result<Option<u32>, String> {
    match reader.u8(base, offset_of!(AdmissionOptionalU32V2, present), what)? {
        0 => Ok(None),
        1 => reader
            .u32(base, offset_of!(AdmissionOptionalU32V2, value), what)
            .map(Some),
        other => Err(format!(
            "{what} in admission ELF {} has invalid option tag {other}",
            reader.source.display(),
        )),
    }
}

fn boolean(
    reader: &ElfStampReader<'_>,
    base: u64,
    offset: usize,
    what: &str,
) -> Result<bool, String> {
    match reader.u8(base, offset, what)? {
        0 => Ok(false),
        1 => Ok(true),
        other => Err(format!(
            "{what} in admission ELF {} has invalid boolean {other}",
            reader.source.display(),
        )),
    }
}

fn topology(
    reader: &ElfStampReader<'_>,
    base: u64,
    what: &str,
) -> Result<AdmissionTopologyDescriptor, String> {
    let numa_nodes = reader.u32(base, offset_of!(AdmissionTopologyStampV2, numa_nodes), what)?;
    let llcs = reader.u32(base, offset_of!(AdmissionTopologyStampV2, llcs), what)?;
    let cores_per_llc = reader.u32(
        base,
        offset_of!(AdmissionTopologyStampV2, cores_per_llc),
        what,
    )?;
    let threads_per_core = reader.u32(
        base,
        offset_of!(AdmissionTopologyStampV2, threads_per_core),
        what,
    )?;
    let explicit_nodes = boolean(
        reader,
        base,
        offset_of!(AdmissionTopologyStampV2, explicit_nodes),
        &format!("{what}.explicit_nodes"),
    )?;
    let llc_cores_present = boolean(
        reader,
        base,
        offset_of!(AdmissionTopologyStampV2, llc_cores_present),
        &format!("{what}.llc_cores_present"),
    )?;
    let node_llcs_len = reader.u16(
        base,
        offset_of!(AdmissionTopologyStampV2, node_llcs_len),
        &format!("{what}.node_llcs_len"),
    )? as usize;
    let llc_cores_len = reader.u16(
        base,
        offset_of!(AdmissionTopologyStampV2, llc_cores_len),
        &format!("{what}.llc_cores_len"),
    )? as usize;
    if node_llcs_len > MAX_TOPOLOGY_COMPONENTS || llc_cores_len > MAX_TOPOLOGY_COMPONENTS {
        return Err(format!(
            "{what} in admission ELF {} exceeds the topology component limit",
            reader.source.display(),
        ));
    }
    if explicit_nodes != (node_llcs_len != 0) {
        return Err(format!(
            "{what} in admission ELF {} has inconsistent explicit-node presence/length",
            reader.source.display(),
        ));
    }
    if llc_cores_present != (llc_cores_len != 0) {
        return Err(format!(
            "{what} in admission ELF {} has inconsistent LLC-core presence/length",
            reader.source.display(),
        ));
    }
    let read_array = |offset: usize, len: usize, label: &str| {
        (0..len)
            .map(|index| {
                reader.u32(
                    base,
                    offset + index * size_of::<u32>(),
                    &format!("{what}.{label}[{index}]"),
                )
            })
            .collect::<Result<Vec<_>, _>>()
    };
    let node_llcs = explicit_nodes
        .then(|| {
            read_array(
                offset_of!(AdmissionTopologyStampV2, node_llcs),
                node_llcs_len,
                "node_llcs",
            )
        })
        .transpose()?;
    let llc_cores = llc_cores_present
        .then(|| {
            read_array(
                offset_of!(AdmissionTopologyStampV2, llc_cores),
                llc_cores_len,
                "llc_cores",
            )
        })
        .transpose()?;

    let descriptor = AdmissionTopologyDescriptor {
        numa_nodes,
        llcs,
        cores_per_llc,
        threads_per_core,
        node_llcs,
        llc_cores,
    };
    validate_topology(reader, what, &descriptor)?;
    Ok(descriptor)
}

fn validate_topology(
    reader: &ElfStampReader<'_>,
    what: &str,
    topology: &AdmissionTopologyDescriptor,
) -> Result<(), String> {
    if topology.numa_nodes == 0
        || topology.llcs == 0
        || topology.cores_per_llc == 0
        || topology.threads_per_core == 0
    {
        return Err(format!(
            "{what} in admission ELF {} has a zero topology dimension",
            reader.source.display(),
        ));
    }
    match &topology.node_llcs {
        Some(node_llcs) => {
            let distributed_llcs = node_llcs
                .iter()
                .try_fold(0u32, |sum, llcs| sum.checked_add(*llcs));
            if node_llcs.len() != topology.numa_nodes as usize
                || distributed_llcs != Some(topology.llcs)
            {
                return Err(format!(
                    "{what} in admission ELF {} has an invalid explicit NUMA LLC distribution",
                    reader.source.display(),
                ));
            }
        }
        None if !topology.llcs.is_multiple_of(topology.numa_nodes) => {
            return Err(format!(
                "{what} in admission ELF {} cannot distribute {} LLCs over {} NUMA nodes",
                reader.source.display(),
                topology.llcs,
                topology.numa_nodes,
            ));
        }
        None => {}
    }
    if let Some(llc_cores) = &topology.llc_cores {
        if llc_cores.len() != topology.llcs as usize
            || llc_cores
                .iter()
                .any(|cores| *cores == 0 || *cores > topology.cores_per_llc)
        {
            return Err(format!(
                "{what} in admission ELF {} has invalid non-uniform LLC core counts",
                reader.source.display(),
            ));
        }
    }
    topology.checked_total_cpus().ok_or_else(|| {
        format!(
            "{what} in admission ELF {} overflows its total CPU count",
            reader.source.display(),
        )
    })?;
    Ok(())
}

fn test_record(
    reader: &ElfStampReader<'_>,
    base: u64,
    index: usize,
) -> Result<ParsedAdmissionTest, String> {
    let what = format!("admission test record {index}");
    let performance_mode = boolean(
        reader,
        base,
        offset_of!(AdmissionTestStampV2, performance_mode),
        &format!("{what}.performance_mode"),
    )?;
    let no_perf_mode = boolean(
        reader,
        base,
        offset_of!(AdmissionTestStampV2, no_perf_mode),
        &format!("{what}.no_perf_mode"),
    )?;
    if performance_mode && no_perf_mode {
        return Err(format!(
            "{what} in admission ELF {} declares both performance and no-perf mode",
            reader.source.display(),
        ));
    }
    Ok(ParsedAdmissionTest {
        record_base: base,
        name: string(
            reader,
            base + offset_of!(AdmissionTestStampV2, name) as u64,
            &format!("{what}.name"),
        )?,
        topology: topology(
            reader,
            base + offset_of!(AdmissionTestStampV2, topology) as u64,
            &format!("{what}.topology"),
        )?,
        cpu_budget: optional_u32(
            reader,
            base + offset_of!(AdmissionTestStampV2, cpu_budget) as u64,
            &format!("{what}.cpu_budget"),
        )?,
        host_only: boolean(
            reader,
            base,
            offset_of!(AdmissionTestStampV2, host_only),
            &format!("{what}.host_only"),
        )?,
        performance_mode,
        no_perf_mode,
        expect_auto_repro: boolean(
            reader,
            base,
            offset_of!(AdmissionTestStampV2, expect_auto_repro),
            &format!("{what}.expect_auto_repro"),
        )?,
        legacy_entry: reader.pointer(
            base,
            offset_of!(AdmissionTestStampV2, legacy_entry),
            &format!("{what}.legacy_entry"),
        )?,
    })
}

fn preset_record(
    reader: &ElfStampReader<'_>,
    base: u64,
    index: usize,
) -> Result<ParsedAdmissionPreset, String> {
    let what = format!("admission preset record {index}");
    Ok(ParsedAdmissionPreset {
        record_base: base,
        name: string(
            reader,
            base + offset_of!(AdmissionPresetStampV2, name) as u64,
            &format!("{what}.name"),
        )?,
        topology: topology(
            reader,
            base + offset_of!(AdmissionPresetStampV2, topology) as u64,
            &format!("{what}.topology"),
        )?,
        forced_cpu_budget: optional_u32(
            reader,
            base + offset_of!(AdmissionPresetStampV2, forced_cpu_budget) as u64,
            &format!("{what}.forced_cpu_budget"),
        )?,
    })
}

fn section_records<T>(
    reader: &ElfStampReader<'_>,
    section: &str,
    record_size: usize,
    expected_kind: u16,
    mut parse: impl FnMut(&ElfStampReader<'_>, u64, usize) -> Result<T, String>,
) -> Result<Option<(Vec<T>, u64)>, String> {
    let Some((section_va, bytes)) = reader.section(section)? else {
        return Ok(None);
    };
    if bytes.is_empty() || !bytes.len().is_multiple_of(record_size) {
        return Err(format!(
            "corrupt admission section {section} in {}: byte length {} is not a non-zero \
             multiple of record size {record_size}",
            reader.source.display(),
            bytes.len(),
        ));
    }
    let mut sentinel = None;
    let mut records = Vec::new();
    for index in 0..bytes.len() / record_size {
        let base = section_va + (index * record_size) as u64;
        match header(reader, base, record_size, section, index)? {
            ADMISSION_KIND_SENTINEL if sentinel.replace(base).is_none() => {}
            ADMISSION_KIND_SENTINEL => {
                return Err(format!(
                    "corrupt admission section {section} in {}: multiple version sentinels",
                    reader.source.display(),
                ));
            }
            kind if kind == expected_kind => records.push(parse(reader, base, index)?),
            kind => {
                return Err(format!(
                    "corrupt admission section {section} in {}: record {index} has kind \
                     {kind}, expected sentinel or {expected_kind}",
                    reader.source.display(),
                ));
            }
        }
    }
    let sentinel = sentinel.ok_or_else(|| {
        format!(
            "corrupt admission section {section} in {}: missing version sentinel",
            reader.source.display(),
        )
    })?;
    Ok(Some((records, sentinel)))
}

fn key_table(
    reader: &ElfStampReader<'_>,
    section: &'static str,
    expected_kind: u16,
    record_size: usize,
    name_hash_offset: usize,
    name_len_offset: usize,
    record_offset: usize,
) -> Result<Option<AdmissionKeyTable>, String> {
    let Some((section_va, bytes)) = reader.section(section)? else {
        return Ok(None);
    };
    if bytes.is_empty() || !bytes.len().is_multiple_of(record_size) {
        return Err(format!(
            "corrupt admission key section {section} in {}: byte length {} is not a non-zero \
             multiple of record size {record_size}",
            reader.source.display(),
            bytes.len(),
        ));
    }
    let mut sentinel_count = 0usize;
    let mut slots = Vec::with_capacity(bytes.len() / record_size);
    for index in 0..bytes.len() / record_size {
        let base = section_va + (index * record_size) as u64;
        let kind = header(reader, base, record_size, section, index)?;
        match kind {
            ADMISSION_KIND_SENTINEL => sentinel_count += 1,
            kind if kind == expected_kind => {}
            kind => {
                return Err(format!(
                    "corrupt admission key section {section} in {}: record {index} has kind \
                     {kind}, expected sentinel or {expected_kind}",
                    reader.source.display(),
                ));
            }
        }
        slots.push(AdmissionKeySlot { base, index, kind });
    }
    if sentinel_count != 1 {
        return Err(format!(
            "corrupt admission key section {section} in {}: expected exactly one version \
             sentinel, found {sentinel_count}",
            reader.source.display(),
        ));
    }
    Ok(Some(AdmissionKeyTable {
        section,
        expected_kind,
        name_hash_offset,
        name_len_offset,
        record_offset,
        slots,
    }))
}

fn test_key_table(reader: &ElfStampReader<'_>) -> Result<Option<AdmissionKeyTable>, String> {
    key_table(
        reader,
        TEST_KEY_SECTION,
        ADMISSION_KIND_TEST_KEY,
        size_of::<AdmissionTestKeyV2>(),
        offset_of!(AdmissionTestKeyV2, name_hash),
        offset_of!(AdmissionTestKeyV2, name_len),
        offset_of!(AdmissionTestKeyV2, record),
    )
}

fn preset_key_table(reader: &ElfStampReader<'_>) -> Result<Option<AdmissionKeyTable>, String> {
    key_table(
        reader,
        PRESET_KEY_SECTION,
        ADMISSION_KIND_PRESET_KEY,
        size_of::<AdmissionPresetKeyV2>(),
        offset_of!(AdmissionPresetKeyV2, name_hash),
        offset_of!(AdmissionPresetKeyV2, name_len),
        offset_of!(AdmissionPresetKeyV2, record),
    )
}

fn key_name_hash(
    reader: &ElfStampReader<'_>,
    table: &AdmissionKeyTable,
    slot: AdmissionKeySlot,
) -> Result<u64, String> {
    reader.raw_u64(
        slot.base + table.name_hash_offset as u64,
        &format!("{} record {} name hash", table.section, slot.index),
    )
}

fn key_name_len(
    reader: &ElfStampReader<'_>,
    table: &AdmissionKeyTable,
    slot: AdmissionKeySlot,
) -> Result<u64, String> {
    reader.raw_u64(
        slot.base + table.name_len_offset as u64,
        &format!("{} record {} name length", table.section, slot.index),
    )
}

fn key_record_target(
    reader: &ElfStampReader<'_>,
    table: &AdmissionKeyTable,
    slot: AdmissionKeySlot,
) -> Result<u64, String> {
    reader.pointer(
        slot.base,
        table.record_offset,
        &format!("{} record {} target", table.section, slot.index),
    )
}

fn matching_key_targets(
    reader: &ElfStampReader<'_>,
    table: &AdmissionKeyTable,
    name: &str,
) -> Result<Vec<u64>, String> {
    let hash = admission_name_hash(name);
    let len = name.len() as u64;
    table
        .slots
        .iter()
        .copied()
        .filter(|slot| slot.kind == table.expected_kind)
        .filter_map(|slot| {
            let matches = (|| {
                Ok::<_, String>(
                    key_name_hash(reader, table, slot)? == hash
                        && key_name_len(reader, table, slot)? == len,
                )
            })();
            match matches {
                Ok(true) => Some(key_record_target(reader, table, slot)),
                Ok(false) => None,
                Err(error) => Some(Err(error)),
            }
        })
        .collect()
}

fn validate_key_registry(
    reader: &ElfStampReader<'_>,
    table: &AdmissionKeyTable,
    records: &BTreeMap<u64, &str>,
) -> Result<(), String> {
    let mut bindings = Vec::new();
    for slot in table
        .slots
        .iter()
        .copied()
        .filter(|slot| slot.kind == table.expected_kind)
    {
        bindings.push((
            slot.index,
            key_record_target(reader, table, slot)?,
            key_name_hash(reader, table, slot)?,
            key_name_len(reader, table, slot)?,
        ));
    }
    validate_key_bindings(reader.source, table.section, records, &bindings)
}

fn validate_key_bindings(
    source: &Path,
    section: &str,
    records: &BTreeMap<u64, &str>,
    bindings: &[(usize, u64, u64, u64)],
) -> Result<(), String> {
    if bindings.len() != records.len() {
        return Err(format!(
            "admission ELF {} has {} {} records but {key_count} compact lookup keys",
            source.display(),
            records.len(),
            section,
            key_count = bindings.len(),
        ));
    }
    let mut seen = BTreeSet::new();
    for &(index, target, encoded_hash, encoded_len) in bindings {
        let name = records.get(&target).ok_or_else(|| {
            format!(
                "{} record {} in admission ELF {} points to 0x{target:x}, which is not a \
                 version-{ADMISSION_VERSION} admission record",
                section,
                index,
                source.display(),
            )
        })?;
        if !seen.insert(target) {
            return Err(format!(
                "admission ELF {} contains duplicate {} keys for record {:?}",
                source.display(),
                section,
                name,
            ));
        }
        if encoded_hash != admission_name_hash(name) || encoded_len != name.len() as u64 {
            return Err(format!(
                "{} record {} in admission ELF {} does not match its target name {:?}",
                section,
                index,
                source.display(),
                name,
            ));
        }
    }
    Ok(())
}

fn validate_test_registry(
    reader: &ElfStampReader<'_>,
    tests: &[ParsedAdmissionTest],
) -> Result<(), String> {
    let (registry_va, bytes) = reader.section(LEGACY_TEST_SECTION)?.ok_or_else(|| {
        format!(
            "admission ELF {} is missing {LEGACY_TEST_SECTION}",
            reader.source.display(),
        )
    })?;
    if !bytes.len().is_multiple_of(size_of::<KtstrTestEntry>()) {
        return Err(format!(
            "admission ELF {} has a malformed {LEGACY_TEST_SECTION} byte length {}",
            reader.source.display(),
            bytes.len(),
        ));
    }
    let linked_count = bytes.len() / size_of::<KtstrTestEntry>();
    if linked_count != tests.len() {
        return Err(format!(
            "admission ELF {} has {linked_count} linked test entries but {} version-{} \
             admission records; rebuild every selected test target with this ktstr version",
            reader.source.display(),
            tests.len(),
            ADMISSION_VERSION,
        ));
    }

    let record_size = size_of::<KtstrTestEntry>() as u64;
    let mut slots = BTreeSet::new();
    for test in tests {
        let relative = test.legacy_entry.checked_sub(registry_va).ok_or_else(|| {
            format!(
                "admission record {:?} in {} points before {LEGACY_TEST_SECTION}",
                test.name,
                reader.source.display(),
            )
        })?;
        if relative % record_size != 0 || relative / record_size >= linked_count as u64 {
            return Err(format!(
                "admission record {:?} in {} points outside or between \
                 {LEGACY_TEST_SECTION} entries",
                test.name,
                reader.source.display(),
            ));
        }
        if !slots.insert(relative / record_size) {
            return Err(format!(
                "admission ELF {} contains duplicate records for a legacy test slot",
                reader.source.display(),
            ));
        }
    }
    Ok(())
}

fn read_index_from_reader(reader: &ElfStampReader<'_>) -> Result<Option<AdmissionIndex>, String> {
    let tests = section_records(
        reader,
        TEST_SECTION,
        size_of::<AdmissionTestStampV2>(),
        ADMISSION_KIND_TEST,
        test_record,
    )?;
    let presets = section_records(
        reader,
        PRESET_SECTION,
        size_of::<AdmissionPresetStampV2>(),
        ADMISSION_KIND_PRESET,
        preset_record,
    )?;
    let test_keys = test_key_table(reader)?;
    let preset_keys = preset_key_table(reader)?;
    if tests.is_none() && presets.is_none() && test_keys.is_none() && preset_keys.is_none() {
        if reader.section(LEGACY_TEST_SECTION)?.is_some() {
            return Err(format!(
                "ktstr-linked test binary {} is missing required version-{} admission stamp \
                 and lookup sections; rebuild every selected test target",
                reader.source.display(),
                ADMISSION_VERSION,
            ));
        }
        return Ok(None);
    }
    let (tests, _) = tests.ok_or_else(|| {
        format!(
            "admission ELF {} is missing required {TEST_SECTION}",
            reader.source.display(),
        )
    })?;
    let (presets, preset_sentinel) = presets.ok_or_else(|| {
        format!(
            "admission ELF {} is missing required {PRESET_SECTION}",
            reader.source.display(),
        )
    })?;
    let test_keys = test_keys.ok_or_else(|| {
        format!(
            "admission ELF {} is missing required {TEST_KEY_SECTION}",
            reader.source.display(),
        )
    })?;
    let preset_keys = preset_keys.ok_or_else(|| {
        format!(
            "admission ELF {} is missing required {PRESET_KEY_SECTION}",
            reader.source.display(),
        )
    })?;

    validate_test_registry(reader, &tests)?;
    let tests = AdmissionIndex { tests, presets };

    let expected = reader.u32(
        preset_sentinel,
        offset_of!(AdmissionPresetStampV2, sentinel_expected_count),
        "admission preset sentinel expected count",
    )? as usize;
    if tests.presets.len() != expected {
        return Err(format!(
            "admission ELF {} contains {} preset records but its version sentinel requires \
             {expected}",
            reader.source.display(),
            tests.presets.len(),
        ));
    }

    let mut entry_names = BTreeSet::new();
    for test in &tests.tests {
        if test.name.is_empty() || !entry_names.insert(test.name.clone()) {
            return Err(format!(
                "admission ELF {} contains an empty or duplicate test-entry name {:?}",
                reader.source.display(),
                test.name,
            ));
        }
    }
    let mut preset_names = BTreeSet::new();
    for preset in &tests.presets {
        if preset.name.is_empty() || !preset_names.insert(preset.name.clone()) {
            return Err(format!(
                "admission ELF {} contains an empty or duplicate preset name {:?}",
                reader.source.display(),
                preset.name,
            ));
        }
    }

    let test_records = tests
        .tests
        .iter()
        .map(|record| (record.record_base, record.name.as_str()))
        .collect::<BTreeMap<_, _>>();
    let preset_records = tests
        .presets
        .iter()
        .map(|record| (record.record_base, record.name.as_str()))
        .collect::<BTreeMap<_, _>>();
    validate_key_registry(reader, &test_keys, &test_records)?;
    validate_key_registry(reader, &preset_keys, &preset_records)?;
    Ok(Some(tests))
}

#[cfg(test)]
fn read_index(path: &Path) -> Result<Option<AdmissionIndex>, String> {
    let file = std::fs::File::open(path).map_err(|error| {
        format!(
            "open warmed test binary {} for admission stamp: {error}",
            path.display(),
        )
    })?;
    // SAFETY: the map is read-only, the file stays alive through creation,
    // and the returned mapping owns its kernel VMA.
    let data = unsafe { memmap2::MmapOptions::new().map(&file) }.map_err(|error| {
        format!(
            "map warmed test binary {} for admission stamp: {error}",
            path.display(),
        )
    })?;
    let reader = ElfStampReader::new(path, &data)?;
    read_index_from_reader(&reader)
}

pub(super) fn validate_admission_stamp_reader(reader: &ElfStampReader<'_>) -> Result<(), String> {
    read_index_from_reader(reader).map(|_| ())
}

fn mode(test: &ParsedAdmissionTest) -> AdmissionMode {
    if test.performance_mode {
        AdmissionMode::Performance
    } else if test.no_perf_mode {
        AdmissionMode::NoPerf
    } else {
        AdmissionMode::Default
    }
}

fn descriptor_from_test(
    exact_name: &str,
    kind: AdmissionCellKind,
    test: &ParsedAdmissionTest,
    topology: AdmissionTopologyDescriptor,
    cpu_budget: Option<u32>,
    preset_name: Option<String>,
    kernel: Option<String>,
) -> AdmissionCellDescriptor {
    AdmissionCellDescriptor {
        exact_name: exact_name.to_string(),
        kind,
        entry_name: Some(test.name.clone()),
        preset_name,
        scheduler_name: None,
        kernel,
        topology,
        cpu_budget,
        mode: mode(test),
        host_only: test.host_only,
        performance_mode: test.performance_mode,
        no_perf_mode: test.no_perf_mode,
        expect_auto_repro: test.expect_auto_repro,
    }
}

#[cfg(test)]
fn optional_kernel_matches<'a>(
    rest: &str,
    tests: &'a [ParsedAdmissionTest],
) -> Vec<(&'a ParsedAdmissionTest, Option<String>)> {
    tests
        .iter()
        .filter_map(|test| {
            if rest == test.name {
                return Some((test, None));
            }
            let suffix = rest.strip_prefix(&format!("{}/", test.name))?;
            (!suffix.is_empty() && !suffix.contains('/')).then(|| (test, Some(suffix.to_string())))
        })
        .collect()
}

fn unique_match<T>(
    path: &Path,
    exact_name: &str,
    family: &str,
    mut matches: Vec<T>,
) -> Result<T, String> {
    match matches.len() {
        1 => Ok(matches.pop().expect("length checked")),
        0 => Err(format!(
            "generated {family} test name {exact_name:?} has no admission stamp in {}",
            path.display(),
        )),
        count => Err(format!(
            "generated {family} test name {exact_name:?} ambiguously matches {count} admission \
             stamps in {}",
            path.display(),
        )),
    }
}

struct DirectAdmissionTables {
    tests: AdmissionKeyTable,
    presets: AdmissionKeyTable,
}

fn validate_record_section_shape(
    reader: &ElfStampReader<'_>,
    section: &str,
    record_size: usize,
) -> Result<(), String> {
    let (_, bytes) = reader.section(section)?.ok_or_else(|| {
        format!(
            "admission ELF {} is missing required {section}",
            reader.source.display(),
        )
    })?;
    if bytes.is_empty() || !bytes.len().is_multiple_of(record_size) {
        return Err(format!(
            "corrupt admission section {section} in {}: byte length {} is not a non-zero \
             multiple of record size {record_size}",
            reader.source.display(),
            bytes.len(),
        ));
    }
    Ok(())
}

fn direct_tables(reader: &ElfStampReader<'_>) -> Result<Option<DirectAdmissionTables>, String> {
    let tests = test_key_table(reader)?;
    let presets = preset_key_table(reader)?;
    match (tests, presets) {
        (None, None) => {
            if reader.section(LEGACY_TEST_SECTION)?.is_some()
                || reader.section(TEST_SECTION)?.is_some()
                || reader.section(PRESET_SECTION)?.is_some()
            {
                return Err(format!(
                    "ktstr-linked test binary {} is missing required version-{} compact \
                     admission lookup sections; rebuild every selected test target",
                    reader.source.display(),
                    ADMISSION_VERSION,
                ));
            }
            Ok(None)
        }
        (Some(_), None) => Err(format!(
            "admission ELF {} contains {TEST_KEY_SECTION} but is missing {PRESET_KEY_SECTION}",
            reader.source.display(),
        )),
        (None, Some(_)) => Err(format!(
            "admission ELF {} contains {PRESET_KEY_SECTION} but is missing {TEST_KEY_SECTION}",
            reader.source.display(),
        )),
        (Some(tests), Some(presets)) => {
            validate_record_section_shape(reader, TEST_SECTION, size_of::<AdmissionTestStampV2>())?;
            validate_record_section_shape(
                reader,
                PRESET_SECTION,
                size_of::<AdmissionPresetStampV2>(),
            )?;
            Ok(Some(DirectAdmissionTables { tests, presets }))
        }
    }
}

fn target_record_index(
    reader: &ElfStampReader<'_>,
    section: &str,
    record_size: usize,
    expected_kind: u16,
    target: u64,
) -> Result<usize, String> {
    let (section_va, bytes) = reader.section(section)?.ok_or_else(|| {
        format!(
            "admission ELF {} is missing required {section}",
            reader.source.display(),
        )
    })?;
    let relative = target.checked_sub(section_va).ok_or_else(|| {
        format!(
            "compact admission key in {} points before {section}",
            reader.source.display(),
        )
    })?;
    if relative % record_size as u64 != 0
        || relative / record_size as u64 >= (bytes.len() / record_size) as u64
    {
        return Err(format!(
            "compact admission key in {} points outside or between {section} records",
            reader.source.display(),
        ));
    }
    let index = usize::try_from(relative / record_size as u64).map_err(|_| {
        format!(
            "compact admission key record index in {} is unrepresentable",
            reader.source.display(),
        )
    })?;
    let kind = header(reader, target, record_size, section, index)?;
    if kind != expected_kind {
        return Err(format!(
            "compact admission key in {} points to {section} record {index} with kind {kind}, \
             expected {expected_kind}",
            reader.source.display(),
        ));
    }
    Ok(index)
}

fn lookup_tests(
    reader: &ElfStampReader<'_>,
    table: &AdmissionKeyTable,
    name: &str,
) -> Result<Vec<ParsedAdmissionTest>, String> {
    let candidates = matching_key_targets(reader, table, name)?
        .into_iter()
        .map(|target| {
            let index = target_record_index(
                reader,
                TEST_SECTION,
                size_of::<AdmissionTestStampV2>(),
                ADMISSION_KIND_TEST,
                target,
            )?;
            test_record(reader, target, index)
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(retain_exact_test_names(candidates, name))
}

fn retain_exact_test_names(
    candidates: Vec<ParsedAdmissionTest>,
    requested: &str,
) -> Vec<ParsedAdmissionTest> {
    candidates
        .into_iter()
        .filter(|record| record.name == requested)
        .collect()
}

fn lookup_presets(
    reader: &ElfStampReader<'_>,
    table: &AdmissionKeyTable,
    name: &str,
) -> Result<Vec<ParsedAdmissionPreset>, String> {
    let candidates = matching_key_targets(reader, table, name)?
        .into_iter()
        .map(|target| {
            let index = target_record_index(
                reader,
                PRESET_SECTION,
                size_of::<AdmissionPresetStampV2>(),
                ADMISSION_KIND_PRESET,
                target,
            )?;
            preset_record(reader, target, index)
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(candidates
        .into_iter()
        .filter(|record| record.name == name)
        .collect())
}

fn direct_test_match(
    reader: &ElfStampReader<'_>,
    table: &AdmissionKeyTable,
    path: &Path,
    exact_name: &str,
    family: &str,
    name: &str,
    host_only: bool,
) -> Result<ParsedAdmissionTest, String> {
    unique_match(
        path,
        exact_name,
        family,
        lookup_tests(reader, table, name)?
            .into_iter()
            .filter(|test| test.host_only == host_only)
            .collect(),
    )
}

fn read_admission_cell_stamp_from_reader(
    reader: &ElfStampReader<'_>,
    path: &Path,
    exact_name: &str,
) -> Result<Option<AdmissionCellDescriptor>, String> {
    let tables = direct_tables(reader)?.ok_or_else(|| {
        format!(
            "generated ktstr test name {exact_name:?} came from {}, which has no admission \
             stamp sections",
            path.display(),
        )
    })?;

    if let Some(rest) = exact_name.strip_prefix("host/") {
        if rest.is_empty() {
            return Err(format!("malformed host test name {exact_name:?}"));
        }
        let test = direct_test_match(reader, &tables.tests, path, exact_name, "host", rest, true)?;
        return Ok(Some(descriptor_from_test(
            exact_name,
            AdmissionCellKind::Host,
            &test,
            test.topology.clone(),
            test.cpu_budget,
            None,
            None,
        )));
    }

    if let Some(rest) = exact_name.strip_prefix("ktstr/") {
        let mut matches = lookup_tests(reader, &tables.tests, rest)?
            .into_iter()
            .filter(|test| !test.host_only)
            .map(|test| (test, None))
            .collect::<Vec<_>>();
        if let Some((entry_name, kernel)) = rest.rsplit_once('/')
            && !entry_name.is_empty()
            && !kernel.is_empty()
        {
            matches.extend(
                lookup_tests(reader, &tables.tests, entry_name)?
                    .into_iter()
                    .filter(|test| !test.host_only)
                    .map(|test| (test, Some(kernel.to_string()))),
            );
        }
        let (test, kernel) = unique_match(path, exact_name, "ktstr", matches)?;
        return Ok(Some(descriptor_from_test(
            exact_name,
            AdmissionCellKind::Ktstr,
            &test,
            test.topology.clone(),
            test.cpu_budget,
            None,
            kernel,
        )));
    }

    if let Some(rest) = exact_name.strip_prefix("gauntlet/") {
        let mut forms = Vec::with_capacity(2);
        if let Some((entry, preset)) = rest.rsplit_once('/')
            && !entry.is_empty()
            && !preset.is_empty()
        {
            forms.push((entry, preset, None));
        }
        if let Some((without_kernel, kernel)) = rest.rsplit_once('/')
            && !kernel.is_empty()
            && let Some((entry, preset)) = without_kernel.rsplit_once('/')
            && !entry.is_empty()
            && !preset.is_empty()
        {
            forms.push((entry, preset, Some(kernel.to_string())));
        }
        let mut matches = Vec::new();
        for (entry_name, preset_name, kernel) in forms {
            let tests = lookup_tests(reader, &tables.tests, entry_name)?;
            let presets = lookup_presets(reader, &tables.presets, preset_name)?;
            for test in tests.into_iter().filter(|test| !test.host_only) {
                for preset in &presets {
                    matches.push((test.clone(), preset.clone(), kernel.clone()));
                }
            }
        }
        let (test, preset, kernel) = unique_match(path, exact_name, "gauntlet", matches)?;
        if preset.topology.llc_cores.is_some() {
            return Err(format!(
                "generated gauntlet test name {exact_name:?} selects verifier-only \
                 non-uniform preset {:?}",
                preset.name,
            ));
        }
        return Ok(Some(descriptor_from_test(
            exact_name,
            AdmissionCellKind::Gauntlet,
            &test,
            preset.topology,
            test.cpu_budget,
            Some(preset.name),
            kernel,
        )));
    }

    let rest = exact_name
        .strip_prefix("verifier/")
        .expect("reserved family checked before opening the ELF");
    let parts = rest.split('/').collect::<Vec<_>>();
    if parts.len() != 3 || parts.iter().any(|part| part.is_empty()) {
        return Err(format!("malformed verifier test name {exact_name:?}"));
    }
    let preset = unique_match(
        path,
        exact_name,
        "verifier",
        lookup_presets(reader, &tables.presets, parts[2])?,
    )?;
    Ok(Some(AdmissionCellDescriptor {
        exact_name: exact_name.to_string(),
        kind: AdmissionCellKind::Verifier,
        entry_name: None,
        preset_name: Some(preset.name),
        scheduler_name: Some(parts[0].to_string()),
        kernel: Some(parts[1].to_string()),
        topology: preset.topology,
        cpu_budget: preset.forced_cpu_budget,
        mode: AdmissionMode::NoPerf,
        host_only: false,
        performance_mode: false,
        no_perf_mode: true,
        expect_auto_repro: false,
    }))
}

/// Resolve one exact generated libtest name directly from a final ELF.
///
/// Unrelated ordinary Rust tests return `Ok(None)`.  Names in the reserved
/// `host/`, `ktstr/`, `gauntlet/`, and `verifier/` families are strict: a
/// malformed name, unknown entry/preset, ambiguous optional kernel suffix, or
/// corrupt/missing admission section is an error.  There is deliberately no
/// process-exec fallback and no mixed-version compatibility path.
pub fn read_admission_cell_stamp(
    path: &Path,
    exact_name: &str,
) -> Result<Option<AdmissionCellDescriptor>, String> {
    let family = exact_name.split_once('/').map(|(family, _)| family);
    if !matches!(family, Some("host" | "ktstr" | "gauntlet" | "verifier")) {
        return Ok(None);
    }
    let file = std::fs::File::open(path).map_err(|error| {
        format!(
            "open test binary {} for direct admission lookup: {error}",
            path.display(),
        )
    })?;
    // SAFETY: the mapping is read-only, the file stays alive through reader
    // construction, and the map owns its kernel VMA.
    let data = unsafe { memmap2::MmapOptions::new().map(&file) }.map_err(|error| {
        format!(
            "map test binary {} for direct admission lookup: {error}",
            path.display(),
        )
    })?;
    let reader = ElfStampReader::new(path, &data)?;
    read_admission_cell_stamp_from_reader(&reader, path, exact_name)
}

#[cfg(test)]
mod tests {
    use super::*;

    static EXPLICIT_NODES: [crate::test_support::NumaNode; 3] = [
        crate::test_support::NumaNode {
            llcs: 1,
            memory_mib: 128,
            latency_ns: None,
            bandwidth_mbs: None,
            mem_side_cache: None,
        },
        crate::test_support::NumaNode {
            llcs: 0,
            memory_mib: 128,
            latency_ns: None,
            bandwidth_mbs: None,
            mem_side_cache: None,
        },
        crate::test_support::NumaNode {
            llcs: 2,
            memory_mib: 128,
            latency_ns: None,
            bandwidth_mbs: None,
            mem_side_cache: None,
        },
    ];

    fn explicit_nodes_fixture(
        _ctx: &crate::scenario::Ctx,
    ) -> anyhow::Result<crate::assert::AssertResult> {
        Ok(crate::assert::AssertResult::pass())
    }

    #[crate::ktstr_test_entry]
    static ADMISSION_EXPLICIT_NODES_ENTRY: KtstrTestEntry = KtstrTestEntry {
        name: "__unit_test_admission_explicit_nodes__",
        func: explicit_nodes_fixture,
        topology: Topology {
            numa_nodes: 3,
            llcs: 3,
            cores_per_llc: 2,
            threads_per_core: 1,
            nodes: Some(&EXPLICIT_NODES),
            distances: None,
            llc_cores: None,
        },
        cpu_budget: Some(3),
        no_perf_mode: true,
        host_only: true,
        ..KtstrTestEntry::DEFAULT
    };

    #[crate::ktstr_test_entry]
    static ADMISSION_GUEST_ENTRY: KtstrTestEntry = KtstrTestEntry {
        name: "__unit_test_admission_guest__",
        func: explicit_nodes_fixture,
        performance_mode: true,
        ..KtstrTestEntry::DEFAULT
    };

    #[test]
    fn v2_wire_layout_is_pinned() {
        assert_eq!(size_of::<AdmissionHeaderV2>(), 16);
        assert_eq!(size_of::<AdmissionStrV2>(), 16);
        assert_eq!(size_of::<AdmissionOptionalU32V2>(), 8);
        assert_eq!(size_of::<AdmissionTopologyStampV2>(), 2064);
        assert_eq!(size_of::<AdmissionTestStampV2>(), 2120);
        assert_eq!(size_of::<AdmissionPresetStampV2>(), 2112);
        assert_eq!(size_of::<AdmissionTestKeyV2>(), 40);
        assert_eq!(size_of::<AdmissionPresetKeyV2>(), 40);
        assert_eq!(offset_of!(AdmissionTestStampV2, header), 0);
        assert_eq!(offset_of!(AdmissionTestStampV2, name), 16);
        assert_eq!(offset_of!(AdmissionTestStampV2, topology), 32);
        assert_eq!(offset_of!(AdmissionTestStampV2, cpu_budget), 2096);
        assert_eq!(offset_of!(AdmissionTestStampV2, legacy_entry), 2112);
        assert_eq!(offset_of!(AdmissionPresetStampV2, topology), 32);
        assert_eq!(offset_of!(AdmissionPresetStampV2, forced_cpu_budget), 2096);
        assert_eq!(offset_of!(AdmissionTestKeyV2, name_hash), 16);
        assert_eq!(offset_of!(AdmissionTestKeyV2, name_len), 24);
        assert_eq!(offset_of!(AdmissionTestKeyV2, record), 32);
    }

    #[test]
    fn exact_name_resolution_preserves_entry_and_preset_admission() {
        let executable = std::env::current_exe().expect("locate unit-test executable");

        let host = read_admission_cell_stamp(&executable, "host/cpu_budget_codegen_probe")
            .expect("read host admission")
            .expect("host descriptor");
        assert_eq!(host.kind, AdmissionCellKind::Host);
        assert_eq!(host.cpu_budget, Some(7));
        assert_eq!(host.mode, AdmissionMode::NoPerf);
        assert!(host.host_only);

        let explicit =
            read_admission_cell_stamp(&executable, "host/__unit_test_admission_explicit_nodes__")
                .expect("read explicit-node admission")
                .expect("explicit-node descriptor");
        assert_eq!(explicit.topology.node_llcs, Some(vec![1, 0, 2]));
        assert_eq!(explicit.topology.total_cpus(), 6);

        let guest = read_admission_cell_stamp(
            &executable,
            "ktstr/__unit_test_admission_guest__/kernel_7_3",
        )
        .expect("read base guest admission")
        .expect("base guest descriptor");
        assert_eq!(guest.mode, AdmissionMode::Performance);
        assert_eq!(guest.kernel.as_deref(), Some("kernel_7_3"));

        let gauntlet = read_admission_cell_stamp(
            &executable,
            "gauntlet/__unit_test_admission_guest__/4cpu-2llc-nosmt/kernel_7_3",
        )
        .expect("read gauntlet admission")
        .expect("gauntlet descriptor");
        assert_eq!(gauntlet.kind, AdmissionCellKind::Gauntlet);
        assert_eq!(gauntlet.topology.total_cpus(), 4);
        assert_eq!(gauntlet.topology.llcs, 2);
        assert_eq!(gauntlet.kernel.as_deref(), Some("kernel_7_3"));
        assert_eq!(gauntlet.mode, AdmissionMode::Performance);

        let verifier =
            read_admission_cell_stamp(&executable, "verifier/scx-ktstr/kernel_7_3/4cpu-1llc-nosmt")
                .expect("read verifier admission")
                .expect("verifier descriptor");
        assert_eq!(verifier.kind, AdmissionCellKind::Verifier);
        assert_eq!(verifier.mode, AdmissionMode::NoPerf);
        assert_eq!(verifier.scheduler_name.as_deref(), Some("scx-ktstr"));
        assert_eq!(verifier.topology.total_cpus(), 4);

        assert!(
            read_admission_cell_stamp(&executable, "ordinary_unit_test")
                .expect("unrelated name")
                .is_none()
        );
        let missing = read_admission_cell_stamp(&executable, "ktstr/__missing_direct_key__")
            .expect_err("reserved-family key miss must be strict");
        assert!(missing.contains("has no admission stamp"), "{missing}");
    }

    #[test]
    fn compact_lookup_hash_collision_still_requires_full_name_match() {
        let executable = std::env::current_exe().expect("locate unit-test executable");
        let index = read_index(&executable)
            .expect("read admission index")
            .expect("linked admission index");
        let prototype = index.tests.first().expect("at least one admission test");
        let mut collision = prototype.clone();
        collision.name = "different-name-with-the-same-synthetic-key".to_string();
        let selected =
            retain_exact_test_names(vec![collision, prototype.clone()], prototype.name.as_str());
        assert_eq!(selected.len(), 1);
        assert_eq!(selected[0].record_base, prototype.record_base);
    }

    #[test]
    fn compact_lookup_global_validation_rejects_duplicate_record_targets() {
        let source = Path::new("synthetic-admission-elf");
        let records = BTreeMap::from([(0x1000, "first"), (0x2000, "second")]);
        let bindings = [
            (0, 0x1000, admission_name_hash("first"), 5),
            (1, 0x1000, admission_name_hash("first"), 5),
        ];
        let error = validate_key_bindings(source, TEST_KEY_SECTION, &records, &bindings)
            .expect_err("duplicate compact key target must be rejected");
        assert!(error.contains("duplicate"), "{error}");
    }

    #[test]
    fn generated_name_grammar_is_strict_and_kernel_suffixes_are_ambiguity_checked() {
        let executable = std::env::current_exe().expect("locate unit-test executable");
        let malformed = read_admission_cell_stamp(&executable, "verifier/scheduler/kernel")
            .expect_err("verifier name without a preset must be rejected");
        assert!(malformed.contains("malformed verifier test name"));

        let index = read_index(&executable)
            .expect("read admission index")
            .expect("linked admission index");
        let prototype = index.tests.first().expect("at least one admission test");
        let mut short = prototype.clone();
        short.name = "ambiguous".to_string();
        let mut long = prototype.clone();
        long.name = "ambiguous/kernel".to_string();
        assert_eq!(
            optional_kernel_matches("ambiguous/kernel", &[short, long]).len(),
            2,
            "an exact entry and an entry-plus-kernel interpretation must remain ambiguous",
        );
    }

    #[cfg(not(target_arch = "aarch64"))]
    #[test]
    fn verifier_only_sparse_preset_keeps_forced_budget() {
        let executable = std::env::current_exe().expect("locate unit-test executable");
        let verifier = read_admission_cell_stamp(
            &executable,
            "verifier/scx-ktstr/kernel_7_3/192cpu-11llc-smt",
        )
        .expect("read verifier admission")
        .expect("verifier descriptor");
        assert_eq!(verifier.cpu_budget, Some(96));
        assert_eq!(verifier.topology.llc_cores.as_ref().map(Vec::len), Some(11));
        assert_eq!(verifier.topology.total_cpus(), 192);

        let error = read_admission_cell_stamp(
            &executable,
            "gauntlet/__unit_test_admission_guest__/192cpu-11llc-smt",
        )
        .expect_err("ordinary gauntlet must reject the sparse verifier-only preset");
        assert!(
            error.contains("verifier-only non-uniform preset"),
            "{error}"
        );
    }

    #[test]
    fn canned_preset_stamp_projection_matches_runtime_presets() {
        let executable = std::env::current_exe().expect("locate unit-test executable");
        let index = read_index(&executable)
            .expect("read admission index")
            .expect("linked admission index");
        let decoded = index
            .presets
            .into_iter()
            .map(|preset| (preset.name, (preset.topology, preset.forced_cpu_budget)))
            .collect::<BTreeMap<_, _>>();
        let runtime = crate::gauntlet::gauntlet_presets();
        assert_eq!(decoded.len(), runtime.len());
        for preset in runtime {
            let (topology, forced_budget) = decoded
                .get(preset.name)
                .unwrap_or_else(|| panic!("missing admission preset {:?}", preset.name));
            assert_eq!(topology.numa_nodes, preset.topology.numa_nodes);
            assert_eq!(topology.llcs, preset.topology.llcs);
            assert_eq!(topology.cores_per_llc, preset.topology.cores_per_llc);
            assert_eq!(topology.threads_per_core, preset.topology.threads_per_core);
            assert_eq!(
                topology.node_llcs.is_some(),
                preset.topology.nodes.is_some()
            );
            assert_eq!(
                topology.llc_cores.as_deref(),
                preset.topology.llc_cores,
                "{}",
                preset.name,
            );
            assert_eq!(*forced_budget, preset.forced_cpu_budget);
        }
    }
}
