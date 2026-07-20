//! Link-retained scheduler-manifest metadata and its no-exec ELF reader.
//!
//! `cargo ktstr` needs the scheduler declarations and test-to-scheduler
//! relationships from every warmed test binary. Starting each binary only to
//! serialize those link-time registries is needlessly expensive, especially
//! under coverage and on heavily oversubscribed CI hosts. The declaration and
//! test macros therefore emit a second, versioned registry whose records are
//! readable directly from the final ELF.
//!
//! The wire records deliberately do not expose the Rust layout of
//! [`Scheduler`](super::Scheduler) or [`KtstrTestEntry`](super::KtstrTestEntry).
//! They contain fixed-width scalars and explicit pointer/length descriptors.
//! The pointers are final-ELF virtual addresses: in PIEs the reader applies
//! dynamic `RELATIVE` relocations, while ET_EXEC and RELR-packed PIEs already
//! carry their addend in the pointer slot.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::mem::{offset_of, size_of};
use std::path::Path;
use std::sync::OnceLock;

use goblin::elf::{
    Elf, dynamic::Dynamic, program_header::ProgramHeader, reloc::RelocSection,
    section_header::SectionHeader,
};
use goblin::strtab::Strtab;
use linkme::distributed_slice;

use super::{
    BinaryKindJson, KtstrTestEntry, Scheduler, SchedulerArtifactRequirement, SchedulerJson,
    SchedulerListEntry, SchedulerManifestProbe, SchedulerSpec, SchedulerTestJson, Sysctl,
    SysctlJson, TopologyConstraintsJson, TopologyJson,
};

const STAMP_MAGIC: [u8; 8] = *b"KTSTRSM1";
const STAMP_VERSION: u16 = 1;
const STAMP_KIND_SENTINEL: u16 = 0;
const STAMP_KIND_DECLARATION: u16 = 1;
const STAMP_KIND_TEST: u16 = 2;

const DECLARATION_SECTION: &str = "linkme_KTSTR_SCHEDULER_MANIFEST_DECLARATIONS_V1";
const TEST_SECTION: &str = "linkme_KTSTR_SCHEDULER_MANIFEST_TESTS_V1";

const BINARY_EEVDF: u8 = 0;
const BINARY_DISCOVER: u8 = 1;
const BINARY_PATH: u8 = 2;
const BINARY_KERNEL_BUILTIN: u8 = 3;

#[repr(C)]
#[derive(Clone, Copy)]
struct StampHeaderV1 {
    magic: [u8; 8],
    version: u16,
    kind: u16,
    record_size: u32,
}

impl StampHeaderV1 {
    const fn new(kind: u16, record_size: usize) -> Self {
        Self {
            magic: STAMP_MAGIC,
            version: STAMP_VERSION,
            kind,
            record_size: record_size as u32,
        }
    }
}

/// Stable pointer/length string descriptor used by scheduler-manifest stamps.
///
/// This type is public only because proc-macro expansions in downstream
/// crates construct it. It is not a source-level extension point.
#[doc(hidden)]
#[repr(C)]
#[derive(Clone, Copy)]
pub struct SchedulerManifestStampStrV1 {
    ptr: *const u8,
    len: u64,
}

// Every pointer is produced from a `'static` immutable string.
unsafe impl Sync for SchedulerManifestStampStrV1 {}

impl SchedulerManifestStampStrV1 {
    /// Project one static string into the v1 ELF wire shape.
    #[doc(hidden)]
    pub const fn new(value: &'static str) -> Self {
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
struct StampOptionalStrV1 {
    value: SchedulerManifestStampStrV1,
    present: u8,
    reserved: [u8; 7],
}

impl StampOptionalStrV1 {
    const fn new(value: Option<&'static str>) -> Self {
        match value {
            Some(value) => Self {
                value: SchedulerManifestStampStrV1::new(value),
                present: 1,
                reserved: [0; 7],
            },
            None => Self {
                value: SchedulerManifestStampStrV1::empty(),
                present: 0,
                reserved: [0; 7],
            },
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
struct StampOptionalU32V1 {
    value: u32,
    present: u8,
    reserved: [u8; 3],
}

impl StampOptionalU32V1 {
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

/// Stable pointer/count slice descriptor used by scheduler-manifest stamps.
///
/// This type is public only for proc-macro output.
#[doc(hidden)]
#[repr(C)]
#[derive(Clone, Copy)]
pub struct SchedulerManifestStampSliceV1<T> {
    ptr: *const T,
    len: u64,
}

// All emitted slices point at immutable promoted/static records.
unsafe impl<T: Sync> Sync for SchedulerManifestStampSliceV1<T> {}

impl<T> SchedulerManifestStampSliceV1<T> {
    /// Project one static slice into the v1 ELF wire shape.
    #[doc(hidden)]
    pub const fn new(values: &'static [T]) -> Self {
        Self {
            ptr: values.as_ptr(),
            len: values.len() as u64,
        }
    }

    const fn empty() -> Self {
        Self {
            ptr: std::ptr::null(),
            len: 0,
        }
    }
}

/// Stable scheduler-sysctl record emitted by `declare_scheduler!`.
#[doc(hidden)]
#[repr(C)]
#[derive(Clone, Copy)]
pub struct SchedulerManifestStampSysctlV1 {
    key: SchedulerManifestStampStrV1,
    value: SchedulerManifestStampStrV1,
}

unsafe impl Sync for SchedulerManifestStampSysctlV1 {}

impl SchedulerManifestStampSysctlV1 {
    /// Project a const-evaluated [`Sysctl`] into the v1 ELF wire shape.
    #[doc(hidden)]
    pub const fn new(sysctl: Sysctl) -> Self {
        Self {
            key: SchedulerManifestStampStrV1::new(sysctl.key()),
            value: SchedulerManifestStampStrV1::new(sysctl.value()),
        }
    }
}

/// Compact scheduler projection attached to each `#[ktstr_test]` edge.
///
/// The complete declaration projection lives in
/// [`SchedulerManifestDeclarationStampV1`]. Test edges need only the fields
/// required to reconstruct the test mapping and executable requirements.
#[doc(hidden)]
#[repr(C)]
#[derive(Clone, Copy)]
pub struct SchedulerManifestUseStampV1 {
    name: SchedulerManifestStampStrV1,
    manifest_dir: SchedulerManifestStampStrV1,
    binary_value: SchedulerManifestStampStrV1,
    binary_kind: u8,
    reserved: [u8; 7],
}

unsafe impl Sync for SchedulerManifestUseStampV1 {}

impl SchedulerManifestUseStampV1 {
    /// Project an arbitrary const-evaluated scheduler into a test-edge stamp.
    #[doc(hidden)]
    pub const fn new(scheduler: &'static Scheduler) -> Self {
        let (binary_kind, binary_value) = match scheduler.binary {
            SchedulerSpec::Eevdf => (BINARY_EEVDF, SchedulerManifestStampStrV1::empty()),
            SchedulerSpec::Discover(value) => {
                (BINARY_DISCOVER, SchedulerManifestStampStrV1::new(value))
            }
            SchedulerSpec::Path(value) => (BINARY_PATH, SchedulerManifestStampStrV1::new(value)),
            SchedulerSpec::KernelBuiltin { .. } => {
                (BINARY_KERNEL_BUILTIN, SchedulerManifestStampStrV1::empty())
            }
        };
        Self {
            name: SchedulerManifestStampStrV1::new(scheduler.name),
            manifest_dir: SchedulerManifestStampStrV1::new(scheduler.manifest_dir),
            binary_value,
            binary_kind,
            reserved: [0; 7],
        }
    }
}

/// Version-1 full scheduler declaration record emitted by
/// `declare_scheduler!`.
#[doc(hidden)]
#[repr(C)]
#[derive(Clone, Copy)]
pub struct SchedulerManifestDeclarationStampV1 {
    header: StampHeaderV1,
    name: SchedulerManifestStampStrV1,
    manifest_dir: SchedulerManifestStampStrV1,
    binary_value: SchedulerManifestStampStrV1,
    binary_kind: u8,
    requires_smt: u8,
    reserved0: [u8; 6],
    num_numa_nodes: u32,
    num_llcs: u32,
    cores_per_llc: u32,
    threads_per_core: u32,
    min_numa_nodes: u32,
    max_numa_nodes: StampOptionalU32V1,
    min_llcs: u32,
    max_llcs: StampOptionalU32V1,
    min_cpus: u32,
    max_cpus: StampOptionalU32V1,
    reserved1: [u8; 4],
    sched_args: SchedulerManifestStampSliceV1<SchedulerManifestStampStrV1>,
    sysctls: SchedulerManifestStampSliceV1<SchedulerManifestStampSysctlV1>,
    kargs: SchedulerManifestStampSliceV1<SchedulerManifestStampStrV1>,
    cgroup_parent: StampOptionalStrV1,
    config_file: StampOptionalStrV1,
    kernels: SchedulerManifestStampSliceV1<SchedulerManifestStampStrV1>,
    verifier_exclude_topologies: SchedulerManifestStampSliceV1<SchedulerManifestStampStrV1>,
}

unsafe impl Sync for SchedulerManifestDeclarationStampV1 {}

impl SchedulerManifestDeclarationStampV1 {
    /// Build a declaration stamp from the compiler-evaluated scheduler and
    /// stable arrays emitted alongside the macro invocation.
    #[doc(hidden)]
    pub const fn new(
        scheduler: &'static Scheduler,
        sched_args: &'static [SchedulerManifestStampStrV1],
        sysctls: &'static [SchedulerManifestStampSysctlV1],
        kargs: &'static [SchedulerManifestStampStrV1],
        kernels: &'static [SchedulerManifestStampStrV1],
        verifier_exclude_topologies: &'static [SchedulerManifestStampStrV1],
    ) -> Self {
        let (binary_kind, binary_value) = match scheduler.binary {
            SchedulerSpec::Eevdf => (BINARY_EEVDF, SchedulerManifestStampStrV1::empty()),
            SchedulerSpec::Discover(value) => {
                (BINARY_DISCOVER, SchedulerManifestStampStrV1::new(value))
            }
            SchedulerSpec::Path(value) => (BINARY_PATH, SchedulerManifestStampStrV1::new(value)),
            SchedulerSpec::KernelBuiltin { .. } => {
                (BINARY_KERNEL_BUILTIN, SchedulerManifestStampStrV1::empty())
            }
        };
        Self {
            header: StampHeaderV1::new(
                STAMP_KIND_DECLARATION,
                size_of::<SchedulerManifestDeclarationStampV1>(),
            ),
            name: SchedulerManifestStampStrV1::new(scheduler.name),
            manifest_dir: SchedulerManifestStampStrV1::new(scheduler.manifest_dir),
            binary_value,
            binary_kind,
            requires_smt: scheduler.constraints.requires_smt as u8,
            reserved0: [0; 6],
            num_numa_nodes: scheduler.topology.numa_nodes,
            num_llcs: scheduler.topology.llcs,
            cores_per_llc: scheduler.topology.cores_per_llc,
            threads_per_core: scheduler.topology.threads_per_core,
            min_numa_nodes: scheduler.constraints.min_numa_nodes,
            max_numa_nodes: StampOptionalU32V1::new(scheduler.constraints.max_numa_nodes),
            min_llcs: scheduler.constraints.min_llcs,
            max_llcs: StampOptionalU32V1::new(scheduler.constraints.max_llcs),
            min_cpus: scheduler.constraints.min_cpus,
            max_cpus: StampOptionalU32V1::new(scheduler.constraints.max_cpus),
            reserved1: [0; 4],
            sched_args: SchedulerManifestStampSliceV1::new(sched_args),
            sysctls: SchedulerManifestStampSliceV1::new(sysctls),
            kargs: SchedulerManifestStampSliceV1::new(kargs),
            cgroup_parent: StampOptionalStrV1::new(match scheduler.cgroup_parent {
                Some(parent) => Some(parent.as_str()),
                None => None,
            }),
            config_file: StampOptionalStrV1::new(scheduler.config_file),
            kernels: SchedulerManifestStampSliceV1::new(kernels),
            verifier_exclude_topologies: SchedulerManifestStampSliceV1::new(
                verifier_exclude_topologies,
            ),
        }
    }

    const fn sentinel() -> Self {
        Self {
            header: StampHeaderV1::new(
                STAMP_KIND_SENTINEL,
                size_of::<SchedulerManifestDeclarationStampV1>(),
            ),
            name: SchedulerManifestStampStrV1::empty(),
            manifest_dir: SchedulerManifestStampStrV1::empty(),
            binary_value: SchedulerManifestStampStrV1::empty(),
            binary_kind: 0,
            requires_smt: 0,
            reserved0: [0; 6],
            num_numa_nodes: 0,
            num_llcs: 0,
            cores_per_llc: 0,
            threads_per_core: 0,
            min_numa_nodes: 0,
            max_numa_nodes: StampOptionalU32V1::new(None),
            min_llcs: 0,
            max_llcs: StampOptionalU32V1::new(None),
            min_cpus: 0,
            max_cpus: StampOptionalU32V1::new(None),
            reserved1: [0; 4],
            sched_args: SchedulerManifestStampSliceV1::empty(),
            sysctls: SchedulerManifestStampSliceV1::empty(),
            kargs: SchedulerManifestStampSliceV1::empty(),
            cgroup_parent: StampOptionalStrV1::new(None),
            config_file: StampOptionalStrV1::new(None),
            kernels: SchedulerManifestStampSliceV1::empty(),
            verifier_exclude_topologies: SchedulerManifestStampSliceV1::empty(),
        }
    }
}

/// Version-1 test-to-scheduler record emitted by `#[ktstr_test]`.
#[doc(hidden)]
#[repr(C)]
#[derive(Clone, Copy)]
pub struct SchedulerManifestTestStampV1 {
    header: StampHeaderV1,
    test: SchedulerManifestStampStrV1,
    schedulers: SchedulerManifestStampSliceV1<SchedulerManifestUseStampV1>,
    expected_scheduler_count: u64,
    legacy_entry: *const KtstrTestEntry,
}

unsafe impl Sync for SchedulerManifestTestStampV1 {}

impl SchedulerManifestTestStampV1 {
    /// Build one test record; the scheduler slice is primary first, then every
    /// staged scheduler in declaration order.
    #[doc(hidden)]
    pub const fn new(
        entry: &'static KtstrTestEntry,
        schedulers: &'static [SchedulerManifestUseStampV1],
    ) -> Self {
        Self {
            header: StampHeaderV1::new(STAMP_KIND_TEST, size_of::<SchedulerManifestTestStampV1>()),
            test: SchedulerManifestStampStrV1::new(entry.name),
            schedulers: SchedulerManifestStampSliceV1::new(schedulers),
            expected_scheduler_count: 1 + entry.staged_schedulers.len() as u64,
            legacy_entry: entry,
        }
    }

    const fn sentinel() -> Self {
        Self {
            header: StampHeaderV1::new(
                STAMP_KIND_SENTINEL,
                size_of::<SchedulerManifestTestStampV1>(),
            ),
            test: SchedulerManifestStampStrV1::empty(),
            schedulers: SchedulerManifestStampSliceV1::empty(),
            expected_scheduler_count: 0,
            legacy_entry: std::ptr::null(),
        }
    }
}

/// Link-retained v1 declaration records.
#[doc(hidden)]
#[distributed_slice]
pub static KTSTR_SCHEDULER_MANIFEST_DECLARATIONS_V1: [SchedulerManifestDeclarationStampV1];

/// Link-retained v1 test records.
#[doc(hidden)]
#[distributed_slice]
pub static KTSTR_SCHEDULER_MANIFEST_TESTS_V1: [SchedulerManifestTestStampV1];

#[distributed_slice(KTSTR_SCHEDULER_MANIFEST_DECLARATIONS_V1)]
static KTSTR_SCHEDULER_MANIFEST_DECLARATIONS_V1_SENTINEL: SchedulerManifestDeclarationStampV1 =
    SchedulerManifestDeclarationStampV1::sentinel();

#[distributed_slice(KTSTR_SCHEDULER_MANIFEST_TESTS_V1)]
static KTSTR_SCHEDULER_MANIFEST_TESTS_V1_SENTINEL: SchedulerManifestTestStampV1 =
    SchedulerManifestTestStampV1::sentinel();

#[derive(Clone, Copy)]
enum PointerRelocation {
    Relative(u64),
    RelativeInPlace,
    Unsupported(u32),
    Duplicate,
}

struct ElfStampReader<'a> {
    source: &'a Path,
    data: &'a [u8],
    elf: Elf<'a>,
    relative_relocation_type: u32,
    dynrela_relative_prefix: usize,
    dynrel_relative_prefix: usize,
    exceptional_pointer_relocations: OnceLock<HashMap<u64, PointerRelocation>>,
}

impl<'a> ElfStampReader<'a> {
    fn new(source: &'a Path, data: &'a [u8]) -> Result<Self, String> {
        // `Elf::parse` eagerly decodes the static and dynamic symbol tables,
        // every section relocation, symbol versions, and other metadata that
        // the stamp reader never consults. Rust test executables routinely
        // carry hundreds of megabytes of that data. Assemble a deliberately
        // sparse `Elf` view instead: ELF/program/section headers, the section
        // name table, and the two dynamic relocation tables needed to resolve
        // pointer fields in stamp records.
        let header = Elf::parse_header(data).map_err(|error| {
            format!(
                "parse scheduler-manifest ELF header {}: {error}",
                source.display()
            )
        })?;
        let ctx = goblin::container::Ctx::new(
            header.container().map_err(|error| {
                format!(
                    "parse scheduler-manifest ELF class {}: {error}",
                    source.display()
                )
            })?,
            header.endianness().map_err(|error| {
                format!(
                    "parse scheduler-manifest ELF byte order {}: {error}",
                    source.display()
                )
            })?,
        );
        let mut elf = Elf::lazy_parse(header).map_err(|error| {
            format!(
                "parse scheduler-manifest ELF identity {}: {error}",
                source.display()
            )
        })?;
        if !elf.is_64 {
            return Err(format!(
                "scheduler-manifest stamp in {} requires ELF64; found ELF32",
                source.display()
            ));
        }
        if !elf.little_endian {
            return Err(format!(
                "scheduler-manifest stamp in {} requires little-endian ELF",
                source.display()
            ));
        }
        if !matches!(
            elf.header.e_type,
            goblin::elf::header::ET_DYN | goblin::elf::header::ET_EXEC
        ) {
            return Err(format!(
                "scheduler-manifest stamp in {} requires ET_DYN or ET_EXEC; found e_type {}",
                source.display(),
                elf.header.e_type,
            ));
        }
        let relative_type = match elf.header.e_machine {
            goblin::elf::header::EM_X86_64 => goblin::elf::reloc::R_X86_64_RELATIVE,
            goblin::elf::header::EM_AARCH64 => goblin::elf::reloc::R_AARCH64_RELATIVE,
            machine => {
                return Err(format!(
                    "scheduler-manifest stamp in {} supports x86_64 and aarch64 ELF; \
                     found e_machine {machine}",
                    source.display(),
                ));
            }
        };

        elf.program_headers = ProgramHeader::parse(
            data,
            elf.header.e_phoff as usize,
            elf.header.e_phnum as usize,
            ctx,
        )
        .map_err(|error| {
            format!(
                "parse scheduler-manifest ELF program headers {}: {error}",
                source.display()
            )
        })?;
        elf.section_headers = SectionHeader::parse(
            data,
            elf.header.e_shoff as usize,
            elf.header.e_shnum as usize,
            ctx,
        )
        .map_err(|error| {
            format!(
                "parse scheduler-manifest ELF section headers {}: {error}",
                source.display()
            )
        })?;
        if !elf.section_headers.is_empty() {
            let section_name_index = if elf.header.e_shstrndx as usize
                == goblin::elf::section_header::SHN_XINDEX as usize
            {
                elf.section_headers[0].sh_link as usize
            } else {
                elf.header.e_shstrndx as usize
            };
            let section_names = elf.section_headers.get(section_name_index).ok_or_else(|| {
                format!(
                    "scheduler-manifest ELF {} has section-name index {} outside {} \
                         section headers",
                    source.display(),
                    section_name_index,
                    elf.section_headers.len(),
                )
            })?;
            elf.shdr_strtab = Strtab::parse(
                data,
                section_names.sh_offset as usize,
                section_names.sh_size as usize,
                0,
            )
            .map_err(|error| {
                format!(
                    "parse scheduler-manifest ELF section names {}: {error}",
                    source.display()
                )
            })?;
        }

        elf.dynamic = Dynamic::parse(data, &elf.program_headers, ctx).map_err(|error| {
            format!(
                "parse scheduler-manifest ELF dynamic table {}: {error}",
                source.display()
            )
        })?;
        if let Some(dynamic) = elf.dynamic.as_ref() {
            elf.dynrelas =
                RelocSection::parse(data, dynamic.info.rela, dynamic.info.relasz, true, ctx)
                    .map_err(|error| {
                        format!(
                            "parse scheduler-manifest ELF dynamic RELA table {}: {error}",
                            source.display()
                        )
                    })?;
            elf.dynrels =
                RelocSection::parse(data, dynamic.info.rel, dynamic.info.relsz, false, ctx)
                    .map_err(|error| {
                        format!(
                            "parse scheduler-manifest ELF dynamic REL table {}: {error}",
                            source.display()
                        )
                    })?;
        }
        let (declared_rela_count, declared_rel_count) = elf
            .dynamic
            .as_ref()
            .map(|dynamic| (dynamic.info.relacount, dynamic.info.relcount))
            .unwrap_or_default();
        let dynrela_relative_prefix = Self::relative_prefix_len(
            &elf.dynrelas,
            declared_rela_count,
            relative_type,
        );
        let dynrel_relative_prefix =
            Self::relative_prefix_len(&elf.dynrels, declared_rel_count, relative_type);

        Ok(Self {
            source,
            data,
            elf,
            relative_relocation_type: relative_type,
            dynrela_relative_prefix,
            dynrel_relative_prefix,
            exceptional_pointer_relocations: OnceLock::new(),
        })
    }

    fn relative_prefix_len(
        relocations: &RelocSection<'_>,
        declared_count: usize,
        relative_type: u32,
    ) -> usize {
        let declared_count = declared_count.min(relocations.len());
        if declared_count != 0 {
            return declared_count;
        }
        if relocations
            .get(0)
            .is_none_or(|relocation| relocation.r_type != relative_type)
        {
            return 0;
        }

        // Mold omits DT_REL[A]COUNT, but still emits an ordered leading block
        // of relative relocations followed by a non-relative suffix. Locate
        // that partition once in O(log n) so every stamp pointer can search
        // the actual relative block. A non-partitioned external ELF can only
        // make this fast path miss; the in-place pointer and lazy exceptional
        // index remain the correctness path for such inputs.
        if relocations
            .get(relocations.len() - 1)
            .is_some_and(|relocation| relocation.r_type == relative_type)
        {
            return relocations.len();
        }
        let mut start = 1usize;
        let mut end = relocations.len();
        while start < end {
            let middle = start + (end - start) / 2;
            if relocations
                .get(middle)
                .is_some_and(|relocation| relocation.r_type == relative_type)
            {
                start = middle + 1;
            } else {
                end = middle;
            }
        }
        start
    }

    fn relocation_at(
        &self,
        relocations: &RelocSection<'_>,
        relative_prefix: usize,
        field_va: u64,
    ) -> Option<goblin::elf::reloc::Reloc> {
        fn binary_search(
            relocations: &RelocSection<'_>,
            mut start: usize,
            mut end: usize,
            field_va: u64,
        ) -> Option<goblin::elf::reloc::Reloc> {
            while start < end {
                let middle = start + (end - start) / 2;
                let relocation = relocations
                    .get(middle)
                    .expect("middle is bounded by relocation count");
                match relocation.r_offset.cmp(&field_va) {
                    std::cmp::Ordering::Less => start = middle + 1,
                    std::cmp::Ordering::Greater => end = middle,
                    std::cmp::Ordering::Equal => return Some(relocation),
                }
            }
            None
        }

        // GNU ld/lld publish DT_REL[A]COUNT; mold's equivalent leading block
        // is derived once in `relative_prefix_len`. All three order that block
        // by r_offset, so a lookup touches O(log n) 24-byte entries instead of
        // materializing the 100k-300k unrelated relocations common in Rust
        // test executables.
        binary_search(
            relocations,
            0,
            relative_prefix.min(relocations.len()),
            field_va,
        )
    }

    fn pointer_relocation(&self, field_va: u64) -> Result<Option<PointerRelocation>, String> {
        let rela = self.relocation_at(
            &self.elf.dynrelas,
            self.dynrela_relative_prefix,
            field_va,
        );
        let rel =
            self.relocation_at(&self.elf.dynrels, self.dynrel_relative_prefix, field_va);
        self.classify_pointer_relocation(field_va, rela, rel)
    }

    fn exceptional_pointer_relocation(
        &self,
        field_va: u64,
    ) -> Result<Option<PointerRelocation>, String> {
        let relocations = self.exceptional_pointer_relocations.get_or_init(|| {
            let mut relocations = HashMap::new();
            for relocation in self.elf.dynrelas.iter().chain(self.elf.dynrels.iter()) {
                let kind = if relocation.r_type != self.relative_relocation_type {
                    PointerRelocation::Unsupported(relocation.r_type)
                } else {
                    match relocation.r_addend {
                        Some(addend) => PointerRelocation::Relative(addend as u64),
                        None => PointerRelocation::RelativeInPlace,
                    }
                };
                relocations
                    .entry(relocation.r_offset)
                    .and_modify(|existing| *existing = PointerRelocation::Duplicate)
                    .or_insert(kind);
            }
            relocations
        });
        match relocations.get(&field_va).copied() {
            Some(PointerRelocation::Duplicate) => Err(format!(
                "scheduler-manifest ELF {} carries duplicate dynamic relocations at \
                 virtual address 0x{field_va:x}",
                self.source.display()
            )),
            relocation => Ok(relocation),
        }
    }

    fn classify_pointer_relocation(
        &self,
        field_va: u64,
        rela: Option<goblin::elf::reloc::Reloc>,
        rel: Option<goblin::elf::reloc::Reloc>,
    ) -> Result<Option<PointerRelocation>, String> {
        let relocation = match (rela, rel) {
            (Some(_), Some(_)) => {
                return Err(format!(
                    "scheduler-manifest ELF {} carries both RELA and REL relocations at \
                     virtual address 0x{field_va:x}",
                    self.source.display()
                ));
            }
            (Some(relocation), None) | (None, Some(relocation)) => relocation,
            (None, None) => return Ok(None),
        };
        if relocation.r_type != self.relative_relocation_type {
            return Ok(Some(PointerRelocation::Unsupported(relocation.r_type)));
        }
        Ok(Some(match relocation.r_addend {
            Some(addend) => PointerRelocation::Relative(addend as u64),
            // REL stores the addend in the relocated pointer slot.
            None => PointerRelocation::RelativeInPlace,
        }))
    }

    fn section(&self, name: &str) -> Result<Option<(u64, &'a [u8])>, String> {
        let mut found = None;
        for section in &self.elf.section_headers {
            if self.elf.shdr_strtab.get_at(section.sh_name) != Some(name) {
                continue;
            }
            if found.is_some() {
                return Err(format!(
                    "scheduler-manifest ELF {} contains duplicate {name} sections",
                    self.source.display()
                ));
            }
            let start = usize::try_from(section.sh_offset).map_err(|_| {
                format!(
                    "scheduler-manifest section {name} in {} has an unrepresentable file offset",
                    self.source.display()
                )
            })?;
            let size = usize::try_from(section.sh_size).map_err(|_| {
                format!(
                    "scheduler-manifest section {name} in {} has an unrepresentable size",
                    self.source.display()
                )
            })?;
            let end = start.checked_add(size).ok_or_else(|| {
                format!(
                    "scheduler-manifest section {name} in {} overflows its file range",
                    self.source.display()
                )
            })?;
            let bytes = self.data.get(start..end).ok_or_else(|| {
                format!(
                    "scheduler-manifest section {name} in {} extends beyond the ELF file",
                    self.source.display()
                )
            })?;
            if section.sh_type != goblin::elf::section_header::SHT_PROGBITS
                || section.sh_flags & u64::from(goblin::elf::section_header::SHF_ALLOC) == 0
            {
                return Err(format!(
                    "scheduler-manifest section {name} in {} must be allocated PROGBITS",
                    self.source.display()
                ));
            }
            found = Some((section.sh_addr, bytes));
        }
        Ok(found)
    }

    fn va_file_offset(&self, va: u64, len: usize, what: &str) -> Result<usize, String> {
        for segment in &self.elf.program_headers {
            if segment.p_type != goblin::elf::program_header::PT_LOAD || va < segment.p_vaddr {
                continue;
            }
            let relative = va - segment.p_vaddr;
            let len_u64 = u64::try_from(len).map_err(|_| {
                format!(
                    "{what} in scheduler-manifest ELF {} has an unrepresentable length",
                    self.source.display()
                )
            })?;
            let end = relative.checked_add(len_u64).ok_or_else(|| {
                format!(
                    "{what} in scheduler-manifest ELF {} overflows its virtual range",
                    self.source.display()
                )
            })?;
            if end > segment.p_filesz {
                continue;
            }
            let offset = segment.p_offset.checked_add(relative).ok_or_else(|| {
                format!(
                    "{what} in scheduler-manifest ELF {} overflows its file offset",
                    self.source.display()
                )
            })?;
            let offset = usize::try_from(offset).map_err(|_| {
                format!(
                    "{what} in scheduler-manifest ELF {} has an unrepresentable file offset",
                    self.source.display()
                )
            })?;
            if offset
                .checked_add(len)
                .is_some_and(|end| end <= self.data.len())
            {
                return Ok(offset);
            }
        }
        Err(format!(
            "{what} in scheduler-manifest ELF {} points outside file-backed PT_LOAD data \
             (virtual address 0x{va:x}, length {len})",
            self.source.display()
        ))
    }

    fn bytes_at_va(&self, va: u64, len: usize, what: &str) -> Result<&'a [u8], String> {
        let offset = self.va_file_offset(va, len, what)?;
        Ok(&self.data[offset..offset + len])
    }

    fn u8(&self, base: u64, offset: usize, what: &str) -> Result<u8, String> {
        let va = base
            .checked_add(offset as u64)
            .ok_or_else(|| format!("{what} address overflow in {}", self.source.display()))?;
        Ok(self.bytes_at_va(va, 1, what)?[0])
    }

    fn u16(&self, base: u64, offset: usize, what: &str) -> Result<u16, String> {
        let va = base
            .checked_add(offset as u64)
            .ok_or_else(|| format!("{what} address overflow in {}", self.source.display()))?;
        let bytes: [u8; 2] = self
            .bytes_at_va(va, 2, what)?
            .try_into()
            .expect("length checked");
        Ok(u16::from_le_bytes(bytes))
    }

    fn u32(&self, base: u64, offset: usize, what: &str) -> Result<u32, String> {
        let va = base
            .checked_add(offset as u64)
            .ok_or_else(|| format!("{what} address overflow in {}", self.source.display()))?;
        let bytes: [u8; 4] = self
            .bytes_at_va(va, 4, what)?
            .try_into()
            .expect("length checked");
        Ok(u32::from_le_bytes(bytes))
    }

    fn raw_u64(&self, va: u64, what: &str) -> Result<u64, String> {
        let bytes: [u8; 8] = self
            .bytes_at_va(va, 8, what)?
            .try_into()
            .expect("length checked");
        Ok(u64::from_le_bytes(bytes))
    }

    fn pointer(&self, base: u64, offset: usize, what: &str) -> Result<u64, String> {
        let field_va = base.checked_add(offset as u64).ok_or_else(|| {
            format!(
                "{what} pointer address overflow in {}",
                self.source.display()
            )
        })?;
        match self.pointer_relocation(field_va)? {
            Some(PointerRelocation::Relative(addend)) => Ok(addend),
            Some(PointerRelocation::RelativeInPlace) => self.raw_u64(field_va, what),
            Some(PointerRelocation::Unsupported(kind)) => Err(format!(
                "{what} in scheduler-manifest ELF {} uses unsupported dynamic relocation \
                 type {kind} at virtual address 0x{field_va:x}",
                self.source.display()
            )),
            Some(PointerRelocation::Duplicate) => unreachable!(
                "duplicate relocations are rejected before pointer relocation is returned"
            ),
            // ET_EXEC stores an absolute VA. DT_RELR-packed ET_DYN stores the
            // relative relocation's addend in place, so the same read covers
            // both without depending on section headers for SHT_RELR.
            None => {
                let pointer = self.raw_u64(field_va, what)?;
                if pointer != 0 && self.va_file_offset(pointer, 1, what).is_ok() {
                    return Ok(pointer);
                }
                // DT_REL[A]COUNT does not itself promise offset ordering.
                // Normal GNU ld/lld/mold output takes the binary-search path,
                // while this fallback preserves correctness for a valid
                // unsorted relative prefix. Only a pointer whose in-place
                // value cannot already designate file-backed ELF data pays
                // to build the exceptional relocation index.
                match self.exceptional_pointer_relocation(field_va)? {
                    Some(PointerRelocation::Relative(addend)) => Ok(addend),
                    Some(PointerRelocation::RelativeInPlace) => Ok(pointer),
                    Some(PointerRelocation::Unsupported(kind)) => Err(format!(
                        "{what} in scheduler-manifest ELF {} uses unsupported dynamic \
                         relocation type {kind} at virtual address 0x{field_va:x}",
                        self.source.display()
                    )),
                    Some(PointerRelocation::Duplicate) => unreachable!(
                        "duplicate relocations are rejected before pointer relocation is returned"
                    ),
                    None => Ok(pointer),
                }
            }
        }
    }

    fn header(
        &self,
        base: u64,
        record_size: usize,
        section: &str,
        index: usize,
    ) -> Result<u16, String> {
        let what = format!("{section} record {index}");
        let magic = self.bytes_at_va(
            base + offset_of!(StampHeaderV1, magic) as u64,
            STAMP_MAGIC.len(),
            &what,
        )?;
        if magic != STAMP_MAGIC {
            return Err(format!(
                "corrupt scheduler-manifest stamp in {}: {what} has invalid magic {:?}",
                self.source.display(),
                magic,
            ));
        }
        let version = self.u16(base, offset_of!(StampHeaderV1, version), &what)?;
        if version != STAMP_VERSION {
            return Err(format!(
                "unsupported scheduler-manifest stamp version {version} in {} ({what}); \
                 cargo-ktstr requires version {STAMP_VERSION}",
                self.source.display()
            ));
        }
        let encoded_size = self.u32(base, offset_of!(StampHeaderV1, record_size), &what)?;
        if encoded_size as usize != record_size {
            return Err(format!(
                "corrupt scheduler-manifest stamp in {}: {what} declares record size \
                 {encoded_size}, expected {record_size}",
                self.source.display()
            ));
        }
        self.u16(base, offset_of!(StampHeaderV1, kind), &what)
    }

    fn string(&self, base: u64, what: &str) -> Result<String, String> {
        let len = self.raw_u64(
            base + offset_of!(SchedulerManifestStampStrV1, len) as u64,
            what,
        )?;
        let len = usize::try_from(len).map_err(|_| {
            format!(
                "{what} in scheduler-manifest ELF {} has an unrepresentable string length",
                self.source.display()
            )
        })?;
        if len == 0 {
            return Ok(String::new());
        }
        let pointer = self.pointer(base, offset_of!(SchedulerManifestStampStrV1, ptr), what)?;
        if pointer == 0 {
            return Err(format!(
                "{what} in scheduler-manifest ELF {} has a null pointer for a non-empty string",
                self.source.display()
            ));
        }
        let bytes = self.bytes_at_va(pointer, len, what)?;
        let value = std::str::from_utf8(bytes).map_err(|error| {
            format!(
                "{what} in scheduler-manifest ELF {} is not UTF-8: {error}",
                self.source.display()
            )
        })?;
        Ok(value.to_string())
    }

    fn optional_string(&self, base: u64, what: &str) -> Result<Option<String>, String> {
        let present = self.u8(
            base,
            offset_of!(StampOptionalStrV1, present),
            &format!("{what} presence"),
        )?;
        match present {
            0 => Ok(None),
            1 => self
                .string(base + offset_of!(StampOptionalStrV1, value) as u64, what)
                .map(Some),
            other => Err(format!(
                "{what} in scheduler-manifest ELF {} has invalid option tag {other}",
                self.source.display()
            )),
        }
    }

    fn optional_u32(&self, base: u64, what: &str) -> Result<Option<u32>, String> {
        let present = self.u8(
            base,
            offset_of!(StampOptionalU32V1, present),
            &format!("{what} presence"),
        )?;
        match present {
            0 => Ok(None),
            1 => self
                .u32(base, offset_of!(StampOptionalU32V1, value), what)
                .map(Some),
            other => Err(format!(
                "{what} in scheduler-manifest ELF {} has invalid option tag {other}",
                self.source.display()
            )),
        }
    }

    fn slice_descriptor<T>(&self, base: u64, what: &str) -> Result<(u64, usize), String> {
        let len = self.raw_u64(
            base + offset_of!(SchedulerManifestStampSliceV1<T>, len) as u64,
            what,
        )?;
        let len = usize::try_from(len).map_err(|_| {
            format!(
                "{what} in scheduler-manifest ELF {} has an unrepresentable element count",
                self.source.display()
            )
        })?;
        let bytes = len.checked_mul(size_of::<T>()).ok_or_else(|| {
            format!(
                "{what} in scheduler-manifest ELF {} overflows its byte length",
                self.source.display()
            )
        })?;
        if bytes == 0 {
            return Ok((0, 0));
        }
        let pointer = self.pointer(
            base,
            offset_of!(SchedulerManifestStampSliceV1<T>, ptr),
            what,
        )?;
        if pointer == 0 {
            return Err(format!(
                "{what} in scheduler-manifest ELF {} has a null pointer for {len} elements",
                self.source.display()
            ));
        }
        self.bytes_at_va(pointer, bytes, what)?;
        Ok((pointer, len))
    }

    fn string_slice(&self, base: u64, what: &str) -> Result<Vec<String>, String> {
        let (pointer, len) = self.slice_descriptor::<SchedulerManifestStampStrV1>(base, what)?;
        (0..len)
            .map(|index| {
                self.string(
                    pointer + (index * size_of::<SchedulerManifestStampStrV1>()) as u64,
                    &format!("{what}[{index}]"),
                )
            })
            .collect()
    }

    fn sysctl_slice(&self, base: u64, what: &str) -> Result<Vec<SysctlJson>, String> {
        let (pointer, len) = self.slice_descriptor::<SchedulerManifestStampSysctlV1>(base, what)?;
        (0..len)
            .map(|index| {
                let element =
                    pointer + (index * size_of::<SchedulerManifestStampSysctlV1>()) as u64;
                Ok(SysctlJson {
                    key: self.string(
                        element + offset_of!(SchedulerManifestStampSysctlV1, key) as u64,
                        &format!("{what}[{index}].key"),
                    )?,
                    value: self.string(
                        element + offset_of!(SchedulerManifestStampSysctlV1, value) as u64,
                        &format!("{what}[{index}].value"),
                    )?,
                })
            })
            .collect()
    }

    fn binary_kind(&self, kind: u8, value_base: u64, what: &str) -> Result<BinaryKindJson, String> {
        match kind {
            BINARY_EEVDF => Ok(BinaryKindJson::Eevdf),
            BINARY_DISCOVER => self
                .string(value_base, &format!("{what} binary value"))
                .map(BinaryKindJson::Discover),
            BINARY_PATH => self
                .string(value_base, &format!("{what} binary value"))
                .map(BinaryKindJson::Path),
            BINARY_KERNEL_BUILTIN => Ok(BinaryKindJson::KernelBuiltin),
            other => Err(format!(
                "{what} in scheduler-manifest ELF {} has invalid binary kind {other}",
                self.source.display()
            )),
        }
    }

    fn declaration(&self, base: u64, index: usize) -> Result<SchedulerJson, String> {
        let what = format!("declaration record {index}");
        let binary_kind_tag = self.u8(
            base,
            offset_of!(SchedulerManifestDeclarationStampV1, binary_kind),
            &what,
        )?;
        let requires_smt = match self.u8(
            base,
            offset_of!(SchedulerManifestDeclarationStampV1, requires_smt),
            &what,
        )? {
            0 => false,
            1 => true,
            other => {
                return Err(format!(
                    "{what} in scheduler-manifest ELF {} has invalid boolean {other}",
                    self.source.display()
                ));
            }
        };
        Ok(SchedulerJson {
            name: self.string(
                base + offset_of!(SchedulerManifestDeclarationStampV1, name) as u64,
                &format!("{what}.name"),
            )?,
            manifest_dir: self.string(
                base + offset_of!(SchedulerManifestDeclarationStampV1, manifest_dir) as u64,
                &format!("{what}.manifest_dir"),
            )?,
            binary_kind: self.binary_kind(
                binary_kind_tag,
                base + offset_of!(SchedulerManifestDeclarationStampV1, binary_value) as u64,
                &what,
            )?,
            topology: TopologyJson {
                num_numa_nodes: self.u32(
                    base,
                    offset_of!(SchedulerManifestDeclarationStampV1, num_numa_nodes),
                    &what,
                )?,
                num_llcs: self.u32(
                    base,
                    offset_of!(SchedulerManifestDeclarationStampV1, num_llcs),
                    &what,
                )?,
                cores_per_llc: self.u32(
                    base,
                    offset_of!(SchedulerManifestDeclarationStampV1, cores_per_llc),
                    &what,
                )?,
                threads_per_core: self.u32(
                    base,
                    offset_of!(SchedulerManifestDeclarationStampV1, threads_per_core),
                    &what,
                )?,
            },
            sched_args: self.string_slice(
                base + offset_of!(SchedulerManifestDeclarationStampV1, sched_args) as u64,
                &format!("{what}.sched_args"),
            )?,
            sysctls: self.sysctl_slice(
                base + offset_of!(SchedulerManifestDeclarationStampV1, sysctls) as u64,
                &format!("{what}.sysctls"),
            )?,
            kargs: self.string_slice(
                base + offset_of!(SchedulerManifestDeclarationStampV1, kargs) as u64,
                &format!("{what}.kargs"),
            )?,
            cgroup_parent: self.optional_string(
                base + offset_of!(SchedulerManifestDeclarationStampV1, cgroup_parent) as u64,
                &format!("{what}.cgroup_parent"),
            )?,
            config_file: self.optional_string(
                base + offset_of!(SchedulerManifestDeclarationStampV1, config_file) as u64,
                &format!("{what}.config_file"),
            )?,
            kernels: self.string_slice(
                base + offset_of!(SchedulerManifestDeclarationStampV1, kernels) as u64,
                &format!("{what}.kernels"),
            )?,
            verifier_exclude_topologies: self.string_slice(
                base + offset_of!(
                    SchedulerManifestDeclarationStampV1,
                    verifier_exclude_topologies
                ) as u64,
                &format!("{what}.verifier_exclude_topologies"),
            )?,
            constraints: TopologyConstraintsJson {
                min_numa_nodes: self.u32(
                    base,
                    offset_of!(SchedulerManifestDeclarationStampV1, min_numa_nodes),
                    &what,
                )?,
                max_numa_nodes: self.optional_u32(
                    base + offset_of!(SchedulerManifestDeclarationStampV1, max_numa_nodes) as u64,
                    &format!("{what}.max_numa_nodes"),
                )?,
                min_llcs: self.u32(
                    base,
                    offset_of!(SchedulerManifestDeclarationStampV1, min_llcs),
                    &what,
                )?,
                max_llcs: self.optional_u32(
                    base + offset_of!(SchedulerManifestDeclarationStampV1, max_llcs) as u64,
                    &format!("{what}.max_llcs"),
                )?,
                requires_smt,
                min_cpus: self.u32(
                    base,
                    offset_of!(SchedulerManifestDeclarationStampV1, min_cpus),
                    &what,
                )?,
                max_cpus: self.optional_u32(
                    base + offset_of!(SchedulerManifestDeclarationStampV1, max_cpus) as u64,
                    &format!("{what}.max_cpus"),
                )?,
            },
        })
    }

    fn scheduler_use(&self, base: u64, what: &str) -> Result<ParsedSchedulerUse, String> {
        let kind = self.u8(
            base,
            offset_of!(SchedulerManifestUseStampV1, binary_kind),
            what,
        )?;
        Ok(ParsedSchedulerUse {
            name: self.string(
                base + offset_of!(SchedulerManifestUseStampV1, name) as u64,
                &format!("{what}.name"),
            )?,
            manifest_dir: self.string(
                base + offset_of!(SchedulerManifestUseStampV1, manifest_dir) as u64,
                &format!("{what}.manifest_dir"),
            )?,
            binary_kind: self.binary_kind(
                kind,
                base + offset_of!(SchedulerManifestUseStampV1, binary_value) as u64,
                what,
            )?,
        })
    }

    fn test(&self, base: u64, index: usize) -> Result<ParsedTestStamp, String> {
        let what = format!("test record {index}");
        let test = self.string(
            base + offset_of!(SchedulerManifestTestStampV1, test) as u64,
            &format!("{what}.test"),
        )?;
        let (pointer, len) = self.slice_descriptor::<SchedulerManifestUseStampV1>(
            base + offset_of!(SchedulerManifestTestStampV1, schedulers) as u64,
            &format!("{what}.schedulers"),
        )?;
        let expected = self.raw_u64(
            base + offset_of!(SchedulerManifestTestStampV1, expected_scheduler_count) as u64,
            &format!("{what}.expected_scheduler_count"),
        )?;
        if len as u64 != expected {
            return Err(format!(
                "{what} in scheduler-manifest ELF {} encodes {len} scheduler edge(s), \
                 but its manual KtstrTestEntry declares {expected}; use #[ktstr_test] \
                 for entries with staged schedulers",
                self.source.display()
            ));
        }
        if len == 0 {
            return Err(format!(
                "{what} in scheduler-manifest ELF {} carries no primary scheduler",
                self.source.display()
            ));
        }
        let schedulers = (0..len)
            .map(|scheduler_index| {
                self.scheduler_use(
                    pointer + (scheduler_index * size_of::<SchedulerManifestUseStampV1>()) as u64,
                    &format!("{what}.schedulers[{scheduler_index}]"),
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        let legacy_entry = self.pointer(
            base,
            offset_of!(SchedulerManifestTestStampV1, legacy_entry),
            &format!("{what}.legacy_entry"),
        )?;
        Ok(ParsedTestStamp {
            test,
            schedulers,
            legacy_entry,
        })
    }
}

#[derive(Debug)]
struct ParsedSchedulerUse {
    name: String,
    manifest_dir: String,
    binary_kind: BinaryKindJson,
}

#[derive(Debug)]
struct ParsedTestStamp {
    test: String,
    schedulers: Vec<ParsedSchedulerUse>,
    legacy_entry: u64,
}

fn parse_record_section<T>(
    reader: &ElfStampReader<'_>,
    section_name: &str,
    wire_size: usize,
    expected_kind: u16,
    mut parse: impl FnMut(u64, usize) -> Result<T, String>,
) -> Result<Option<Vec<T>>, String> {
    let Some((section_va, bytes)) = reader.section(section_name)? else {
        return Ok(None);
    };
    if bytes.is_empty() || bytes.len() % wire_size != 0 {
        return Err(format!(
            "corrupt scheduler-manifest section {section_name} in {}: byte length {} \
             is not a non-zero multiple of record size {wire_size}",
            reader.source.display(),
            bytes.len(),
        ));
    }
    let mut sentinel_count = 0usize;
    let mut records = Vec::new();
    for index in 0..(bytes.len() / wire_size) {
        let base = section_va + (index * wire_size) as u64;
        match reader.header(base, wire_size, section_name, index)? {
            STAMP_KIND_SENTINEL => sentinel_count += 1,
            kind if kind == expected_kind => records.push(parse(base, index)?),
            kind => {
                return Err(format!(
                    "corrupt scheduler-manifest section {section_name} in {}: record {index} \
                     has kind {kind}, expected sentinel or {expected_kind}",
                    reader.source.display()
                ));
            }
        }
    }
    if sentinel_count != 1 {
        return Err(format!(
            "corrupt scheduler-manifest section {section_name} in {}: expected exactly one \
             version sentinel, found {sentinel_count}",
            reader.source.display()
        ));
    }
    Ok(Some(records))
}

fn reconstruct_manifest(
    declarations: Vec<SchedulerJson>,
    tests: Vec<ParsedTestStamp>,
) -> Result<SchedulerManifestProbe, String> {
    let mut test_counts: HashMap<String, usize> = HashMap::new();
    let mut test_mappings = Vec::new();

    struct RequirementAccumulator {
        binary_kind: BinaryKindJson,
        manifest_dir: String,
        schedulers: BTreeSet<String>,
        use_count: usize,
    }
    let mut requirements: BTreeMap<(u8, String, String), RequirementAccumulator> = BTreeMap::new();

    for test in tests {
        for scheduler in test.schedulers {
            let count = test_counts.entry(scheduler.name.clone()).or_default();
            *count = count
                .checked_add(1)
                .ok_or_else(|| format!("scheduler {:?} test-count overflow", scheduler.name))?;
            test_mappings.push(SchedulerTestJson {
                test: test.test.clone(),
                scheduler: scheduler.name.clone(),
            });
            let (kind_order, value) = match &scheduler.binary_kind {
                BinaryKindJson::Discover(value) => (0, value.clone()),
                BinaryKindJson::Path(value) => (1, value.clone()),
                BinaryKindJson::Eevdf | BinaryKindJson::KernelBuiltin => continue,
            };
            let key = (kind_order, value, scheduler.manifest_dir.clone());
            let requirement = requirements
                .entry(key)
                .or_insert_with(|| RequirementAccumulator {
                    binary_kind: scheduler.binary_kind.clone(),
                    manifest_dir: scheduler.manifest_dir.clone(),
                    schedulers: BTreeSet::new(),
                    use_count: 0,
                });
            requirement.schedulers.insert(scheduler.name);
            requirement.use_count = requirement
                .use_count
                .checked_add(1)
                .ok_or_else(|| "scheduler artifact requirement use-count overflow".to_string())?;
        }
    }

    Ok(SchedulerManifestProbe {
        declarations: declarations
            .into_iter()
            .map(|scheduler| SchedulerListEntry {
                test_count: test_counts.get(&scheduler.name).copied().unwrap_or(0),
                scheduler,
            })
            .collect(),
        artifact_requirements: requirements
            .into_values()
            .map(|requirement| SchedulerArtifactRequirement {
                binary_kind: requirement.binary_kind,
                manifest_dir: requirement.manifest_dir,
                schedulers: requirement.schedulers.into_iter().collect(),
                use_count: requirement.use_count,
            })
            .collect(),
        tests: test_mappings,
    })
}

fn validate_registry_count(
    reader: &ElfStampReader<'_>,
    section: &str,
    element_size: usize,
    stamped_count: usize,
    records: &str,
) -> Result<(), String> {
    let Some((_, bytes)) = reader.section(section)? else {
        return Err(format!(
            "scheduler-manifest ELF {} has v1 {records} but is missing its linked \
             runtime registry section {section}; rebuild the selected target",
            reader.source.display()
        ));
    };
    if bytes.len() % element_size != 0 {
        return Err(format!(
            "scheduler-manifest ELF {} has a malformed {section} byte length {}; \
             expected a multiple of {element_size}",
            reader.source.display(),
            bytes.len()
        ));
    }
    let linked_count = bytes.len() / element_size;
    if linked_count != stamped_count {
        return Err(format!(
            "scheduler-manifest ELF {} has {linked_count} linked {records} but only \
             {stamped_count} version-{STAMP_VERSION} stamp record(s); rebuild all \
             dependencies and register manual tests with #[ktstr::ktstr_test_entry]",
            reader.source.display()
        ));
    }
    Ok(())
}

fn order_tests_by_legacy_registry(
    reader: &ElfStampReader<'_>,
    tests: &mut [ParsedTestStamp],
) -> Result<(), String> {
    let section = "linkme_KTSTR_TESTS";
    let (base, bytes) = reader.section(section)?.ok_or_else(|| {
        format!(
            "scheduler-manifest ELF {} is missing {section}",
            reader.source.display()
        )
    })?;
    let record_size = size_of::<KtstrTestEntry>() as u64;
    let record_count = bytes.len() / size_of::<KtstrTestEntry>();
    let mut seen = BTreeSet::new();
    for test in tests.iter() {
        let relative = test.legacy_entry.checked_sub(base).ok_or_else(|| {
            format!(
                "test {:?} scheduler-manifest stamp in {} points before {section}",
                test.test,
                reader.source.display()
            )
        })?;
        if relative % record_size != 0 {
            return Err(format!(
                "test {:?} scheduler-manifest stamp in {} points to an unaligned \
                 {section} address 0x{:x}",
                test.test,
                reader.source.display(),
                test.legacy_entry
            ));
        }
        let index = usize::try_from(relative / record_size).map_err(|_| {
            format!(
                "test {:?} scheduler-manifest registry index is unrepresentable in {}",
                test.test,
                reader.source.display()
            )
        })?;
        if index >= record_count {
            return Err(format!(
                "test {:?} scheduler-manifest stamp in {} points outside {section}",
                test.test,
                reader.source.display()
            ));
        }
        if !seen.insert(index) {
            return Err(format!(
                "scheduler-manifest ELF {} has duplicate stamps for {section} record {index}",
                reader.source.display()
            ));
        }
    }
    tests.sort_by_key(|test| test.legacy_entry);
    Ok(())
}

/// Read the versioned scheduler manifest directly from a final ELF.
///
/// `Ok(None)` means the binary does not link ktstr's v1 stamp registries (for
/// example, an unrelated test target in the same Cargo selection). Once either
/// v1 section is present, the format is strict: a missing peer section,
/// missing sentinel, unsupported version, invalid relocation, or out-of-range
/// pointer is a corruption error. There is intentionally no process-exec
/// fallback.
#[doc(hidden)]
pub fn read_scheduler_manifest_stamp(
    path: &Path,
) -> Result<Option<SchedulerManifestProbe>, String> {
    let file = std::fs::File::open(path).map_err(|error| {
        format!(
            "open warmed test binary {} for scheduler-manifest stamp: {error}",
            path.display()
        )
    })?;
    // SAFETY: the mapping is read-only, `file` stays alive through creation,
    // and no mutable alias is created. The returned map owns the kernel VMA.
    let data = unsafe { memmap2::MmapOptions::new().map(&file) }.map_err(|error| {
        format!(
            "map warmed test binary {} for scheduler-manifest stamp: {error}",
            path.display()
        )
    })?;
    let reader = ElfStampReader::new(path, &data)?;
    let declarations = parse_record_section(
        &reader,
        DECLARATION_SECTION,
        size_of::<SchedulerManifestDeclarationStampV1>(),
        STAMP_KIND_DECLARATION,
        |base, index| reader.declaration(base, index),
    )?;
    let tests = parse_record_section(
        &reader,
        TEST_SECTION,
        size_of::<SchedulerManifestTestStampV1>(),
        STAMP_KIND_TEST,
        |base, index| reader.test(base, index),
    )?;
    match (declarations, tests) {
        (None, None) => {
            let linked_legacy_registry = reader.section("linkme_KTSTR_SCHEDULERS")?.is_some()
                || reader.section("linkme_KTSTR_TESTS")?.is_some();
            if linked_legacy_registry {
                Err(format!(
                    "ktstr-linked test binary {} is missing required version-{STAMP_VERSION} \
                     scheduler-manifest stamp sections; rebuild every selected test target \
                     with this cargo-ktstr/ktstr version",
                    path.display()
                ))
            } else {
                Ok(None)
            }
        }
        (Some(_), None) => Err(format!(
            "scheduler-manifest ELF {} contains {DECLARATION_SECTION} but is missing \
             required peer section {TEST_SECTION}",
            path.display()
        )),
        (None, Some(_)) => Err(format!(
            "scheduler-manifest ELF {} contains {TEST_SECTION} but is missing \
             required peer section {DECLARATION_SECTION}",
            path.display()
        )),
        (Some(declarations), Some(mut tests)) => {
            validate_registry_count(
                &reader,
                "linkme_KTSTR_SCHEDULERS",
                size_of::<&Scheduler>(),
                declarations.len(),
                "scheduler declaration(s)",
            )?;
            validate_registry_count(
                &reader,
                "linkme_KTSTR_TESTS",
                size_of::<KtstrTestEntry>(),
                tests.len(),
                "test entry/entries",
            )?;
            order_tests_by_legacy_registry(&reader, &mut tests)?;
            reconstruct_manifest(declarations, tests).map(Some)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn map_current_test_executable_copy() -> (std::path::PathBuf, memmap2::MmapMut) {
        let executable = std::env::current_exe().expect("locate unit-test executable");
        let file = std::fs::File::open(&executable).expect("open unit-test executable");
        // SAFETY: this is a private, copy-on-write mapping. Mutations in the
        // relocation-order test cannot affect the executable on disk or any
        // concurrently running test process.
        let mapping = unsafe { memmap2::MmapOptions::new().map_copy(&file) }
            .expect("copy-map unit-test executable");
        (executable, mapping)
    }

    #[test]
    fn v1_wire_layout_is_pinned() {
        assert_eq!(size_of::<StampHeaderV1>(), 16);
        assert_eq!(size_of::<SchedulerManifestStampStrV1>(), 16);
        assert_eq!(size_of::<StampOptionalStrV1>(), 24);
        assert_eq!(size_of::<StampOptionalU32V1>(), 8);
        assert_eq!(
            size_of::<SchedulerManifestStampSliceV1<SchedulerManifestStampStrV1>>(),
            16
        );
        assert_eq!(size_of::<SchedulerManifestStampSysctlV1>(), 32);
        assert_eq!(size_of::<SchedulerManifestUseStampV1>(), 56);
        assert_eq!(size_of::<SchedulerManifestDeclarationStampV1>(), 256);
        assert_eq!(size_of::<SchedulerManifestTestStampV1>(), 64);
        assert_eq!(offset_of!(StampHeaderV1, magic), 0);
        assert_eq!(offset_of!(StampHeaderV1, version), 8);
        assert_eq!(offset_of!(StampHeaderV1, kind), 10);
        assert_eq!(offset_of!(StampHeaderV1, record_size), 12);
        assert_eq!(offset_of!(SchedulerManifestStampStrV1, ptr), 0);
        assert_eq!(offset_of!(SchedulerManifestStampStrV1, len), 8);
        assert_eq!(offset_of!(SchedulerManifestStampSliceV1<u8>, ptr), 0);
        assert_eq!(offset_of!(SchedulerManifestStampSliceV1<u8>, len), 8);
        assert_eq!(offset_of!(StampOptionalStrV1, value), 0);
        assert_eq!(offset_of!(StampOptionalStrV1, present), 16);
        assert_eq!(offset_of!(StampOptionalU32V1, value), 0);
        assert_eq!(offset_of!(StampOptionalU32V1, present), 4);
        assert_eq!(offset_of!(SchedulerManifestUseStampV1, name), 0);
        assert_eq!(offset_of!(SchedulerManifestUseStampV1, manifest_dir), 16);
        assert_eq!(offset_of!(SchedulerManifestUseStampV1, binary_value), 32);
        assert_eq!(offset_of!(SchedulerManifestUseStampV1, binary_kind), 48);
        assert_eq!(offset_of!(SchedulerManifestDeclarationStampV1, header), 0);
        assert_eq!(offset_of!(SchedulerManifestDeclarationStampV1, name), 16);
        assert_eq!(
            offset_of!(SchedulerManifestDeclarationStampV1, binary_kind),
            64
        );
        assert_eq!(
            offset_of!(SchedulerManifestDeclarationStampV1, num_numa_nodes),
            72
        );
        assert_eq!(
            offset_of!(SchedulerManifestDeclarationStampV1, sched_args),
            128
        );
        assert_eq!(
            offset_of!(SchedulerManifestDeclarationStampV1, cgroup_parent),
            176
        );
        assert_eq!(
            offset_of!(SchedulerManifestDeclarationStampV1, kernels),
            224
        );
        assert_eq!(offset_of!(SchedulerManifestTestStampV1, header), 0);
        assert_eq!(offset_of!(SchedulerManifestTestStampV1, test), 16);
        assert_eq!(offset_of!(SchedulerManifestTestStampV1, schedulers), 32);
        assert_eq!(
            offset_of!(SchedulerManifestTestStampV1, expected_scheduler_count),
            48
        );
        assert_eq!(offset_of!(SchedulerManifestTestStampV1, legacy_entry), 56);
    }

    #[test]
    fn sparse_reader_leaves_unrelated_elf_tables_unparsed() {
        let (executable, mapping) = map_current_test_executable_copy();
        let reader =
            ElfStampReader::new(&executable, &mapping).expect("sparsely parse test executable");

        assert!(!reader.elf.program_headers.is_empty());
        assert!(!reader.elf.section_headers.is_empty());
        assert_ne!(reader.elf.shdr_strtab.len(), 0);
        assert!(
            reader.elf.syms.is_empty(),
            "the static symbol table must remain lazy"
        );
        assert!(
            reader.elf.strtab.len() == 0,
            "the static symbol string table must remain lazy"
        );
        assert!(
            reader.elf.dynsyms.is_empty(),
            "dynamic symbols are unrelated to relative stamp pointers"
        );
        assert!(
            reader.elf.dynstrtab.len() == 0,
            "dynamic symbol strings are unrelated to stamp pointers"
        );
        assert!(
            reader.elf.pltrelocs.is_empty(),
            "PLT relocations cannot back immutable stamp pointers"
        );
        assert!(
            reader.elf.shdr_relocs.is_empty(),
            "section relocation tables must remain lazy"
        );
    }

    #[test]
    fn stamp_relative_relocations_are_resolved_by_the_sparse_fast_path() {
        let (executable, mapping) = map_current_test_executable_copy();
        let reader =
            ElfStampReader::new(&executable, &mapping).expect("sparsely parse test executable");
        let stamp_ranges = [DECLARATION_SECTION, TEST_SECTION]
            .map(|section| {
                let (address, bytes) = reader
                    .section(section)
                    .expect("find scheduler-manifest stamp section")
                    .expect("scheduler-manifest stamp section must be linked");
                address..address + bytes.len() as u64
            });

        let mut checked = 0usize;
        for (relocations, relative_prefix) in [
            (&reader.elf.dynrelas, reader.dynrela_relative_prefix),
            (&reader.elf.dynrels, reader.dynrel_relative_prefix),
        ] {
            for relocation in relocations.iter().filter(|relocation| {
                relocation.r_type == reader.relative_relocation_type
                    && stamp_ranges
                        .iter()
                        .any(|range| range.contains(&relocation.r_offset))
            }) {
                assert!(
                    reader
                        .relocation_at(relocations, relative_prefix, relocation.r_offset)
                        .is_some(),
                    "relative stamp relocation at 0x{:x} must be found in the ordered prefix",
                    relocation.r_offset,
                );
                checked += 1;
            }
        }
        assert_ne!(
            checked, 0,
            "unit-test executable must carry relative stamp relocations"
        );

        let declared_count = reader
            .elf
            .dynamic
            .as_ref()
            .map(|dynamic| dynamic.info.relacount)
            .unwrap_or_default();
        if declared_count == 0 && !reader.elf.dynrelas.is_empty() {
            assert_ne!(
                reader.dynrela_relative_prefix, 0,
                "count-less mold output must derive its leading relative block"
            );
        }
    }

    #[test]
    fn unsorted_dynamic_relocations_use_sound_exceptional_index() {
        let (executable, mut mapping) = map_current_test_executable_copy();
        let (
            test_record,
            field_va,
            expected_pointer,
            pointer_slot_file_offset,
            relocation_file_offset,
            relocation_size,
            relocation_index,
        ) = {
            let reader = ElfStampReader::new(&executable, &mapping).expect("parse test executable");
            let (test_section, bytes) = reader
                .section(TEST_SECTION)
                .expect("find test stamp section")
                .expect("test stamp section must be linked");
            let record_size = size_of::<SchedulerManifestTestStampV1>();
            let record_index = (0..bytes.len() / record_size)
                .find(|index| {
                    reader
                        .header(
                            test_section + (*index * record_size) as u64,
                            record_size,
                            TEST_SECTION,
                            *index,
                        )
                        .expect("decode test stamp header")
                        == STAMP_KIND_TEST
                })
                .expect("unit-test executable must contain a real ktstr test stamp");
            let test_record = test_section + (record_index * record_size) as u64;
            let field_va = test_record + offset_of!(SchedulerManifestTestStampV1, test) as u64;
            let dynamic = reader
                .elf
                .dynamic
                .as_ref()
                .expect("PIE unit-test executable must have a dynamic table");
            let relocation_index = reader
                .elf
                .dynrelas
                .iter()
                .position(|relocation| relocation.r_offset == field_va)
                .expect("test-name pointer must have a dynamic RELA relocation");
            assert!(
                relocation_index > 0,
                "test-name relocation must leave room for an ordering inversion"
            );
            let relocation = reader
                .elf
                .dynrelas
                .get(relocation_index)
                .expect("indexed relocation");
            assert_eq!(relocation.r_type, reader.relative_relocation_type);
            let expected_pointer = relocation
                .r_addend
                .expect("x86_64/aarch64 dynamic pointer uses RELA")
                as u64;
            let pointer_slot_file_offset = reader
                .va_file_offset(field_va, size_of::<u64>(), "test-name pointer slot")
                .expect("test-name pointer slot must be file-backed");
            let relocation_size = dynamic.info.relaent as usize;
            assert!(relocation_size > 0);
            (
                test_record,
                field_va,
                expected_pointer,
                pointer_slot_file_offset,
                dynamic.info.rela,
                relocation_size,
                relocation_index,
            )
        };

        let target_start = relocation_file_offset + relocation_index * relocation_size;
        for index in 0..relocation_size {
            mapping.swap(relocation_file_offset + index, target_start + index);
        }
        mapping[pointer_slot_file_offset..pointer_slot_file_offset + size_of::<u64>()].fill(0);

        let reader =
            ElfStampReader::new(&executable, &mapping).expect("parse unsorted test executable");
        assert!(
            reader.exceptional_pointer_relocations.get().is_none(),
            "exceptional relocation index must stay lazy"
        );
        assert!(
            reader
                .pointer_relocation(field_va)
                .expect("fast relocation lookup")
                .is_none(),
            "fixture must force the ordered-prefix fast path to miss"
        );
        assert_eq!(
            reader
                .pointer(
                    test_record,
                    offset_of!(SchedulerManifestTestStampV1, test),
                    "unsorted test-name pointer",
                )
                .expect("exceptional index must resolve the pointer"),
            expected_pointer,
        );
        assert!(
            reader.exceptional_pointer_relocations.get().is_some(),
            "binary-search miss must initialize the exceptional index"
        );
    }

    #[test]
    fn reconstructed_requirements_include_direct_scheduler_uses() {
        let scheduler = Scheduler::named("direct")
            .binary(SchedulerSpec::Discover("scx-direct"))
            .manifest_dir("/workspace/direct");
        let parsed = reconstruct_manifest(
            Vec::new(),
            vec![ParsedTestStamp {
                test: "uses_direct".to_string(),
                schedulers: vec![ParsedSchedulerUse {
                    name: scheduler.name.to_string(),
                    manifest_dir: scheduler.manifest_dir.to_string(),
                    binary_kind: BinaryKindJson::Discover("scx-direct".to_string()),
                }],
                legacy_entry: 0,
            }],
        )
        .expect("reconstruct");
        assert!(parsed.declarations.is_empty());
        assert_eq!(
            parsed.tests,
            vec![SchedulerTestJson {
                test: "uses_direct".to_string(),
                scheduler: "direct".to_string(),
            }]
        );
        assert_eq!(
            parsed.artifact_requirements,
            vec![SchedulerArtifactRequirement {
                binary_kind: BinaryKindJson::Discover("scx-direct".to_string()),
                manifest_dir: "/workspace/direct".to_string(),
                schedulers: vec!["direct".to_string()],
                use_count: 1,
            }]
        );
    }
}
