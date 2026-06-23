//! BTF type anchor generation for BPF schedulers.
//!
//! BPF compilers eliminate struct types from BTF when all referencing
//! functions are inlined. This module generates a `-include` header
//! with weak global anchors that force the compiler to retain every
//! struct definition discovered in the scheduler's source tree.
//!
//! The pipeline:
//! 1. Extract `.bpf.c` source paths from BTF string tables in prior
//!    build's `.bpf.o` files (clang embeds absolute paths).
//! 2. Run `clang -M` on each source (in parallel) with the build's
//!    cflags to get the transitive include chain, filtering out
//!    system/kernel headers.
//! 3. Parse every file in the dep list with tree-sitter-c to extract
//!    struct definitions.
//! 4. Generate a header with weak global pointer declarations that
//!    anchor each struct in BTF.
//!
//! The anchor is cached in `target/ktstr_btf_anchor.h` with an ahash
//! of all inputs (ktstr version, source paths, cflags, .bpf.o sizes).
//! Regenerated only when inputs change.

use std::collections::{BTreeSet, HashSet};
use std::hash::{BuildHasher as _, Hash as _, Hasher as _};
use std::path::{Path, PathBuf};
use std::process::Command;

/// Discover sources via BTF strings, run clang -M for deps, extract
/// structs via tree-sitter-c, write the anchor header.
pub(crate) fn generate_btf_anchor(
    bpf_object_dir: &Path,
    clang: &str,
    cflags: &[String],
    anchor_path: &Path,
) -> Option<PathBuf> {
    // Step 1: find .bpf.o files and extract source paths from BTF
    let mut bpf_sources = discover_sources_from_objects(bpf_object_dir);
    if bpf_sources.is_empty() {
        tracing::debug!("btf_anchor: no .bpf.c sources found via BTF");
        return None;
    }
    tracing::debug!(
        sources = bpf_sources.len(),
        "btf_anchor: discovered BPF sources via BTF"
    );

    // Fast path: hash all inputs that affect the anchor output.
    // - ktstr version: pipeline logic changes invalidate
    // - source paths: file set changes
    // - .bpf.o sizes: proxy for source content changes (recompilation
    //   changes object size via BTF/code changes)
    // - cflags: different includes change the dep chain
    bpf_sources.sort();
    let input_hash = {
        let mut h = ahash::RandomState::with_seeds(0x6b74, 0x7374, 0x7200, 0x616e).build_hasher();
        env!("CARGO_PKG_VERSION").hash(&mut h);
        for p in &bpf_sources {
            p.to_string_lossy().hash(&mut h);
        }
        for cflag in cflags {
            cflag.hash(&mut h);
        }
        if let Ok(entries) = std::fs::read_dir(bpf_object_dir) {
            let mut sizes: Vec<(String, u64)> = entries
                .flatten()
                .filter_map(|e| {
                    let name = e.file_name().to_string_lossy().to_string();
                    if name.ends_with(".bpf.o") {
                        e.metadata().ok().map(|m| (name, m.len()))
                    } else {
                        None
                    }
                })
                .collect();
            sizes.sort();
            for (name, size) in &sizes {
                name.hash(&mut h);
                size.hash(&mut h);
            }
        }
        h.finish()
    };
    if let Some(old_hash) = read_anchor_hash(anchor_path)
        && old_hash == input_hash
    {
        tracing::debug!("btf_anchor: cached anchor is current");
        let abs = std::fs::canonicalize(anchor_path).unwrap_or_else(|_| anchor_path.to_path_buf());
        return Some(abs);
    }

    // Step 2: clang -M on each source to get transitive deps
    let dep_files = collect_dep_files(&bpf_sources, clang, cflags);
    if dep_files.is_empty() {
        tracing::debug!("btf_anchor: clang -M produced no dep files");
        return None;
    }
    tracing::debug!(
        files = dep_files.len(),
        "btf_anchor: collected dep files via clang -M"
    );

    // Step 3: tree-sitter-c parse for struct definitions
    let structs = extract_struct_names(&dep_files);
    if structs.is_empty() {
        tracing::debug!("btf_anchor: no struct definitions found");
        return None;
    }
    tracing::debug!(
        structs = structs.len(),
        "btf_anchor: extracted struct definitions"
    );

    if let Some(parent) = anchor_path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    write_anchor_header(anchor_path, &structs, input_hash)?;

    let abs = std::fs::canonicalize(anchor_path).unwrap_or_else(|_| anchor_path.to_path_buf());
    Some(abs)
}

/// Extract .bpf.c source paths from the BTF string table in each
/// .bpf.o file. clang embeds full absolute paths in BTF, making
/// this work regardless of build system.
fn discover_sources_from_objects(dir: &Path) -> Vec<PathBuf> {
    let mut sources: HashSet<PathBuf> = HashSet::new();

    let Ok(entries) = std::fs::read_dir(dir) else {
        return Vec::new();
    };

    for entry in entries.flatten() {
        let path = entry.path();
        let Some(name) = path.file_name().and_then(|n| n.to_str()) else {
            continue;
        };
        if !name.ends_with(".bpf.o") || name == "bpf.bpf.o" {
            continue;
        }

        let Ok(bytes) = std::fs::read(&path) else {
            continue;
        };

        if let Some(btf_data) = find_btf_section_raw(&bytes) {
            for s in btf_strings(btf_data) {
                if s.ends_with(".bpf.c") {
                    let p = PathBuf::from(s);
                    if p.is_file()
                        && let Ok(canonical) = std::fs::canonicalize(&p)
                    {
                        sources.insert(canonical);
                    }
                }
            }
        }
    }

    sources.into_iter().collect()
}

fn find_btf_section_raw(bytes: &[u8]) -> Option<&[u8]> {
    if bytes.len() < 64 {
        return None;
    }
    let e_shoff = u64::from_le_bytes(bytes[40..48].try_into().ok()?) as usize;
    let e_shentsize = u16::from_le_bytes(bytes[58..60].try_into().ok()?) as usize;
    let e_shnum = u16::from_le_bytes(bytes[60..62].try_into().ok()?) as usize;
    let e_shstrndx = u16::from_le_bytes(bytes[62..64].try_into().ok()?) as usize;

    if e_shstrndx >= e_shnum || e_shentsize < 64 {
        return None;
    }

    let strtab_base = e_shoff + e_shstrndx * e_shentsize;
    if strtab_base + 64 > bytes.len() {
        return None;
    }
    let strtab_off =
        u64::from_le_bytes(bytes[strtab_base + 24..strtab_base + 32].try_into().ok()?) as usize;
    let strtab_size =
        u64::from_le_bytes(bytes[strtab_base + 32..strtab_base + 40].try_into().ok()?) as usize;
    if strtab_off + strtab_size > bytes.len() {
        return None;
    }
    let strtab = &bytes[strtab_off..strtab_off + strtab_size];

    for i in 0..e_shnum {
        let base = e_shoff + i * e_shentsize;
        if base + 64 > bytes.len() {
            break;
        }
        let sh_name = u32::from_le_bytes(bytes[base..base + 4].try_into().ok()?) as usize;
        if sh_name + 4 >= strtab.len() {
            continue;
        }
        if &strtab[sh_name..sh_name + 4] != b".BTF" {
            continue;
        }
        if sh_name + 4 < strtab.len() && strtab[sh_name + 4] != 0 {
            continue;
        }
        let sh_offset = u64::from_le_bytes(bytes[base + 24..base + 32].try_into().ok()?) as usize;
        let sh_size = u64::from_le_bytes(bytes[base + 32..base + 40].try_into().ok()?) as usize;
        if sh_offset + sh_size <= bytes.len() && sh_size >= 24 {
            return Some(&bytes[sh_offset..sh_offset + sh_size]);
        }
    }
    None
}

fn btf_strings(btf: &[u8]) -> Vec<&str> {
    if btf.len() < 24 {
        return Vec::new();
    }
    let hdr_len = u32::from_le_bytes([btf[4], btf[5], btf[6], btf[7]]) as usize;
    let str_off = u32::from_le_bytes([btf[16], btf[17], btf[18], btf[19]]) as usize;
    let str_len = u32::from_le_bytes([btf[20], btf[21], btf[22], btf[23]]) as usize;
    let str_start = hdr_len + str_off;
    let str_end = str_start + str_len;
    if str_end > btf.len() {
        return Vec::new();
    }
    let str_section = &btf[str_start..str_end];
    let mut result = Vec::new();
    for chunk in str_section.split(|&b| b == 0) {
        if let Ok(s) = std::str::from_utf8(chunk)
            && !s.is_empty()
        {
            result.push(s);
        }
    }
    result
}

/// Run `clang -M -MG` on each source in parallel and collect every
/// dependency, filtering out system/kernel headers.
fn collect_dep_files(sources: &[PathBuf], clang: &str, cflags: &[String]) -> Vec<PathBuf> {
    let all_deps = std::sync::Mutex::new(HashSet::<PathBuf>::new());

    std::thread::scope(|s| {
        for source in sources {
            let deps_ref = &all_deps;
            s.spawn(move || {
                let output = Command::new(clang)
                    .arg("-M")
                    .arg("-MG")
                    .arg("-target")
                    .arg("bpf")
                    .args(cflags)
                    .arg(source)
                    .output();

                let Ok(output) = output else { return };
                if !output.status.success() {
                    return;
                }

                let mut local = HashSet::new();
                let stdout = String::from_utf8_lossy(&output.stdout);
                let joined = stdout.replace("\\\n", " ");
                for line in joined.lines() {
                    let deps_part = match line.split_once(':') {
                        Some((_, deps)) => deps,
                        None => line,
                    };
                    for token in deps_part.split_whitespace() {
                        let p = PathBuf::from(token);
                        if p.is_file()
                            && let Ok(canonical) = std::fs::canonicalize(&p)
                            && !is_system_header(&canonical)
                        {
                            local.insert(canonical);
                        }
                    }
                }
                deps_ref.lock().unwrap().extend(local);
            });
        }
    });

    all_deps.into_inner().unwrap().into_iter().collect()
}

fn is_system_header(path: &Path) -> bool {
    let s = path.to_string_lossy();
    if s.contains("/usr/include/") || s.contains("/usr/lib/") {
        return true;
    }
    if let Some(name) = path.file_name().and_then(|n| n.to_str())
        && (name == "vmlinux.h" || name == "vmlinux.bpf.h")
    {
        return true;
    }
    if s.contains("scx_utils-bpf_h/") {
        return true;
    }
    false
}

/// Parse C files with tree-sitter-c and extract named struct definitions.
fn extract_struct_names(files: &[PathBuf]) -> BTreeSet<String> {
    let mut names = BTreeSet::new();
    let mut parser = tree_sitter::Parser::new();
    parser
        .set_language(&tree_sitter_c::LANGUAGE.into())
        .expect("tree-sitter-c language");

    for file in files {
        let Ok(content) = std::fs::read_to_string(file) else {
            continue;
        };
        let Some(tree) = parser.parse(&content, None) else {
            continue;
        };
        collect_structs(tree.root_node(), content.as_bytes(), &mut names);
    }
    names
}

fn collect_structs(node: tree_sitter::Node, source: &[u8], names: &mut BTreeSet<String>) {
    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        if child.kind() == "struct_specifier"
            && child.child_by_field_name("body").is_some()
            && let Some(name_node) = child.child_by_field_name("name")
            && let Ok(name) = std::str::from_utf8(&source[name_node.byte_range()])
            && !name.is_empty()
            && !name.starts_with("__")
        {
            names.insert(name.to_string());
        }
        collect_structs(child, source, names);
    }
}

fn read_anchor_hash(path: &Path) -> Option<u64> {
    let content = std::fs::read_to_string(path).ok()?;
    let line = content.lines().find(|l| l.starts_with("/* ktstr_hash="))?;
    let hex = line.strip_prefix("/* ktstr_hash=")?.strip_suffix(" */")?;
    u64::from_str_radix(hex, 16).ok()
}

fn write_anchor_header(path: &Path, structs: &BTreeSet<String>, hash: u64) -> Option<()> {
    let mut src = String::new();
    src.push_str(&format!("/* ktstr_hash={hash:016x} */\n"));
    src.push_str("#ifndef __KTSTR_BTF_ANCHOR_H\n");
    src.push_str("#define __KTSTR_BTF_ANCHOR_H\n");
    for (i, s) in structs.iter().enumerate() {
        src.push_str(&format!(
            "struct {s} __attribute__((weak)) *__ktstr_keep_{i};\n"
        ));
    }
    src.push_str("#endif\n");
    std::fs::write(path, &src).ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- BTF blob builder -------------------------------------------------
    //
    // Mirrors the layout `btf_strings` reads (see uapi struct btf_header):
    //   [0..4]   magic(u16) + version(u8) + flags(u8)  -- unread by btf_strings
    //   [4..8]   hdr_len   (u32 LE)
    //   [8..16]  type_off, type_len                    -- unread by btf_strings
    //   [16..20] str_off   (u32 LE)
    //   [20..24] str_len   (u32 LE)
    // The string section starts at `hdr_len + str_off`.
    fn btf_blob(hdr_len: u32, str_off: u32, str_section: &[u8]) -> Vec<u8> {
        let mut v = vec![0u8; 24];
        // magic/version/flags are not parsed by btf_strings; leave as 0.
        v[4..8].copy_from_slice(&hdr_len.to_le_bytes());
        v[16..20].copy_from_slice(&str_off.to_le_bytes());
        v[20..24].copy_from_slice(&(str_section.len() as u32).to_le_bytes());
        // Pad up to the string-section start, then append the bytes.
        let start = hdr_len as usize + str_off as usize;
        if v.len() < start {
            v.resize(start, 0);
        }
        v.extend_from_slice(str_section);
        v
    }

    // ---- ELF64 fixture builder -------------------------------------------
    //
    // Produces a minimal little-endian ELF64 image with exactly
    // `sections.len()` 64-byte section headers, a section-header string
    // table (".shstrtab"-style) whose contents are the given names, and
    // each section's payload laid out after the header table.
    //
    // `find_btf_section_raw` reads:
    //   e_shoff      [40..48] u64
    //   e_shentsize  [58..60] u16
    //   e_shnum      [60..62] u16
    //   e_shstrndx   [62..64] u16
    // and per 64-byte Shdr entry: sh_name[0..4] u32, sh_offset[24..32] u64,
    // sh_size[32..40] u64.
    const SHENTSIZE: usize = 64;

    struct Sect {
        name: &'static str,
        payload: Vec<u8>,
    }

    /// Build an ELF64 image. `shstrndx` is the index of the section whose
    /// payload is treated as the section-header string table; that
    /// section's payload is auto-generated to hold all section names
    /// (its declared `payload` is ignored).
    fn build_elf(sections: &[Sect], shstrndx: u16) -> Vec<u8> {
        let shnum = sections.len();
        // Header (64 bytes) then the section-header table.
        let ehdr_size = 64usize;
        let e_shoff = ehdr_size;
        let sht_size = shnum * SHENTSIZE;

        // Build the shstrtab payload: a leading NUL, then each name + NUL,
        // recording each name's byte offset.
        let mut shstrtab = vec![0u8]; // index 0 = empty string
        let mut name_off = vec![0u32; shnum];
        for (i, s) in sections.iter().enumerate() {
            name_off[i] = shstrtab.len() as u32;
            shstrtab.extend_from_slice(s.name.as_bytes());
            shstrtab.push(0);
        }

        // Payloads start right after the section-header table. The
        // shstrndx section uses the generated shstrtab; others use their
        // declared payload.
        let payload_base = e_shoff + sht_size;
        let mut payloads: Vec<Vec<u8>> = Vec::with_capacity(shnum);
        for (i, s) in sections.iter().enumerate() {
            if i == shstrndx as usize {
                payloads.push(shstrtab.clone());
            } else {
                payloads.push(s.payload.clone());
            }
        }

        // Compute each payload's file offset (contiguous, in order).
        let mut sh_offsets = vec![0usize; shnum];
        let mut cur = payload_base;
        for i in 0..shnum {
            sh_offsets[i] = cur;
            cur += payloads[i].len();
        }
        let total = cur;

        let mut img = vec![0u8; total];
        // ELF ident magic (cosmetic; find_btf_section_raw does not check it).
        img[0..4].copy_from_slice(b"\x7fELF");
        img[4] = 2; // ELFCLASS64
        img[5] = 1; // ELFDATA2LSB
        img[40..48].copy_from_slice(&(e_shoff as u64).to_le_bytes());
        img[58..60].copy_from_slice(&(SHENTSIZE as u16).to_le_bytes());
        img[60..62].copy_from_slice(&(shnum as u16).to_le_bytes());
        img[62..64].copy_from_slice(&shstrndx.to_le_bytes());

        for i in 0..shnum {
            let base = e_shoff + i * SHENTSIZE;
            img[base..base + 4].copy_from_slice(&name_off[i].to_le_bytes());
            img[base + 24..base + 32].copy_from_slice(&(sh_offsets[i] as u64).to_le_bytes());
            img[base + 32..base + 40].copy_from_slice(&(payloads[i].len() as u64).to_le_bytes());
        }
        for i in 0..shnum {
            let off = sh_offsets[i];
            img[off..off + payloads[i].len()].copy_from_slice(&payloads[i]);
        }
        img
    }

    // ====================== btf_strings ==================================

    /// NUL-delimited string section yields each non-empty UTF-8 chunk in
    /// order; the BTF index-0 empty string is dropped.
    #[test]
    fn btf_strings_splits_on_nul_and_drops_empties() {
        let section = b"\0foo\0bar.bpf.c\0\0baz\0";
        let blob = btf_blob(24, 0, section);
        assert_eq!(btf_strings(&blob), vec!["foo", "bar.bpf.c", "baz"]);
    }

    /// `str_off` is relative to the end of the BTF header (`hdr_len`),
    /// so a non-zero hdr_len shifts the string-section start.
    #[test]
    fn btf_strings_honors_hdr_len_plus_str_off() {
        // hdr_len=32 (8 bytes of post-24 header padding), str_off=4.
        let section = b"\0alpha\0beta\0";
        let blob = btf_blob(32, 4, section);
        assert_eq!(btf_strings(&blob), vec!["alpha", "beta"]);
    }

    /// A blob shorter than the 24-byte header returns no strings rather
    /// than reading out of bounds.
    #[test]
    fn btf_strings_too_short_returns_empty() {
        assert!(btf_strings(&[0u8; 23]).is_empty());
    }

    /// A declared string section that runs past the blob end is rejected
    /// (returns empty), guarding the slice index.
    #[test]
    fn btf_strings_section_past_end_returns_empty() {
        let mut blob = btf_blob(24, 0, b"\0only\0");
        // Inflate the declared str_len so str_end > blob.len().
        let bogus_len = (blob.len() as u32) + 100;
        blob[20..24].copy_from_slice(&bogus_len.to_le_bytes());
        assert!(btf_strings(&blob).is_empty());
    }

    /// Invalid UTF-8 chunks are skipped; valid neighbors still emit.
    #[test]
    fn btf_strings_skips_invalid_utf8_chunk() {
        let section = b"\0good\0\xff\xfe\0also_good\0";
        let blob = btf_blob(24, 0, section);
        assert_eq!(btf_strings(&blob), vec!["good", "also_good"]);
    }

    // =================== find_btf_section_raw ============================

    /// Locates the `.BTF` section by name and returns its exact payload
    /// when sh_size >= 24 and the slice is in bounds.
    #[test]
    fn find_btf_section_raw_returns_btf_payload() {
        let btf_payload: Vec<u8> = (0u8..40).collect(); // size 40 >= 24
        let img = build_elf(
            &[
                Sect {
                    name: "",
                    payload: vec![],
                }, // SHN_UNDEF placeholder
                Sect {
                    name: ".text",
                    payload: vec![0xab; 8],
                },
                Sect {
                    name: ".BTF",
                    payload: btf_payload.clone(),
                },
                Sect {
                    name: ".shstrtab",
                    payload: vec![],
                },
            ],
            3,
        );
        let found = find_btf_section_raw(&img).expect("should find .BTF");
        assert_eq!(found, btf_payload.as_slice());
    }

    /// An ELF shorter than the 64-byte header is rejected outright.
    #[test]
    fn find_btf_section_raw_too_short_is_none() {
        assert!(find_btf_section_raw(&[0u8; 63]).is_none());
    }

    /// No section named `.BTF` -> None even when the image is otherwise
    /// well-formed.
    #[test]
    fn find_btf_section_raw_absent_section_is_none() {
        let img = build_elf(
            &[
                Sect {
                    name: "",
                    payload: vec![],
                },
                Sect {
                    name: ".data",
                    payload: vec![1, 2, 3],
                },
                Sect {
                    name: ".shstrtab",
                    payload: vec![],
                },
            ],
            2,
        );
        assert!(find_btf_section_raw(&img).is_none());
    }

    /// A section whose name is the prefix `.BTF` but continues (e.g.
    /// `.BTF.ext`) must NOT match: the byte after `.BTF` is non-NUL.
    #[test]
    fn find_btf_section_raw_rejects_btf_ext_prefix_collision() {
        let payload: Vec<u8> = (0u8..40).collect();
        let img = build_elf(
            &[
                Sect {
                    name: "",
                    payload: vec![],
                },
                Sect {
                    name: ".BTF.ext",
                    payload,
                },
                Sect {
                    name: ".shstrtab",
                    payload: vec![],
                },
            ],
            2,
        );
        assert!(find_btf_section_raw(&img).is_none());
    }

    /// A `.BTF` section smaller than 24 bytes is rejected (a real BTF
    /// blob always carries at least its 24-byte header).
    #[test]
    fn find_btf_section_raw_rejects_undersized_btf() {
        let img = build_elf(
            &[
                Sect {
                    name: "",
                    payload: vec![],
                },
                Sect {
                    name: ".BTF",
                    payload: vec![0u8; 23],
                }, // 23 < 24
                Sect {
                    name: ".shstrtab",
                    payload: vec![],
                },
            ],
            2,
        );
        assert!(find_btf_section_raw(&img).is_none());
    }

    /// e_shstrndx >= e_shnum is rejected (the string-table index is out
    /// of range).
    #[test]
    fn find_btf_section_raw_rejects_shstrndx_out_of_range() {
        let mut img = build_elf(
            &[
                Sect {
                    name: "",
                    payload: vec![],
                },
                Sect {
                    name: ".BTF",
                    payload: (0u8..40).collect(),
                },
                Sect {
                    name: ".shstrtab",
                    payload: vec![],
                },
            ],
            2,
        );
        // Corrupt e_shstrndx to equal e_shnum (== 3, out of range).
        img[62..64].copy_from_slice(&3u16.to_le_bytes());
        assert!(find_btf_section_raw(&img).is_none());
    }

    /// e_shentsize < 64 is rejected (entries narrower than an Elf64_Shdr).
    #[test]
    fn find_btf_section_raw_rejects_small_shentsize() {
        let mut img = build_elf(
            &[
                Sect {
                    name: "",
                    payload: vec![],
                },
                Sect {
                    name: ".BTF",
                    payload: (0u8..40).collect(),
                },
                Sect {
                    name: ".shstrtab",
                    payload: vec![],
                },
            ],
            2,
        );
        img[58..60].copy_from_slice(&63u16.to_le_bytes());
        assert!(find_btf_section_raw(&img).is_none());
    }

    /// The byte-for-byte returned slice matches a known BTF blob, and
    /// piping it through btf_strings recovers the embedded source path —
    /// the exact two-stage path discover_sources_from_objects relies on.
    #[test]
    fn find_btf_section_raw_then_btf_strings_recovers_source_path() {
        let btf = btf_blob(24, 0, b"\0/abs/sched.bpf.c\0task_struct\0");
        let img = build_elf(
            &[
                Sect {
                    name: "",
                    payload: vec![],
                },
                Sect {
                    name: ".BTF",
                    payload: btf.clone(),
                },
                Sect {
                    name: ".shstrtab",
                    payload: vec![],
                },
            ],
            2,
        );
        let raw = find_btf_section_raw(&img).expect(".BTF present");
        assert_eq!(raw, btf.as_slice());
        assert_eq!(btf_strings(raw), vec!["/abs/sched.bpf.c", "task_struct"],);
    }

    // ======================= is_system_header ===========================

    #[test]
    fn is_system_header_matrix() {
        let cases: &[(&str, bool)] = &[
            ("/usr/include/stdio.h", true),
            ("/usr/lib/clang/18/include/stddef.h", true),
            ("/home/x/.cache/vmlinux.h", true),
            ("/some/dir/vmlinux.bpf.h", true),
            ("/build/scx_utils-bpf_h/bpf/common.bpf.h", true),
            ("/home/x/scheds/scx_simple.bpf.c", false),
            ("/home/x/scheds/intf.h", false),
            // "vmlinux.h" must match only the exact file name, not a path
            // segment substring.
            ("/home/x/vmlinux.h.bak", false),
            // "/usr/include/" must match as a path segment, not a suffix
            // like "myusr/include" without the leading slash boundary.
            ("/opt/myusr/include/foo.h", false),
        ];
        for (p, expected) in cases {
            assert_eq!(
                is_system_header(Path::new(p)),
                *expected,
                "is_system_header({p:?}) should be {expected}",
            );
        }
    }

    // ============= extract_struct_names / collect_structs ===============

    /// Parse real C source: named struct definitions are collected
    /// (sorted, deduped via BTreeSet); forward declarations, anonymous
    /// structs, typedef-only aliases, and `__`-prefixed names are excluded.
    #[test]
    fn extract_struct_names_collects_defined_named_structs() {
        let dir = tempfile::tempdir().expect("tempdir");
        let src = r#"
            struct foo { int a; };              /* defined, named -> kept */
            struct bar;                         /* forward decl -> no body, skip */
            struct { int x; } anon_var;         /* anonymous -> no name, skip */
            struct __internal { int y; };       /* __-prefixed -> skip */
            typedef struct baz { int z; } baz_t;/* named body -> baz kept */
            void use_bar(struct bar *p) {}      /* reference only -> no body */
        "#;
        let file = dir.path().join("types.h");
        std::fs::write(&file, src).expect("write src");

        let got = extract_struct_names(&[file]);
        // BTreeSet -> sorted; only `foo` and `baz` qualify.
        let names: Vec<&str> = got.iter().map(String::as_str).collect();
        assert_eq!(names, vec!["baz", "foo"]);
    }

    /// The same struct name defined in two files is deduplicated.
    #[test]
    fn extract_struct_names_dedups_across_files() {
        let dir = tempfile::tempdir().expect("tempdir");
        let a = dir.path().join("a.h");
        let b = dir.path().join("b.h");
        std::fs::write(&a, "struct shared { int a; };\nstruct only_a { int x; };").unwrap();
        std::fs::write(&b, "struct shared { int a; };\nstruct only_b { int y; };").unwrap();

        let got = extract_struct_names(&[a, b]);
        let names: Vec<&str> = got.iter().map(String::as_str).collect();
        assert_eq!(names, vec!["only_a", "only_b", "shared"]);
    }

    /// A path that cannot be read is skipped silently (the loop uses
    /// let-else `continue`), so a missing file does not poison the run.
    #[test]
    fn extract_struct_names_skips_unreadable_file() {
        let dir = tempfile::tempdir().expect("tempdir");
        let good = dir.path().join("good.h");
        std::fs::write(&good, "struct kept { int a; };").unwrap();
        let missing = dir.path().join("does_not_exist.h");

        let got = extract_struct_names(&[missing, good]);
        let names: Vec<&str> = got.iter().map(String::as_str).collect();
        assert_eq!(names, vec!["kept"]);
    }

    /// Nested struct definitions (a struct defined inside another
    /// struct's body) are collected via the recursive descent.
    #[test]
    fn collect_structs_recurses_into_nested_definitions() {
        let dir = tempfile::tempdir().expect("tempdir");
        let file = dir.path().join("nested.h");
        std::fs::write(&file, "struct outer { struct inner { int a; } i; int b; };").unwrap();
        let got = extract_struct_names(&[file]);
        let names: Vec<&str> = got.iter().map(String::as_str).collect();
        assert_eq!(names, vec!["inner", "outer"]);
    }

    // ====================== anchor hash round-trip ======================

    /// write_anchor_header emits a hash marker that read_anchor_hash
    /// recovers exactly, across the full u64 range (0, mid, MAX).
    #[test]
    fn anchor_hash_round_trips_full_u64_range() {
        let dir = tempfile::tempdir().expect("tempdir");
        let mut structs = BTreeSet::new();
        structs.insert("task_struct".to_string());

        for hash in [0u64, 0x0123_4567_89ab_cdef, u64::MAX] {
            let path = dir.path().join(format!("anchor_{hash:x}.h"));
            write_anchor_header(&path, &structs, hash).expect("write");
            assert_eq!(
                read_anchor_hash(&path),
                Some(hash),
                "round-trip failed for hash {hash:#018x}",
            );
        }
    }

    /// The generated header body declares one weak anchor pointer per
    /// struct, indexed in BTreeSet (sorted) order, wrapped in the include
    /// guard. Pins the exact rendered C text.
    #[test]
    fn write_anchor_header_renders_exact_body() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("anchor.h");
        let mut structs = BTreeSet::new();
        // Insertion order reversed to prove BTreeSet sorts before render.
        structs.insert("zeta".to_string());
        structs.insert("alpha".to_string());

        write_anchor_header(&path, &structs, 0xdead_beef).expect("write");
        let content = std::fs::read_to_string(&path).expect("read back");
        let expected = "\
/* ktstr_hash=00000000deadbeef */
#ifndef __KTSTR_BTF_ANCHOR_H
#define __KTSTR_BTF_ANCHOR_H
struct alpha __attribute__((weak)) *__ktstr_keep_0;
struct zeta __attribute__((weak)) *__ktstr_keep_1;
#endif
";
        assert_eq!(content, expected);
    }

    /// An empty struct set still emits a valid, hash-marked, guarded
    /// header with zero anchor declarations.
    #[test]
    fn write_anchor_header_empty_set_emits_guarded_header() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("empty.h");
        let structs = BTreeSet::new();
        write_anchor_header(&path, &structs, 0x42).expect("write");
        let content = std::fs::read_to_string(&path).expect("read");
        let expected = "\
/* ktstr_hash=0000000000000042 */
#ifndef __KTSTR_BTF_ANCHOR_H
#define __KTSTR_BTF_ANCHOR_H
#endif
";
        assert_eq!(content, expected);
        // And the hash still round-trips out of the marker line.
        assert_eq!(read_anchor_hash(&path), Some(0x42));
    }

    // ======================= read_anchor_hash ===========================

    /// A missing anchor file yields None (the `?` on read_to_string).
    #[test]
    fn read_anchor_hash_missing_file_is_none() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("nope.h");
        assert_eq!(read_anchor_hash(&path), None);
    }

    /// A file with no `/* ktstr_hash= */` marker line yields None.
    #[test]
    fn read_anchor_hash_no_marker_line_is_none() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("plain.h");
        std::fs::write(&path, "#ifndef X\n#define X\n#endif\n").unwrap();
        assert_eq!(read_anchor_hash(&path), None);
    }

    /// A marker line whose hex payload contains a non-hex character
    /// yields None (from_str_radix fails), rather than a garbage value.
    #[test]
    fn read_anchor_hash_non_hex_payload_is_none() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("bad.h");
        std::fs::write(&path, "/* ktstr_hash=00zz0000deadbeef */\n").unwrap();
        assert_eq!(read_anchor_hash(&path), None);
    }

    /// A marker line missing the trailing ` */` suffix yields None
    /// (strip_suffix fails before parsing).
    #[test]
    fn read_anchor_hash_missing_suffix_is_none() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("nosuffix.h");
        std::fs::write(&path, "/* ktstr_hash=00000000deadbeef\n").unwrap();
        assert_eq!(read_anchor_hash(&path), None);
    }
}
