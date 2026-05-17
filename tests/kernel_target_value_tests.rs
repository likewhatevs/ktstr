//! Minimal type-construction tests for [`KernelTarget`] and
//! [`KernelValue`]. The variants are pure data carriers with no
//! behaviour beyond `Clone` / `PartialEq` / `Eq` (the runtime
//! invariants — symbol resolution, KVA translation, per-CPU offset
//! lookup — live in the op handler modules). These tests pin the
//! construction surface so a future serde / wire / resolver refactor
//! that breaks the shape fails at the source instead of at the
//! consumer.

use ktstr::prelude::*;

#[test]
fn kernel_target_symbol_clone_eq() {
    let a = KernelTarget::symbol("jiffies_64");
    let b = KernelTarget::symbol("jiffies_64");
    assert_eq!(a, b);
    assert_eq!(a.clone(), b);

    let other = KernelTarget::symbol("other");
    assert_ne!(a, other);

    // Distinct variants with the same backing u64 are not equal.
    let direct = KernelTarget::direct(0);
    assert_ne!(a, direct);
}

#[test]
fn kernel_target_per_cpu_field_constructor_shape() {
    let t = KernelTarget::per_cpu_field("runqueues", "clock", 3);
    let same = KernelTarget::per_cpu_field("runqueues", "clock", 3);
    assert_eq!(t, same);

    // Same symbol+field on a different CPU is a different target —
    // catches a regression where the constructor drops the cpu
    // arg or collapses variants by symbol alone.
    let different_cpu = KernelTarget::per_cpu_field("runqueues", "clock", 7);
    assert_ne!(t, different_cpu);
}

#[test]
fn kernel_value_variants_distinct() {
    // U32(0), U64(0), Bytes(vec![]) all encode "zero" but the
    // typed-variant discriminant must keep them distinguishable —
    // catches a future flattening regression that would let
    // U32(0) compare equal to U64(0).
    assert_ne!(KernelValue::u32(0), KernelValue::u64(0));
    assert_ne!(KernelValue::u64(0), KernelValue::bytes(Vec::<u8>::new()));
    assert_ne!(KernelValue::u32(0), KernelValue::bytes(Vec::<u8>::new()));
}

#[test]
fn kernel_value_bytes_clone_preserves_payload() {
    let v = KernelValue::bytes(vec![1u8, 2, 3]);
    let cloned = v.clone();
    assert_eq!(v, cloned);

    // Different payload of same length must compare unequal so a
    // future Bytes(Arc<...>) refactor that lost per-byte equality
    // surfaces immediately.
    let different = KernelValue::bytes(vec![1u8, 2, 4]);
    assert_ne!(v, different);
}
