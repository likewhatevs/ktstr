// Pins the SOFT HYPHEN (U+00AD) rejection extension to is_visibly_empty.
// SHY is the most common Cf-category invisible in browser-wrapped
// text (line-break hyphenation hint). A rejection set covering only
// the U+200B..U+200F ZWSP range would leave SHY (U+00AD) outside it,
// so the literal would pass validation.
use ktstr::declare_scheduler;

declare_scheduler!(INVISIBLE_NAME_SHY, {
    name = "\u{00AD}",
    binary = "scx_invisible_name_shy",
});

fn main() {}
