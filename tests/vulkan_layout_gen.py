#!/usr/bin/env python3
"""
tests/vulkan_layout_gen.py -- Generate 'tests/vulkan_layout.cpp'

The Vulkan backend of Dr.Jit-Core ships self-contained declarations of the
subset of the Vulkan and SPIRV-Tools C APIs that it uses ('src/vulkan_api.h').
This script parses these declarations and generates a C++ test that compares
them against the official headers (<vulkan/vulkan.h> and, optionally,
<spirv-tools/libspirv.h>) when those are available at build time:

- size & alignment of all structures and unions, as well as the offset and
  type (pointer-ness, integer/floating point category, size, pointee layout)
  of every field. Bit fields are compared at run time.
- values of all enumerators, and the size of all enumerations.
- all typedefs and handle types.
- the signature (arity, return and parameter types) of all function pointer
  types.
- values of all macros that the header defines when official headers are
  missing (API version helpers, limits, extension names).

When the official headers are passed via --vulkan-header / --spirv-header,
the script additionally verifies that each structure declares exactly the
same fields (in the same order) as its official counterpart, which catches
omitted trailing fields that would otherwise be hidden by padding.

Re-run this script after modifying 'src/vulkan_api.h':

    python3 tests/vulkan_layout_gen.py \
        --vulkan-header /usr/include/vulkan/vulkan_core.h \
        [--spirv-header /path/to/spirv-tools/libspirv.h]
"""

import argparse
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
HEADER = ROOT / 'src' / 'vulkan_api.h'
OUTPUT = ROOT / 'tests' / 'vulkan_layout.cpp'

# Macros that cannot/should not be compared against the official headers
SKIP_MACROS = {
    'VKAPI_ATTR', 'VKAPI_CALL', 'VKAPI_PTR', 'VK_NULL_HANDLE',
    'DRJIT_VK_HANDLE'
}


def strip_comments(text):
    text = re.sub(r'/\*.*?\*/', '', text, flags=re.S)
    return re.sub(r'//[^\n]*', '', text)


def section(text, start, end):
    i = text.index(start)
    j = text.index(end, i)
    return text[i:j]


def parse_records(text):
    """Return [(kind, name, [(field, bits)])] for all struct/union typedefs"""
    records = []
    for m in re.finditer(r'typedef\s+(struct|union)\s+(\w+)\s*\{(.*?)\}\s*\2\s*;',
                         text, flags=re.S):
        fields = []
        for decl in m.group(3).split(';'):
            decl = ' '.join(decl.split())
            if not decl:
                continue
            fm = re.match(r'^(.*?)(\w+)\s*((?:\[[^\]]+\])*)\s*(?::\s*(\d+))?$',
                          decl)
            if not fm:
                raise RuntimeError('Cannot parse field "%s" of %s' %
                                   (decl, m.group(2)))
            fields.append((fm.group(2), fm.group(4)))
        records.append((m.group(1), m.group(2), fields))
    return records


def parse_enums(text):
    """Return [(name, [enumerator])] for all enum typedefs"""
    enums = []
    for m in re.finditer(r'typedef\s+enum\s+(\w+)\s*\{(.*?)\}\s*\1\s*;',
                         text, flags=re.S):
        values = []
        for item in m.group(2).split(','):
            item = item.strip()
            if not item:
                continue
            em = re.match(r'^(\w+)', item)
            values.append(em.group(1))
        enums.append((m.group(1), values))
    return enums


def parse_typedefs(text):
    """Return names of plain typedefs (e.g. 'typedef uint32_t VkFlags;')"""
    names = []
    for m in re.finditer(r'^\s*typedef\s+([^;(){}]*?[\s\*])(\w+)\s*;', text,
                         flags=re.M):
        if re.match(r'\s*(struct|union|enum)\b', m.group(1)):
            continue
        names.append(m.group(2))
    return names


def parse_pfns(text):
    return re.findall(r'\(\s*(?:VKAPI_PTR\s*)?\*\s*(PFN_\w+)\s*\)', text)


def parse_macros(text):
    """Return [(name, params or None, body)] for all #define directives"""
    text = text.replace('\\\n', ' ')
    macros = []
    for m in re.finditer(r'^\s*#\s*define\s+(\w+)(\([^)]*\))?[ \t]*(.*)$',
                         text, flags=re.M):
        name, params, body = m.group(1), m.group(2), m.group(3).strip()
        if name in SKIP_MACROS or not body:
            continue
        macros.append((name, params, body))
    return macros


def check_fields(ours, official_text, label):
    """Compare the field lists of our records with the official headers"""
    official = {name: fields for _, name, fields in parse_records(
        strip_comments(official_text))}
    errors = 0
    for _, name, fields in ours:
        if name not in official:
            print('error: %s "%s" not found in the official %s header' %
                  ('record', name, label), file=sys.stderr)
            errors += 1
            continue
        a = [f for f, _ in fields]
        b = [f for f, _ in official[name]]
        if a != b:
            print('error: fields of "%s" differ from the official %s header:\n'
                  '  ours:     %s\n  official: %s' % (name, label, a, b),
                  file=sys.stderr)
            errors += 1
    return errors


def emit_checks(out, records, enums, typedefs, pfns, spv):
    """Emit static assertions, return (count, runtime bit field checks)"""
    n = 0
    runtime = []

    def sa(cond, msg):
        nonlocal n
        n += 1
        out.append('static_assert(%s, "%s");' % (cond, msg))

    for name in typedefs:
        sa('check::same_shape<ours::%s, ::%s>()' % (name, name),
           'typedef %s' % name)

    for name, values in enums:
        sa('sizeof(ours::%s) == sizeof(::%s)' % (name, name),
           'sizeof(%s)' % name)
        for v in values:
            sa('(long long) ours::%s == (long long) ::%s' % (v, v), v)

    for kind, name, fields in records:
        sa('sizeof(ours::%s) == sizeof(::%s)' % (name, name),
           'sizeof(%s)' % name)
        sa('alignof(ours::%s) == alignof(::%s)' % (name, name),
           'alignof(%s)' % name)
        bitfields = []
        for field, bits in fields:
            if bits:
                bitfields.append((field, int(bits)))
                continue
            sa('offsetof(ours::%s, %s) == offsetof(::%s, %s)' %
               (name, field, name, field), '%s::%s: offset' % (name, field))
            sa('check::same_shape<decltype(ours::%s::%s), decltype(::%s::%s)>()'
               % (name, field, name, field), '%s::%s: type' % (name, field))
        if bitfields:
            runtime.append((name, bitfields))

    for pfn in pfns:
        if spv:
            ref = 'decltype(&::%s)' % pfn[4:]
        else:
            ref = '::%s' % pfn
        sa('check::same_shape<ours::%s, %s>()' % (pfn, ref), pfn)

    return n, runtime


def emit_runtime_bitfield_checks(out, runtime):
    for i, (name, bitfields) in enumerate(runtime):
        out.append('    {')
        out.append('        ours::%s a; ::%s b;' % (name, name))
        out.append('        memset((void *) &a, 0, sizeof(a));')
        out.append('        memset((void *) &b, 0, sizeof(b));')
        for j, (field, bits) in enumerate(bitfields):
            value = (0x5A3C96E1 >> (j * 3)) & ((1 << bits) - 1)
            out.append('        a.%s = b.%s = 0x%x;' % (field, field, value))
        out.append('        CHECK(memcmp((const void *) &a, (const void *) &b, '
                   'sizeof(a)) == 0, "%s: bit fields");' % name)
        out.append('    }')


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    parser.add_argument('--vulkan-header', help='official vulkan_core.h (for '
                        'checking that all fields are declared)')
    parser.add_argument('--spirv-header', help='official libspirv.h (for '
                        'checking that all fields are declared)')
    args = parser.parse_args()

    raw = HEADER.read_text()
    vk = strip_comments(section(raw, '#else', '#endif // DRJIT_DYNAMIC_VULKAN'))
    spv = strip_comments(section(raw, 'typedef struct spv_context_t',
                                 '#define DRJIT_VK_GLOBAL_FUNCS'))

    vk_records, spv_records = parse_records(vk), parse_records(spv)
    errors = 0
    if args.vulkan_header:
        errors += check_fields(vk_records,
                               pathlib.Path(args.vulkan_header).read_text(),
                               'Vulkan')
    if args.spirv_header:
        errors += check_fields(spv_records,
                               pathlib.Path(args.spirv_header).read_text(),
                               'SPIRV-Tools')
    if errors:
        sys.exit(1)

    vk_handles = re.findall(r'^DRJIT_VK_HANDLE\((\w+)\)', vk, flags=re.M)
    vk_out, spv_out = [], []
    n_vk, rt_vk = emit_checks(vk_out, vk_records, parse_enums(vk),
                              vk_handles + parse_typedefs(vk), parse_pfns(vk),
                              False)
    n_spv, rt_spv = emit_checks(spv_out, spv_records, parse_enums(spv),
                                parse_typedefs(spv), parse_pfns(spv), True)
    assert not rt_spv

    macros = parse_macros(vk)
    macro_names = {name for name, _, _ in macros}

    def rename(body):
        return re.sub(r'\b(\w+)\b', lambda m: 'DRJIT_OURS_' + m.group(1)
                      if m.group(1) in macro_names else m.group(1), body)

    macro_defs, macro_checks, macro_runtime = [], [], []
    for name, params, body in macros:
        macro_defs.append('#define DRJIT_OURS_%s%s %s' %
                          (name, params or '', rename(body)))
        if params:
            arity = len(params.strip('()').split(','))
            samples = {
                1: ['0u', '0x00403123u', '0xFFFFFFFFu'],
                4: ['0, 1, 2, 3', '0, 1, 3, 0', '7, 127, 1023, 4095']
            }[arity]
            for s in samples:
                macro_checks.append(
                    'static_assert(DRJIT_OURS_%s(%s) == %s(%s), "%s(%s)");' %
                    (name, s, name, s, name, s))
        elif body.startswith('"'):
            macro_runtime.append(
                '    CHECK(strcmp(DRJIT_OURS_%s, %s) == 0, "%s");' %
                (name, name, name))
        else:
            macro_checks.append(
                'static_assert((DRJIT_OURS_%s) == (%s), "%s");' %
                (name, name, name))

    n_vk += len(macro_checks)

    out = []
    out.append('''/*
    tests/vulkan_layout.cpp -- Check the self-contained Vulkan and SPIRV-Tools
    declarations in 'src/vulkan_api.h' against the official headers

    GENERATED by 'tests/vulkan_layout_gen.py' -- do not edit! Re-run the script
    after modifying 'src/vulkan_api.h'.

    Copyright (c) 2026 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#define VK_NO_PROTOTYPES
#include <vulkan/vulkan.h>
#if defined(DRJIT_CHECK_SPIRV_TOOLS)
#  include <spirv-tools/libspirv.h>
#endif
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string_view>
#include <tuple>
#include <type_traits>

#if defined(VK_API_VERSION_1_3) // Official headers must cover Vulkan 1.3

// Dr.Jit's declarations (compiled in self-contained mode) in a namespace
namespace ours {
#define DRJIT_DYNAMIC_VULKAN 1
#include "vulkan_api.h"
}

namespace check {
    /// Is 'T' a complete type?
    template <typename T, typename = void> struct is_complete : std::false_type { };
    template <typename T>
    struct is_complete<T, std::void_t<decltype(sizeof(T))>> : std::true_type { };

    /// Name of a type without the 'ours::' prefix (empty if unsupported)
    template <typename T> constexpr std::string_view type_name() {
    #if defined(__GNUC__) || defined(__clang__)
        std::string_view s = __PRETTY_FUNCTION__;
        size_t start = s.find("T = ") + 4, end = s.find_first_of(";]", start);
        s = s.substr(start, end - start);
        if (s.substr(0, 6) == "ours::")
            s.remove_prefix(6);
        return s;
    #else
        return std::string_view();
    #endif
    }

    template <typename A, typename B> constexpr bool same_shape();

    template <typename... A1, typename... A2>
    constexpr bool same_args(std::tuple<A1...> *, std::tuple<A2...> *) {
        if constexpr (sizeof...(A1) != sizeof...(A2))
            return false;
        else
            return (true && ... && same_shape<A1, A2>());
    }

    /// Do two function pointer types have compatible signatures?
    template <typename A, typename B> struct same_sig : std::false_type { };
    template <typename R1, typename... A1, typename R2, typename... A2>
    struct same_sig<R1 (VKAPI_PTR *)(A1...), R2 (VKAPI_PTR *)(A2...)> {
        static constexpr bool value =
            same_shape<R1, R2>() &&
            same_args((std::tuple<A1...> *) nullptr,
                      (std::tuple<A2...> *) nullptr);
    };

    /**
     * Are 'A' and 'B' layout-compatible? Handles and structures of the two
     * declarations are different C++ types, hence this structural check:
     * pointers must match in constness and pointee type (only by name if the
     * pointee is opaque), scalars in category, signedness, and size, arrays in
     * extent and element type, and aggregates/enumerations in name, size, and
     * alignment.
     */
    template <typename A, typename B> constexpr bool same_shape() {
        using UA = std::remove_cv_t<A>;
        using UB = std::remove_cv_t<B>;
        if constexpr (std::is_pointer_v<UA> != std::is_pointer_v<UB>) {
            return false;
        } else if constexpr (std::is_pointer_v<UA>) {
            using PA = std::remove_pointer_t<UA>;
            using PB = std::remove_pointer_t<UB>;
            if constexpr (std::is_const_v<PA> != std::is_const_v<PB>)
                return false;
            else if constexpr (std::is_function_v<PA> || std::is_function_v<PB>)
                return std::is_function_v<PA> && std::is_function_v<PB> &&
                       same_sig<UA, UB>::value;
            else if constexpr (std::is_void_v<PA> || std::is_void_v<PB>)
                return std::is_void_v<PA> && std::is_void_v<PB>;
            else if constexpr (is_complete<PA>::value && is_complete<PB>::value)
                return same_shape<PA, PB>();
            else if constexpr (std::is_class_v<PA> && std::is_class_v<PB>)
                return type_name<std::remove_cv_t<PA>>() ==
                       type_name<std::remove_cv_t<PB>>(); // opaque pointee
            else
                return false;
        } else if constexpr (std::is_array_v<UA> || std::is_array_v<UB>) {
            if constexpr (std::is_array_v<UA> && std::is_array_v<UB>)
                return std::extent_v<UA> == std::extent_v<UB> &&
                       same_shape<std::remove_extent_t<UA>,
                                  std::remove_extent_t<UB>>();
            else
                return false;
        } else if constexpr (std::is_void_v<UA> || std::is_void_v<UB>) {
            return std::is_void_v<UA> && std::is_void_v<UB>;
        } else if constexpr (std::is_floating_point_v<UA> !=
                             std::is_floating_point_v<UB>) {
            return false;
        } else if constexpr ((std::is_integral_v<UA> || std::is_enum_v<UA>) !=
                             (std::is_integral_v<UB> || std::is_enum_v<UB>)) {
            return false;
        } else if constexpr (std::is_integral_v<UA> && std::is_integral_v<UB>) {
            return sizeof(UA) == sizeof(UB) &&
                   std::is_signed_v<UA> == std::is_signed_v<UB>;
        } else {
            // Aggregates and enumerations must also have the same name
            constexpr bool nominal =
                (std::is_class_v<UA> || std::is_union_v<UA> || std::is_enum_v<UA>) &&
                (std::is_class_v<UB> || std::is_union_v<UB> || std::is_enum_v<UB>);
            return sizeof(UA) == sizeof(UB) && alignof(UA) == alignof(UB) &&
                   (!nominal || type_name<UA>() == type_name<UB>());
        }
    }
}

// ----------------------------------------------------------------------
//              Vulkan declarations vs. <vulkan/vulkan.h>
// ----------------------------------------------------------------------
''')
    out += vk_out
    out.append('')
    out.append('// Macros that \'vulkan_api.h\' defines if the official headers '
               'are missing')
    out += macro_defs
    out += macro_checks
    out.append('''
#if defined(DRJIT_CHECK_SPIRV_TOOLS)
// ----------------------------------------------------------------------
//          SPIRV-Tools declarations vs. <spirv-tools/libspirv.h>
// ----------------------------------------------------------------------
''')
    out += spv_out
    out.append('#endif // DRJIT_CHECK_SPIRV_TOOLS')
    out.append('''
static int checks_run = 0, checks_failed = 0;

#define CHECK(cond, msg)                                                       \\
    do {                                                                       \\
        checks_run++;                                                          \\
        if (!(cond)) {                                                         \\
            fprintf(stderr, "Check failed: %s\\n", msg);                        \\
            checks_failed++;                                                   \\
        }                                                                      \\
    } while (0)

#endif // defined(VK_API_VERSION_1_3)

int main() {
#if !defined(VK_API_VERSION_1_3)
    printf("test_vulkan_layout: skipped (the official Vulkan headers are too "
           "old, version 1.3 or newer is required).\\n");
    return 0;
#else''')
    emit_runtime_bitfield_checks(out, rt_vk)
    out += macro_runtime
    out.append('''
#if defined(DRJIT_CHECK_SPIRV_TOOLS)
    const int n_static = %d + %d;
    const char *what = "Vulkan and SPIRV-Tools";
#else
    const int n_static = %d;
    const char *what = "Vulkan";
#endif

    if (checks_failed) {
        fprintf(stderr, "test_vulkan_layout: %%i/%%i runtime checks failed!\\n",
                checks_failed, checks_run);
        return 1;
    }

    printf("test_vulkan_layout: %%s declarations match the official headers "
           "(%%i static and %%i runtime checks passed).\\n", what, n_static,
           checks_run);
    return 0;
#endif
}''' % (n_vk, n_spv, n_vk))

    OUTPUT.write_text('\n'.join(out) + '\n')
    print('Wrote %s (%d Vulkan and %d SPIRV-Tools static checks, %d records, '
          '%d macros).' % (OUTPUT.relative_to(ROOT), n_vk, n_spv,
                           len(vk_records) + len(spv_records), len(macros)))


if __name__ == '__main__':
    main()
