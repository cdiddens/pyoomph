// Does the GiNaC we are about to link actually order terms the same way in every process?
//
// pyoomph's JIT code cache is content-addressed: identical generated C implies it is safe to
// reuse a compiled library. That premise dies if GiNaC's canonical ordering moves between runs,
// which is what citools/patches/ginac-deterministic-*.patch exist to prevent - unpatched, GiNaC
// derives hash seeds from the ADDRESS of an RTTI type-name string, and every std::set/std::map
// keyed on GiNaC::ex_is_less that pyoomph's code generator builds then orders by ASLR.
//
// Until now nothing checked. PYOOMPH_ASSUME_GINAC_HASH_PATCHED asserted the property, CMake
// believed the assertion, and the runtime cache trusted CMake - so a prebuilt GiNaC that predated
// the patches produced reordered code, wheels shipped with the cache live over it, and the only
// symptom was four tests failing on one platform.
//
// This prints the two things the patches actually change. The caller runs it several times, in
// separate processes, and compares: same output every time means the ordering is stable across
// exec boundaries, which is precisely the property the cache needs.
//
//   * Pi.gethash() - constant::calchash() had its own inline copy of the address-derived seed.
//   * the printed form of a sum mixing several expression TYPES - make_hash_seed() gives each
//     class its seed, so an unpatched build reorders the terms of this sum from run to run.
//
// Exit code is not the signal; stdout is. A build that cannot run this (cross-compilation) falls
// back to the assertion, loudly.
#include <ginac/ginac.h>

#include <iostream>

int main() {
    GiNaC::symbol a("a"), b("b");
    // Deliberately heterogeneous: a numeric, a constant, a power, a function and a product, so the
    // sum's canonical order depends on the per-class hash seeds rather than on symbol serials
    // (which were already deterministic - symbol::serial is a counter, not an address).
    GiNaC::ex mixed = GiNaC::Pi + 2 * a + GiNaC::pow(b, 3) + GiNaC::sin(a) + a * b + GiNaC::Euler;
    std::cout << "pi_hash=" << GiNaC::Pi.gethash() << "\n";
    std::cout << "euler_hash=" << GiNaC::Euler.gethash() << "\n";
    std::cout << "mixed=" << mixed << "\n";
    return 0;
}
