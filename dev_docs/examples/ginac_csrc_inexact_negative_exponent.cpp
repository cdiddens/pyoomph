// GiNaC: print_csrc silently drops the reciprocal of a factor whose exponent is an inexact -1, i.e.
// numeric(-1.0) rather than numeric(-1). The generated C source then computes something different
// from the expression it was generated from, with no diagnostic.
//
// Seen on 1.8.7 (Ubuntu libginac-dev, shared, against libcln.so.6) and on 1.8.10 built from source.
// The output of this program is the same for both bar one line: GiNaC's canonical factor order puts
// the affected factor first in the product in one and second in the other, so between the two runs
// both the "1.0/" prefix path and the "/" separator path of mul::do_print_csrc are covered, and the
// reciprocal is lost either way.
//
// GiNaC compares numbers by value, so numeric(-1) and numeric(-1.0) are is_equal and hash alike;
// the two expressions below are one and the same expression as far as every computation is
// concerned. mul::do_print_csrc, however, asks two different questions about the exponent and gets
// inconsistent answers (ginac/mul.cpp, do_print_csrc):
//
//     // If the first argument is a negative integer power, it gets printed as "1.0/<expr>"
//     if (it == seq.begin() && it->coeff.info(info_flags::negint)) { ... c.s << "1.0/"; }
//
//     // If the exponent is 1 or -1, it is left out
//     if (it->coeff.is_equal(_ex1) || it->coeff.is_equal(_ex_1))
//         it->rest.print(c, precedence());
//     ...
//     // Separator is "/" for negative integer powers, "*" otherwise
//     if (it->coeff.info(info_flags::negint)) c.s << "/"; else c.s << "*";
//
// numeric::info(info_flags::negint) requires a CLN integer and is therefore false for -1.0, while
// is_equal(_ex_1) compares by value and is therefore true. So neither the "1.0/" prefix nor the "/"
// separator is emitted, and the exponent is left out all the same: x^(-1.0) inside a product is
// printed as a plain multiplication by x.
//
// power::do_print_csrc is not affected: it tests only exponent.is_equal(_ex_1) and prints
// "1.0/(...)" for either representation - as the standalone case below shows. That also suggests the
// smallest fix, in mul::do_print_csrc:
//
//     if (it->coeff.is_equal(_ex1) || (it->coeff.is_equal(_ex_1) && it->coeff.info(info_flags::negint)))
//
// which lets an inexact -1 fall through to the generic ex(power(rest, coeff)).print(...) branch,
// where power::do_print_csrc already gets it right.
//
// print_csrc_float and print_csrc_cl_N go through the same mul::do_print_csrc and are affected
// identically (the cl_N context loses its recip() for the same reason).
//
// Such an exponent is easy to end up with without meaning to: any pow(x, numeric(d)) with a double
// d does it. We met it through a cache of our own, keyed - reasonably enough, we thought - on
// ex::gethash() and confirmed with ex::is_equal, which cannot tell an exact -1 from an inexact one
// and so handed the exponent of a reciprocal the representation of an unrelated float coefficient.
// Nothing downstream noticed until the emitted C computed the wrong thing.
//
// Build and run:
//     g++ -o csrc_bug csrc_inexact_negative_exponent.cpp -lginac -lcln
//     ./csrc_bug

#include <ginac/ginac.h>
#include <iostream>
#include <sstream>
#include <string>

using namespace GiNaC;

template <typename Context>
static std::string csrc(const ex &e)
{
	std::ostringstream os;
	e.print(Context(os));
	return os.str();
}

// `as_printed` is what the emitted C source literally says, written out as an expression again, so
// that the two can be evaluated side by side. It is 0 where the output is correct.
static void show(const std::string &what, const ex &e, const symbol &x, const ex &as_printed = 0)
{
	std::cout << "  " << what << "\n"
			  << "      expression      : " << e << "\n"
			  << "      C source        : " << csrc<print_csrc_double>(e) << "\n"
			  << "      expression @x=2 : " << e.subs(x == 2).evalf() << "\n";
	if (!as_printed.is_zero())
		std::cout << "      that C   @x=2   : " << as_printed.subs(x == 2).evalf() << "   <-- WRONG\n";
	std::cout << "\n";
}

int main()
{
	symbol x("x");

	std::cout << "GiNaC " << GINACLIB_MAJOR_VERSION << "." << GINACLIB_MINOR_VERSION
			  << "." << GINACLIB_MICRO_VERSION << "\n\n";

	const ex exact = 3 * pow(1 + x, -1);              // exponent is a CLN integer
	const ex inexact = 3 * pow(1 + x, numeric(-1.0)); // exponent is a CLN float

	const ex minus_one = numeric(-1), minus_one_point_zero = numeric(-1.0);
	std::cout << "The two exponents, and hence the two expressions, are equal to GiNaC:\n"
			  << "  -1 is_equal -1.0             = " << minus_one.is_equal(minus_one_point_zero) << "\n"
			  << "  equal hashes                 = "
			  << (minus_one.gethash() == minus_one_point_zero.gethash()) << "\n"
			  << "  (exact - inexact).is_zero()  = " << (exact - inexact).is_zero() << "\n"
			  << "  info(negint) of -1 and -1.0  = " << minus_one.info(info_flags::negint) << " and "
			  << minus_one_point_zero.info(info_flags::negint) << "    <-- the inconsistency\n\n";

	std::cout << "They are not printed as the same C code:\n\n";
	show("3*(1+x)^(-1)     correct", exact, x);
	show("3*(1+x)^(-1.0)   the reciprocal is gone", inexact, x, 3 * (1 + x));

	std::cout << "Only an exponent of exactly -1 is affected, and only inside a product:\n\n";
	show("(1+x)^(-1.0)     correct - power::do_print_csrc handles it", pow(1 + x, numeric(-1.0)), x);
	show("3*(1+x)^(-2.0)   correct - falls through to pow()", 3 * pow(1 + x, numeric(-2.0)), x);
	show("3*(1+x)^(-1/2)   correct - not an integer either way", 3 * pow(1 + x, numeric(-1, 2)), x);
	show("x*(1+x)^(-1.0)   the reciprocal is gone", x * pow(1 + x, numeric(-1.0)), x, x * (1 + x));

	std::cout << "The other C source contexts print it the same way:\n"
			  << "  print_csrc_double : " << csrc<print_csrc_double>(inexact) << "\n"
			  << "  print_csrc_float  : " << csrc<print_csrc_float>(inexact) << "\n"
			  << "  print_csrc_cl_N   : " << csrc<print_csrc_cl_N>(inexact) << "\n"
			  << "for comparison, the same expression with an exact exponent:\n"
			  << "  print_csrc_double : " << csrc<print_csrc_double>(exact) << "\n"
			  << "  print_csrc_float  : " << csrc<print_csrc_float>(exact) << "\n"
			  << "  print_csrc_cl_N   : " << csrc<print_csrc_cl_N>(exact) << "\n\n";

	// Two expressions that are is_equal have to print as the same C source; the labels above
	// describe stock GiNaC and no longer apply once the mul.cpp line quoted at the top is fixed.
	const bool present = csrc<print_csrc_double>(exact) != csrc<print_csrc_double>(inexact);
	std::cout << (present ? "BUG PRESENT: equal expressions print as different C source\n"
						  : "ok: equal expressions print as the same C source\n");
	return present ? 1 : 0;
}
