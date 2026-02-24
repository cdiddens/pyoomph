#!/usr/bin/env python3
"""
Test script for ARM64 pyoomph installation
Tests basic functionality to ensure the installation is working correctly
"""

import sys
import traceback

# Import pyoomph at module level
try:
    from pyoomph import *
    from pyoomph.expressions import *
    from pyoomph.expressions.units import meter, second, kilogram
    from pyoomph.equations.poisson import PoissonEquation
    IMPORTS_OK = True
except Exception as e:
    IMPORTS_OK = False
    IMPORT_ERROR = str(e)

# Color codes for output
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'

def print_test(test_name, passed, message=""):
    """Print test result with color"""
    if passed:
        print(f"{GREEN}✓ {test_name}{RESET}")
    else:
        print(f"{RED}✗ {test_name}{RESET}")
        if message:
            print(f"  {YELLOW}{message}{RESET}")

def test_imports():
    """Test basic imports"""
    print(f"\n{BLUE}Testing imports...{RESET}")
    
    if not IMPORTS_OK:
        print_test("Import pyoomph modules", False, IMPORT_ERROR)
        assert False, "Failed to import pyoomph modules"
    
    print_test("Import pyoomph", True)
    print_test("Import pyoomph.expressions", True)
    print_test("Import specific equation module", True)
    
    assert True

def test_simple_poisson():
    """Test solving a simple 1D Poisson problem"""
    print(f"\n{BLUE}Testing 1D Poisson equation...{RESET}")
    
    if not IMPORTS_OK:
        print_test("1D Poisson problem", False, "Imports failed")
        assert False, "Failed to import pyoomph modules"
    
    try:
        
        class SimplePoissonProblem(Problem):
            def define_problem(self):
                # Simple 1D mesh
                mesh = LineMesh(minimum=0, size=1, N=10)
                self.add_mesh(mesh)
                
                # Add Poisson equation: -∇²u = 1
                eqs = PoissonEquation(source=1)
                eqs += DirichletBC(u=0)  # u=0 at boundaries
                self.add_equations(eqs @ "domain")
        
        # Solve the problem
        with SimplePoissonProblem() as problem:
            problem.solve()
            problem.output()
        
        print_test("1D Poisson problem", True)
        assert True
        
    except Exception as e:
        print_test("1D Poisson problem", False, f"{type(e).__name__}: {str(e)}")
        traceback.print_exc()
        assert False, f"Failed to solve 1D Poisson problem: {type(e).__name__}: {str(e)}"

def test_2d_problem():
    """Test a simple 2D problem"""
    print(f"\n{BLUE}Testing 2D problem...{RESET}")
    
    if not IMPORTS_OK:
        print_test("2D rectangular mesh problem", False, "Imports failed")
        assert False, "Failed to import pyoomph modules"
    
    try:
        
        class Simple2DProblem(Problem):
            def define_problem(self):
                # Simple rectangular mesh
                mesh = RectangularQuadMesh(size=[1, 1], N=[5, 5])
                self.add_mesh(mesh)
                
                # Add Poisson equation
                eqs = PoissonEquation()
                eqs += DirichletBC(u=0) @ "left"
                eqs += DirichletBC(u=0) @ "right"
                eqs += DirichletBC(u=0) @ "top"
                eqs += DirichletBC(u=0) @ "bottom"
                self.add_equations(eqs @ "domain")
        
        with Simple2DProblem() as problem:
            problem.solve()
        
        print_test("2D rectangular mesh problem", True)
        assert True
        
    except Exception as e:
        print_test("2D rectangular mesh problem", False, f"{type(e).__name__}: {str(e)}")
        assert False, f"Failed to solve 2D rectangular mesh problem: {type(e).__name__}: {str(e)}"

def test_custom_equation():
    """Test defining a custom equation"""
    print(f"\n{BLUE}Testing custom equation definition...{RESET}")
    
    if not IMPORTS_OK:
        print_test("Custom equation definition", False, "Imports failed")
        assert False, "Failed to import pyoomph modules"
    
    try:
        
        class MyCustomEquation(Equations):
            def define_fields(self):
                self.define_scalar_field("u", "C2")
            
            def define_residuals(self):
                u, u_test = var_and_test("u")
                residual = weak(grad(u), grad(u_test)) + weak(u, u_test)
                self.add_residual(residual)
        
        class CustomProblem(Problem):
            def define_problem(self):
                mesh = LineMesh(size=1, N=5)
                self.add_mesh(mesh)
                
                eqs = MyCustomEquation()
                eqs += DirichletBC(u=0)
                self.add_equations(eqs @ "domain")
        
        with CustomProblem() as problem:
            problem.solve()
        
        print_test("Custom equation definition", True)
        assert True
        
    except Exception as e:
        print_test("Custom equation definition", False, f"{type(e).__name__}: {str(e)}")
        assert False, f"Failed to define custom equation: {type(e).__name__}: {str(e)}"

def test_units():
    """Test units system"""
    print(f"\n{BLUE}Testing units system...{RESET}")
    
    if not IMPORTS_OK:
        print_test("Units system", False, "Imports failed")
        assert False, "Failed to import pyoomph modules"
    
    try:
        
        # Test basic unit operations
        length = 2 * meter
        time = 3 * second
        velocity = length / time
        
        print_test("Units system", True)
        assert True
        
    except Exception as e:
        print_test("Units system", False, f"{type(e).__name__}: {str(e)}")
        assert False, f"Failed to test units system: {type(e).__name__}: {str(e)}"

def test_compiler():
    """Test C compiler availability"""
    print(f"\n{BLUE}Testing C compiler...{RESET}")
    
    try:
        from pyoomph.generic.ccompiler import get_ccompiler
        cc = get_ccompiler()
        print_test(f"C compiler ({type(cc).__name__})", True)
        assert True
        
    except Exception as e:
        print_test("C compiler", False, f"{type(e).__name__}: {str(e)}")
        assert False, f"Failed to test C compiler: {type(e).__name__}: {str(e)}"

def test_solver_availability():
    """Test available solvers"""
    print(f"\n{BLUE}Testing solver availability...{RESET}")
    
    try:
        from pyoomph.generic import Problem
        from pyoomph.solvers.generic import GenericLinearSystemSolver, GenericEigenSolver
        
        p = Problem()
        
        # Test linear solvers
        linear_solvers = ['superlu', 'umfpack', 'pardiso']
        for solver in linear_solvers:
            try:
                GenericLinearSystemSolver.factory_solver(solver, p)
                print_test(f"Linear solver: {solver}", True)
            except:
                print_test(f"Linear solver: {solver}", False, "Not available")
        
        # Test eigen solvers
        eigen_solvers = ['scipy', 'pardiso']
        for solver in eigen_solvers:
            try:
                GenericEigenSolver.factory_solver(solver, p)
                print_test(f"Eigen solver: {solver}", True)
            except:
                print_test(f"Eigen solver: {solver}", False, "Not available")
        
        assert True
        
    except Exception as e:
        print_test("Solver availability check", False, f"{type(e).__name__}: {str(e)}")
        assert False, f"Failed to check solver availability: {type(e).__name__}: {str(e)}"

def main():
    """Run all tests"""
    print(f"{BLUE}{'='*60}{RESET}")
    print(f"{BLUE}pyoomph ARM64 Installation Test{RESET}")
    print(f"{BLUE}{'='*60}{RESET}")
    
    # Track test results
    results = []
    
    # Run tests
    results.append(("Imports", test_imports()))
    results.append(("Simple Poisson", test_simple_poisson()))
    results.append(("2D Problem", test_2d_problem()))
    results.append(("Custom Equation", test_custom_equation()))
    results.append(("Units System", test_units()))
    results.append(("C Compiler", test_compiler()))
    results.append(("Solver Availability", test_solver_availability()))
    
    # Summary
    print(f"\n{BLUE}{'='*60}{RESET}")
    print(f"{BLUE}Test Summary:{RESET}")
    print(f"{BLUE}{'='*60}{RESET}")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = f"{GREEN}PASSED{RESET}" if result else f"{RED}FAILED{RESET}"
        print(f"{test_name:.<40} {status}")
    
    print(f"\n{BLUE}Total: {passed}/{total} tests passed{RESET}")
    
    if passed == total:
        print(f"\n{GREEN}✓ All tests passed! pyoomph is working correctly on ARM64.{RESET}")
        return 0
    else:
        print(f"\n{YELLOW}⚠ Some tests failed. Check the output above for details.{RESET}")
        return 1

if __name__ == "__main__":
    sys.exit(main())