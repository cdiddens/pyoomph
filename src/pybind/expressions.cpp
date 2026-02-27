/*================================================================================
pyoomph - a multi-physics finite element framework based on oomph-lib and GiNaC 
Copyright (C) 2021-2025  Christian Diddens & Duarte Rocha

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>. 

The authors may be contacted at c.diddens@utwente.nl and d.rocha@utwente.nl

================================================================================*/

#include <pybind11/pybind11.h>
#include <pybind11/operators.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/functional.h>
#include <pybind11/complex.h>

namespace py = pybind11;

#include <sstream>

#include "../codegen.hpp"
#include "../ccompiler.hpp"
#include "../expressions.hpp"
#include "../exception.hpp"
#include "../problem.hpp"

namespace pyoomph
{

	class CustomMathExpression : public CustomMathExpressionBase
	{
	protected:
		py::array_t<double> argbuffer;
		py::buffer_info argbuff;
		unsigned int lastsize = 1;

	public:
		CustomMathExpression() : CustomMathExpressionBase(), argbuffer(1)
		{
			argbuff = argbuffer.request();
		}
		virtual double eval(py::array_t<double> &args) { return 0; }

		virtual double _call(double *args, unsigned int nargs)
		{
			if (argbuff.shape[0] != nargs)
			{
				argbuffer.resize({nargs});
				argbuff = argbuffer.request();
			}
			for (unsigned int i = 0; i < nargs; i++)
				((double *)(argbuff.ptr))[i] = args[i];
			return this->eval(argbuffer);
		}
	};

	class PyCustomMathExpression : public CustomMathExpression
	{
	public:
		using CustomMathExpression::CustomMathExpression;

		double eval(py::array_t<double> &args) override
		{
			PYBIND11_OVERLOAD_PURE(double, CustomMathExpression, eval, args);
		}

		GiNaC::ex outer_derivative(const GiNaC::ex arglist, int index) override
		{
			PYBIND11_OVERLOAD(GiNaC::ex, CustomMathExpression, outer_derivative, arglist, index);
		}

		std::string get_id_name() override
		{
			PYBIND11_OVERLOAD(std::string, CustomMathExpression, get_id_name);
		}

		GiNaC::ex get_argument_unit(unsigned int i) override
		{
			PYBIND11_OVERLOAD(GiNaC::ex, CustomMathExpression, get_argument_unit, i);
		}
		GiNaC::ex get_result_unit() override
		{
			PYBIND11_OVERLOAD(GiNaC::ex, CustomMathExpression, get_result_unit);
		}

		GiNaC::ex real_part(GiNaC::ex invok, std::vector<GiNaC::ex> arglist) override
		{
			PYBIND11_OVERLOAD(GiNaC::ex, CustomMathExpression, real_part, invok, arglist);
		}
		GiNaC::ex imag_part(GiNaC::ex invok, std::vector<GiNaC::ex> arglist) override
		{
			PYBIND11_OVERLOAD(GiNaC::ex, CustomMathExpression, imag_part, invok, arglist);
		}
	};

	class CustomMultiReturnExpression : public CustomMultiReturnExpressionBase
	{
	protected:
		py::array_t<double> argbuffer;
		py::buffer_info argbuff;
		py::array_t<double> resbuffer;
		py::buffer_info resbuff;
		py::array_t<double> derivbuffer;
		py::buffer_info derivbuff;

	public:
		CustomMultiReturnExpression() : CustomMultiReturnExpressionBase(), argbuffer(1), resbuffer(1), derivbuffer({1, 1})
		{
			argbuff = argbuffer.request();
			resbuff = resbuffer.request();
			derivbuff = derivbuffer.request();
		}
		virtual void eval(int flag, py::array_t<double> &args, py::array_t<double> &result, py::array_t<double> &derivs) {}

		virtual void _debug_c_code_call(int flag, double *args, unsigned int nargs, double *res, unsigned int nres, double *derivs)
		{
			if (argbuff.shape[0] != nargs)
			{
				argbuffer.resize({nargs});
				argbuff = argbuffer.request();
			}
			if (resbuff.shape[0] != nres)
			{
				resbuffer.resize({nres});
				resbuff = resbuffer.request();
			}
			if (flag && (derivbuff.shape[0] != nres || derivbuff.shape[1] != nargs))
			{
				derivbuffer.resize({nres, nargs});
				derivbuff = derivbuffer.request();
			}
			for (unsigned int i = 0; i < nargs; i++)
				((double *)(argbuff.ptr))[i] = args[i];
			this->eval(flag, argbuffer, resbuffer, derivbuffer);
			for (unsigned int i = 0; i < nres; i++)
			{
				double resi = ((double *)(resbuff.ptr))[i];
				double diff = resi - res[i];
				if (std::fabs(diff) > this->debug_c_code_epsilon)
				{
					std::cout << "MULTI-RET Python Vs C difference (flag=" << flag << "):  Result " << i << " is " << resi << " (Python) and " << res[i] << " (C) at arguments: ";
					for (unsigned int ia = 0; ia < nargs; ia++)
						std::cout << args[ia] << (ia + 1 < nargs ? "," : "");
					std::cout << std::endl;
				}
			}
			if (flag)
			{
				for (unsigned int i = 0; i < nres; i++)
					for (unsigned int j = 0; j < nargs; j++)
					{
						double resi = ((double *)(derivbuff.ptr))[i * nargs + j];
						double diff = resi - derivs[i * nargs + j];
						if (std::fabs(diff) > this->debug_c_code_epsilon)
						{
							std::cout << "MULTI-RET Python Vs C difference (flag=" << flag << "): dResult " << i << "/dArg" << j << " is " << resi << " (Python) and " << derivs[i * nargs + j] << " (C) at arguments: ";
							for (unsigned int ia = 0; ia < nargs; ia++)
								std::cout << args[ia] << (ia + 1 < nargs ? "," : "");
							std::cout << std::endl;
						}
					}
			}
		}

		virtual void _call(int flag, double *args, unsigned int nargs, double *res, unsigned int nres, double *derivs)
		{
			if (flag & 128)
			{
				flag &= ~(128);
				_debug_c_code_call(flag, args, nargs, res, nres, derivs);
				return;
			}
			if (argbuff.shape[0] != nargs)
			{
				argbuffer.resize({nargs});
				argbuff = argbuffer.request();
			}
			if (resbuff.shape[0] != nres)
			{
				resbuffer.resize({nres});
				resbuff = resbuffer.request();
			}
			if (flag && (derivbuff.shape[0] != nres || derivbuff.shape[1] != nargs))
			{
				derivbuffer.resize({nres, nargs});
				derivbuff = derivbuffer.request();
			}
			for (unsigned int i = 0; i < nargs; i++)
				((double *)(argbuff.ptr))[i] = args[i];
			this->eval(flag, argbuffer, resbuffer, derivbuffer);
			for (unsigned int i = 0; i < nres; i++)
				res[i] = ((double *)(resbuff.ptr))[i];
			if (flag)
			{
				for (unsigned int i = 0; i < nres; i++)
					for (unsigned int j = 0; j < nargs; j++)
						derivs[i * nargs + j] = ((double *)(derivbuff.ptr))[i * nargs + j];
			}
		}
	};

	class PyCustomMultiReturnExpression : public CustomMultiReturnExpression
	{
	public:
		using CustomMultiReturnExpression::CustomMultiReturnExpression;
		std::string get_id_name() override
		{
			PYBIND11_OVERLOAD(std::string, CustomMultiReturnExpression, get_id_name);
		}
		void eval(int flag, py::array_t<double> &arg_list, py::array_t<double> &result_list, py::array_t<double> &derivative_matrix) override
		{
			PYBIND11_OVERLOAD_PURE(void, CustomMultiReturnExpression, eval, flag, arg_list, result_list, derivative_matrix);
		}
		std::string _get_c_code() override
		{
			PYBIND11_OVERLOAD(std::string, CustomMultiReturnExpression, _get_c_code);
		}
		std::pair<bool, GiNaC::ex> _get_symbolic_derivative(const std::vector<GiNaC::ex> &arg_list, const int &i_res, const int &j_arg) override
		{
			typedef std::pair<bool, GiNaC::ex> sym_expr_ret_pair;
			PYBIND11_OVERLOAD(sym_expr_ret_pair, CustomMultiReturnExpression, _get_symbolic_derivative, arg_list, i_res, j_arg);
		}
	};

	class PyCustomCoordinateSystem : public CustomCoordinateSystem
	{
	public:
		using CustomCoordinateSystem::CustomCoordinateSystem;

		int vector_gradient_dimension(unsigned int basedim, bool lagrangian) override
		{
			PYBIND11_OVERLOAD(int, CustomCoordinateSystem, vector_gradient_dimension, basedim, lagrangian);
		}

		GiNaC::ex grad(const GiNaC::ex &f, int ndim, int edim, int flags) override
		{
			PYBIND11_OVERLOAD(GiNaC::ex, CustomCoordinateSystem, grad, f, ndim, edim, flags);
		}

		GiNaC::ex directional_derivative(const GiNaC::ex &f, const GiNaC::ex &d, int ndim, int edim, int flags) override
		{
			PYBIND11_OVERLOAD(GiNaC::ex, CustomCoordinateSystem, directional_derivative, f, d, ndim, edim, flags);
		}

		GiNaC::ex general_weak_differential_contribution(std::string funcname, std::vector<GiNaC::ex> lhs, GiNaC::ex test, int dim, int edim, int flags) override
		{
			PYBIND11_OVERLOAD(GiNaC::ex, CustomCoordinateSystem, general_weak_differential_contribution, funcname, lhs, test, dim, edim, flags);
		}

		GiNaC::ex div(const GiNaC::ex &v, int ndim, int edim, int flags) override
		{
			PYBIND11_OVERLOAD(GiNaC::ex, CustomCoordinateSystem, div, v, ndim, edim, flags);
		}

		GiNaC::ex geometric_jacobian() override
		{
			PYBIND11_OVERLOAD(GiNaC::ex, CustomCoordinateSystem, geometric_jacobian);
		}

		GiNaC::ex jacobian_for_element_size() override
		{
			PYBIND11_OVERLOAD(GiNaC::ex, CustomCoordinateSystem, jacobian_for_element_size);
		}

		std::string get_id_name() override
		{
			PYBIND11_OVERLOAD(std::string, CustomCoordinateSystem, get_id_name);
		}

		GiNaC::ex get_mode_expansion_of_var_or_test(pyoomph::FiniteElementCode *mycode, std::string fieldname, bool is_field, bool is_dim, GiNaC::ex expr, std::string where, int expansion_mode) override
		{
			PYBIND11_OVERLOAD(GiNaC::ex, CustomCoordinateSystem, get_mode_expansion_of_var_or_test, mycode, fieldname, is_field, is_dim, expr, where, expansion_mode);
		}
	};

	static GiNaC::ex GiNaCFromString(const std::string &v)
	{
		return GiNaC::ex(v, GiNaC::lst());
	}

	static GiNaC::ex GiNaCFromExArray(std::vector<GiNaC::ex> v)
	{
		return 0 + GiNaC::matrix(v.size(), 1, GiNaC::lst(v.begin(), v.end()));
	}

	static GiNaC::ex GiNaCFromDoubleArray(std::vector<double> v)
	{
		std::vector<GiNaC::ex> vex;
		for (auto e : v)
			vex.push_back(e);
		return GiNaCFromExArray(vex);
	}

	static GiNaC::ex GiNaCFromGlobalParam(GiNaC::GiNaCGlobalParameterWrapper w)
	{
		return 0 + w;
	}

	static GiNaC::ex GiNaCFromDouble(const double &v)
	{
		return GiNaC::ex(v);
	}

}

void PyReg_Expressions(py::module &m)
{

	py::class_<GiNaC::ex>(m, "Expression")
		.def(py::init<const int &>())
		.def(py::init<const double &>())
		.def(py::init<const GiNaC::ex &>())
		.def(py::init(&pyoomph::GiNaCFromDouble))
		.def(py::init(&pyoomph::GiNaCFromString))
		.def(py::init(&pyoomph::GiNaCFromDoubleArray))
		.def(py::init(&pyoomph::GiNaCFromExArray))
		.def(py::init(&pyoomph::GiNaCFromGlobalParam))
		.def(py::self + py::self)
		.def(int() + py::self)
		.def(py::self + int())
		.def(double() + py::self)
		.def(py::self + double())

		.def(py::self - py::self)
		.def(int() - py::self)
		.def(py::self - int())
		.def(double() - py::self)
		.def(py::self - double())

		.def(py::self * py::self)
		.def(double() * py::self)
		.def(py::self * double())
		.def(int() * py::self)
		.def(py::self * int())

		.def(py::self / py::self)
		.def(int() / py::self)
		.def(py::self / int())
		.def(double() / py::self)
		.def(py::self / double())

		.def(py::self += py::self)
		.def(py::self -= py::self)
		.def(py::self *= py::self)
		.def(py::self *= int())
		.def(py::self *= double())
		.def(py::self /= py::self)
		.def(py::self /= int())
		.def(py::self /= double())

		.def("__add__", [](const GiNaC::ex &lh, const std::complex<double> &rh)
			 { return lh + (rh.real()+GiNaC::I*rh.imag()); }, py::is_operator())
		.def("__sub__", [](const GiNaC::ex &lh, const std::complex<double> &rh)
			 { return lh - (rh.real()+GiNaC::I*rh.imag()); }, py::is_operator())
		.def("__mul__", [](const GiNaC::ex &lh, const std::complex<double> &rh)
			 { return lh * (rh.real()+GiNaC::I*rh.imag()); }, py::is_operator())
		.def("__truediv__", [](const GiNaC::ex &lh, const std::complex<double> &rh)
			 { return lh / (rh.real()+GiNaC::I*rh.imag()); }, py::is_operator())
		.def("__iadd__", [](const GiNaC::ex &lh, const std::complex<double> &rh)
			 { return lh + (rh.real()+GiNaC::I*rh.imag()); }, py::is_operator())
		.def("__isub__", [](const GiNaC::ex &lh, const std::complex<double> &rh)
			 { return lh - (rh.real()+GiNaC::I*rh.imag()); }, py::is_operator())
		.def("__imul__", [](const GiNaC::ex &lh, const std::complex<double> &rh)
			 { return lh * (rh.real()+GiNaC::I*rh.imag()); }, py::is_operator())
		.def("__itruediv__", [](const GiNaC::ex &lh, const std::complex<double> &rh)
			 { return lh / (rh.real()+GiNaC::I*rh.imag()); }, py::is_operator())

		.def("__radd__", [](const GiNaC::ex &lh, const std::complex<double> &rh)
			 { return (rh.real()+GiNaC::I*rh.imag())+lh; }, py::is_operator())
		.def("__rsub__", [](const GiNaC::ex &lh, const std::complex<double> &rh)
			 { return (rh.real()+GiNaC::I*rh.imag())-lh; }, py::is_operator())
		.def("__rmul__", [](const GiNaC::ex &lh, const std::complex<double> &rh)
			 { return (rh.real()+GiNaC::I*rh.imag())*lh; }, py::is_operator())
		.def("__rtruediv__", [](const GiNaC::ex &lh, const std::complex<double> &rh)
			 { return (rh.real()+GiNaC::I*rh.imag())/lh; }, py::is_operator())
		

		



		.def("__add__", [](const GiNaC::ex &lh, const GiNaC::GiNaCGlobalParameterWrapper &rh)
			 { return lh + rh; }, py::is_operator())
		.def("__sub__", [](const GiNaC::ex &lh, const GiNaC::GiNaCGlobalParameterWrapper &rh)
			 { return lh - rh; }, py::is_operator())
		.def("__mul__", [](const GiNaC::ex &lh, const GiNaC::GiNaCGlobalParameterWrapper &rh)
			 { return lh * rh; }, py::is_operator())
		.def("__truediv__", [](const GiNaC::ex &lh, const GiNaC::GiNaCGlobalParameterWrapper &rh)
			 { return lh / rh; }, py::is_operator())
		.def("__imul__", [](const GiNaC::ex &lh, const GiNaC::GiNaCGlobalParameterWrapper &rh)
			 { return lh * rh; }, py::is_operator())
		.def("__itruediv__", [](const GiNaC::ex &lh, const GiNaC::GiNaCGlobalParameterWrapper &rh)
			 { return lh / rh; }, py::is_operator())
		.def("__iadd__", [](const GiNaC::ex &lh, const GiNaC::GiNaCGlobalParameterWrapper &rh)
			 { return lh + rh; }, py::is_operator())
		.def("__isub__", [](const GiNaC::ex &lh, const GiNaC::GiNaCGlobalParameterWrapper &rh)
			 { return lh - rh; }, py::is_operator())
		.def("__round__",[](const GiNaC::ex &self)
		{
		  try 
			   {
			     double res=pyoomph::expressions::eval_to_double(self);
			     return (long int)std::round(res);			     
			   }
			   catch (const std::exception &e) 
			   {
			     std::ostringstream oss; oss<<"Cannot convert " << self << " to double for rounding";
			     throw_runtime_error(oss.str());
			   } 
		})
		// Functionalities to use e.g. numpy.sqrt on GiNaC expressions. They will be GiNaC, though
		.def("sqrt", [](const GiNaC::ex &lh)
			 { return GiNaC::sqrt(lh); })
		.def("exp", [](const GiNaC::ex &lh)
			 { return 0 + GiNaC::exp(lh); })
		.def("cos", [](const GiNaC::ex &lh)
			 { return 0 + GiNaC::cos(lh); })
		.def("sin", [](const GiNaC::ex &lh)
			 { return 0 + GiNaC::sin(lh); })
		.def("tan", [](const GiNaC::ex &lh)
			 { return 0 + GiNaC::tan(lh); })
		.def("tanh", [](const GiNaC::ex &lh)
			 { return 0 + GiNaC::tanh(lh); })
		.def("atan", [](const GiNaC::ex &lh)
			 { return 0 + GiNaC::atan(lh); })
		.def("atan2", [](const GiNaC::ex &lh, const GiNaC::ex &lh2)
			 { return 0 + GiNaC::atan2(lh, lh2); })
		.def("acos", [](const GiNaC::ex &lh)
			 { return 0 + GiNaC::acos(lh); })
		.def("asin", [](const GiNaC::ex &lh)
			 { return 0 + GiNaC::asin(lh); })
		.def("log", [](const GiNaC::ex &lh)
			 { return 0 + GiNaC::log(lh); })

		.def(-py::self)
		.def("__pow__", [](const GiNaC::ex &lh, const GiNaC::ex &rh)
			 { return GiNaC::pow(lh, rh); }, py::is_operator())
		.def("__pow__", [](const GiNaC::ex &lh, const int &rh)
			 { return GiNaC::pow(lh, rh); }, py::is_operator())
		.def("__pow__", [](const int &lh, const GiNaC::ex &rh)
			 { return GiNaC::pow(lh, rh); }, py::is_operator())
		.def("__pow__", [](const double &lh, const GiNaC::ex &rh)
			 { return GiNaC::pow(lh, rh); }, py::is_operator())
		.def("__pow__", [](const GiNaC::ex &lh, const double &rh)
			 { return GiNaC::pow(lh, rh); }, py::is_operator())
		.def("__pow__", [](const GiNaC::ex &lh, const GiNaC::GiNaCGlobalParameterWrapper &rh)
			 { return GiNaC::pow(lh, rh); }, py::is_operator())

		.def("__rpow__", [](const GiNaC::ex &rh, const int &lh)
			 { return GiNaC::pow(lh, rh); }, py::is_operator())
		.def("__rpow__", [](const GiNaC::ex &rh, const int &lh)
			 { return GiNaC::pow(lh, rh); }, py::is_operator())
		.def("__rpow__", [](const GiNaC::ex &rh, const double &lh)
			 { return GiNaC::pow(lh, rh); }, py::is_operator())

		.def("__matmul__", [](const GiNaC::ex &lh, const GiNaC::ex &rh)
			 { return 0 + pyoomph::expressions::contract(lh, rh); }, py::is_operator())
		.def("get_type_information", [](const GiNaC::ex &self)
			 {
         auto & ci=GiNaC::ex_to<GiNaC::basic>(self).get_class_info().options;
         std::map<std::string,std::string> res;
         res["class_name"]=std::string(ci.get_name());
         res["parent_class_name"]=std::string(ci.get_parent_name());
         if (GiNaC::is_a<GiNaC::numeric>(self))
         {
          GiNaC::numeric num=GiNaC::ex_to<GiNaC::numeric>(self);
          res["is_integer"]=(num.is_integer() ? "true" : "false");
          res["is_real"]=(num.is_real() ? "true" : "false");
          res["is_rational"]=(num.is_rational() ? "true" : "false");          
         }
         else if (GiNaC::is_a<GiNaC::function>(self))
         {
          GiNaC::function fun=GiNaC::ex_to<GiNaC::function>(self);
          res["function_name"]=fun.get_name();
         }
         return res; })
		.def("op", &GiNaC::ex::op, py::return_value_policy::reference)
		.def("nops", &GiNaC::ex::nops)
		.def("numer", &GiNaC::ex::numer, py::return_value_policy::reference)
		.def("denom", &GiNaC::ex::denom, py::return_value_policy::reference)
		.def("evalm", &GiNaC::ex::evalm, py::return_value_policy::reference)
		.def("evalf", &GiNaC::ex::evalf, py::return_value_policy::reference)
		.def("merge_units",[](const GiNaC::ex &self)
			 { 
			 GiNaC::ex factor,units,rest; bool res=pyoomph::expressions::collect_base_units(self,factor,units,rest); return 0+factor*rest*units;			   
			 }, py::return_value_policy::reference)
		.def("parameters_to_current_values",[](const GiNaC::ex &self)
			 { 
			   return 0+pyoomph::expressions::replace_global_params_by_current_values(self);			   
			 }, py::return_value_policy::reference)
		.def("is_zero", &GiNaC::ex::is_zero)
		.def("__float__", [](const GiNaC::ex &self)
			 { try 
			   {
			     double res=pyoomph::expressions::eval_to_double(self);
			     return res; 
			   }
			   catch (const std::exception &e) 
			   {
			     std::ostringstream oss; oss<<"Cannot convert " << self << " to double";
			     throw_runtime_error(oss.str());
			   } })
		.def("__complex__", [](const GiNaC::ex &self)
			 { try 
			   {
			     std::complex<double> res=pyoomph::expressions::eval_to_complex(self);
			     return res; 
			   }
			   catch (const std::exception &e) 
			   {
			     std::ostringstream oss; oss<<"Cannot convert " << self << " to complex";
			     throw_runtime_error(oss.str());
			   } })
		.def("__int__", [](const GiNaC::ex &self)
			 { 
		GiNaC::ex v = GiNaC::evalf(self);
   	 if (GiNaC::is_a<GiNaC::numeric>(v)) {
        return (long int)(GiNaC::ex_to<GiNaC::numeric>(v).to_double());
   	 } else {
			std::ostringstream oss; oss << "Cannot cast the following into a numeric: "<< v ; throw_runtime_error(oss.str());
		} })
		.def("float_value", [](const GiNaC::ex &self)
			 { return pyoomph::expressions::eval_to_double(self); })
		.def("__getitem__", [](const GiNaC::ex &self, const int &i)
			 {  
			GiNaC::ex evm=self.evalm();
			if (!GiNaC::is_a<GiNaC::matrix>(evm)) {
				return 0+pyoomph::expressions::single_index(evm,GiNaC::numeric(i));
			}
			GiNaC::matrix m=GiNaC::ex_to<GiNaC::matrix>(evm);
			return m(i,0); }, py::return_value_policy::reference)
		.def("__getitem__", [](const GiNaC::ex &self, const py::tuple &ind)
			 {  
			GiNaC::ex evm=self.evalm();
			if (!GiNaC::is_a<GiNaC::matrix>(evm)) {
				return 0+pyoomph::expressions::double_index(evm,GiNaC::numeric(ind[0].cast<int>()),GiNaC::numeric(ind[1].cast<int>()));
			}
			GiNaC::matrix m=GiNaC::ex_to<GiNaC::matrix>(evm);
			return m(ind[0].cast<int>(),ind[1].cast<int>()); }, py::return_value_policy::reference)
		.def("__repr__", [](const GiNaC::ex &self)
			 { 
  	 std::ostringstream oss; 
  	 GiNaC::print_python pypc(oss);  	 
  	 (self+0).print(pypc);  	 
	 return oss.str(); })
		.def("print_latex", [](const GiNaC::ex &self)
			 { 
  	 std::ostringstream oss; 
  	 GiNaC::print_latex pypc(oss);  	 
  	 self.print(pypc);
	 return oss.str(); });

	py::class_<pyoomph::CustomCoordinateSystem, pyoomph::PyCustomCoordinateSystem>(m, "CustomCoordinateSystem")
		.def(py::init<>())
		.def("vector_gradient_dimension", &pyoomph::CustomCoordinateSystem::vector_gradient_dimension, py::arg("basedim"), py::arg("lagrangian"))
		.def("grad", &pyoomph::CustomCoordinateSystem::grad, py::arg("arg"), py::arg("ndim"), py::arg("edim"), py::arg("flags"))
		.def("directional_derivative", &pyoomph::CustomCoordinateSystem::directional_derivative, py::arg("arg"), py::arg("direct"), py::arg("ndim"), py::arg("edim"), py::arg("flags"))
		.def("general_weak_differential_contribution", &pyoomph::CustomCoordinateSystem::general_weak_differential_contribution, py::arg("funcname"), py::arg("lhs"), py::arg("test"), py::arg("dim"), py::arg("edim"), py::arg("flags"))
		.def("div", &pyoomph::CustomCoordinateSystem::div, py::arg("arg"), py::arg("ndim"), py::arg("edim"), py::arg("flags"))
		.def("geometric_jacobian", &pyoomph::CustomCoordinateSystem::geometric_jacobian)
		.def("jacobian_for_element_size", &pyoomph::CustomCoordinateSystem::jacobian_for_element_size)
		.def("get_mode_expansion_of_var_or_test", &pyoomph::CustomCoordinateSystem::get_mode_expansion_of_var_or_test, py::arg("code"), py::arg("fieldname"), py::arg("is_field"), py::arg("is_dim"), py::arg("expr"), py::arg("where"), py::arg("expansion_mode"))
		.def("get_id_name", &pyoomph::CustomCoordinateSystem::get_id_name);

	py::class_<pyoomph::CustomMathExpressionBase>(m, "CustomMathExpressionBase");

	py::class_<pyoomph::CustomMathExpression, pyoomph::PyCustomMathExpression, pyoomph::CustomMathExpressionBase>(m, "CustomMathExpression")
		.def(py::init<>())
		.def("get_id_name", &pyoomph::CustomMathExpression::get_id_name)
		.def("outer_derivative", &pyoomph::CustomMathExpression::outer_derivative, py::arg("x"), py::arg("index"))
		.def("get_diff_index", &pyoomph::CustomMathExpression::get_diff_index)
		.def("get_diff_parent", &pyoomph::CustomMathExpression::get_diff_parent, py::return_value_policy::reference)
		.def(
			"set_as_derivative", [](pyoomph::CustomMathExpression *self, pyoomph::CustomMathExpression *p, int index)
			{ self->set_as_derivative(p, index); },
			py::keep_alive<1, 2>())
		.def("get_argument_unit", &pyoomph::CustomMathExpression::get_argument_unit, py::arg("index"))
		.def("get_result_unit", &pyoomph::CustomMathExpression::get_result_unit)
		.def("real_part", &pyoomph::CustomMathExpression::real_part, py::arg("invokation"), py::arg("arglst"))
		.def("imag_part", &pyoomph::CustomMathExpression::imag_part, py::arg("invokation"), py::arg("arglst"))
		.def(
			"set_as_derivative", [](pyoomph::CustomMathExpression *self, pyoomph::CustomMathExpression *p, int index)
			{ self->set_as_derivative(p, index); },
			py::keep_alive<1, 2>())
		.def("eval", &pyoomph::CustomMathExpression::eval, py::arg("arg_array"));

	py::class_<pyoomph::CustomMultiReturnExpressionBase>(m, "CustomMultiReturnExpressionBase");
	py::class_<pyoomph::CustomMultiReturnExpression, pyoomph::PyCustomMultiReturnExpression, pyoomph::CustomMultiReturnExpressionBase>(m, "CustomMultiReturnExpression")
		.def(py::init<>())
		.def("get_id_name", &pyoomph::CustomMultiReturnExpression::get_id_name)
		.def("eval", &pyoomph::CustomMultiReturnExpression::eval, py::arg("flag"), py::arg("arg_list"), py::arg("result_list"), py::arg("derivative_matrix"))
		.def("set_debug_python_vs_c_epsilon", [](pyoomph::CustomMultiReturnExpression *self, double eps)
			 { self->debug_c_code_epsilon = eps; })
		.def("_get_symbolic_derivative", &pyoomph::CustomMultiReturnExpression::_get_symbolic_derivative)
		.def("_get_c_code", &pyoomph::CustomMultiReturnExpression::_get_c_code);

	m.def(
		"GiNaC_rational_number", [](const int &num, const int &denom)
		{ return GiNaC::ex(GiNaC::numeric(num, denom)); },
		"Rational number");
	m.def("GiNaC_imaginary_i", []() -> GiNaC::ex
		  { return 0 + GiNaC::I; });
	m.def("GiNaC_get_real_part", [](const GiNaC::ex &arg) -> GiNaC::ex
		  { return 0 + pyoomph::expressions::get_real_part(arg); });
	m.def("GiNaC_get_imag_part", [](const GiNaC::ex &arg) -> GiNaC::ex
		  { return 0 + pyoomph::expressions::get_imag_part(arg); });
	m.def("GiNaC_split_subexpressions_in_real_and_imaginary_parts",[](const GiNaC::ex &arg) -> GiNaC::ex
		  { return 0 + pyoomph::expressions::split_subexpressions_in_real_and_imaginary_parts(arg); });
	m.def(
		"GiNaC_sin", [](const GiNaC::ex &arg)
		{ return 0 + GiNaC::sin(arg); },
		"Calculates the sine");
	m.def(
		"GiNaC_sinh", [](const GiNaC::ex &arg)
		{ return 0 + GiNaC::sinh(arg); },
		"Calculates the sine hyperbolicus");
	m.def(
		"GiNaC_cosh", [](const GiNaC::ex &arg)
		{ return 0 + GiNaC::cosh(arg); },
		"Calculates the sine hyperbolicus");
	m.def(
		"GiNaC_asin", [](const GiNaC::ex &arg)
		{ return 0 + GiNaC::asin(arg); },
		"Calculates asin");
	m.def(
		"GiNaC_asin", [](const double &arg)
		{ return 0 + GiNaC::asin(arg); },
		"Calculates asin");
	m.def(
		"GiNaC_cos", [](const GiNaC::ex &arg)
		{ return 0 + GiNaC::cos(arg); },
		"Calculates the cosine");
	m.def(
		"GiNaC_acos", [](const GiNaC::ex &arg)
		{ return 0 + GiNaC::acos(arg); },
		"Calculates acos");
	m.def(
		"GiNaC_tan", [](const GiNaC::ex &arg)
		{ return 0 + GiNaC::tan(arg); },
		"Calculates tan");
	m.def(
		"GiNaC_tanh", [](const GiNaC::ex &arg)
		{ return 0 + GiNaC::tanh(arg); },
		"Calculates tanh");
	m.def(
		"GiNaC_atan", [](const GiNaC::ex &arg)
		{ return 0 + GiNaC::atan(arg); },
		"Calculates atan");
	m.def(
		"GiNaC_atan2", [](const GiNaC::ex &y, const GiNaC::ex &x)
		{ return 0 + GiNaC::atan2(y, x); },
		"Calculates atan2");
	m.def(
		"GiNaC_exp", [](const GiNaC::ex &arg)
		{ return 0 + GiNaC::exp(arg); },
		"Calculates exp");
	m.def(
		"GiNaC_log", [](const GiNaC::ex &arg)
		{ return 0 + GiNaC::log(arg); },
		"Calculates natural log");
	m.def(
		"GiNaC_heaviside", [](const GiNaC::ex &arg)
		{ return 0 + pyoomph::expressions::heaviside(arg); },
		"Calculates the step function"); // TODO Derivatives of step
	m.def(
		"GiNaC_piecewise_geq0", [](const GiNaC::ex &cond, const GiNaC::ex &a,const GiNaC::ex &b)
		{ return 0 + pyoomph::expressions::piecewise_geq0(cond, a,b); },		
		"Returns a if cond>=0, else b"); // TODO Derivatives of step		
	m.def(
		"GiNaC_minimum", [](const GiNaC::ex &a, const GiNaC::ex &b)
		{ return 0 + pyoomph::expressions::minimum(a, b); },
		"Calculates the minimum"); // TODO Derivatives of step
	m.def(
		"GiNaC_maximum", [](const GiNaC::ex &a, const GiNaC::ex &b)
		{ return 0 + pyoomph::expressions::maximum(a, b); },
		"Calculates the maximum"); // TODO Derivatives of step
	m.def(
		"GiNaC_absolute", [](const GiNaC::ex &arg)
		{ return 0 + pyoomph::expressions::absolute(arg); },
		"Calculates the absolute value. Note: It will differentiate as absolute(f(x))'=signum(f(x))*f'(x)");
	m.def(
		"GiNaC_signum", [](const GiNaC::ex &arg)
		{ return 0 + pyoomph::expressions::signum(arg); },
		"Calculates the signum of the argument. Note: It will differentiate to 0, even at x=0");
	
	m.def("GiNaC_is_a_matrix", [](const GiNaC::ex &arg)
		  {GiNaC::ex evm=arg.evalm(); return GiNaC::is_a<GiNaC::matrix>(evm); });

	m.def("GiNaC_debug_ex", [](const GiNaC::ex &arg)
		  { return 0 + pyoomph::expressions::debug_ex(arg); });
	m.def("GiNaC_matproduct", [](const GiNaC::ex &m1, const GiNaC::ex &m2)
		  { return 0 + pyoomph::expressions::matproduct(m1, m2); });

	m.def(
		"GiNaC_expand", [](const GiNaC::ex &arg)
		{ return 0 + pyoomph::expressions::ginac_expand(arg); },
		"Expand expression after internal expansion of all fields with GiNaC::expand");

	m.def("GiNaC_collect", [](const GiNaC::ex &arg, const GiNaC::ex &s)
		  { return 0 + pyoomph::expressions::ginac_collect(arg, s); });
	m.def("GiNaC_factor", [](const GiNaC::ex &arg)
		  { return 0 + pyoomph::expressions::ginac_factor(arg); });
	m.def("GiNaC_normal", [](const GiNaC::ex &arg)
		  { return 0 + pyoomph::expressions::ginac_normal(arg); });
	m.def("GiNaC_collect_common_factors", [](const GiNaC::ex &arg)
		  { return 0 + pyoomph::expressions::ginac_collect_common_factors(arg); });
	m.def("GiNaC_series", [](const GiNaC::ex &arg, const GiNaC::ex &x, const GiNaC::ex &x0, const GiNaC::ex &order)
		  { return 0 + pyoomph::expressions::ginac_series(arg, x, x0, order); });

	m.def(
		"GiNaC_wrap_coordinate_system", [](pyoomph::CustomCoordinateSystem &sys) -> GiNaC::ex
		{ return 0 + GiNaC::GiNaCCustomCoordinateSystemWrapper(pyoomph::CustomCoordinateSystemWrapper(&sys)); },
		py::keep_alive<0, 1>(), py::keep_alive<1, 0>()); // TODO: Bad hack. Mutual Keep alive will force them to live for ever

	m.def(
		"GiNaC_python_cb_function", [](pyoomph::CustomMathExpression *pfunc, const std::vector<GiNaC::ex> &args)
		{
			std::vector<GiNaC::ex> ndargs(args.size());
			for (unsigned int i = 0; i < args.size(); i++)
            {
                ndargs[i] = args[i] / (pfunc->get_argument_unit(i));
            }
			return 0 + pfunc->get_result_unit() * pyoomph::expressions::python_cb_function(GiNaC::GiNaCCustomMathExpressionWrapper(pyoomph::CustomMathExpressionWrapper(pfunc)), GiNaC::lst(ndargs.begin(), ndargs.end())); },
		py::keep_alive<0, 1>(), py::keep_alive<1, 0>()); // TODO: Bad hack. Mutual Keep alive will force them to live for ever

	m.def(
		"GiNaC_python_multi_cb_function", [](pyoomph::CustomMultiReturnExpressionBase *pfunc, const std::vector<GiNaC::ex> &args, const int &numret)
		{ return 0 + pyoomph::expressions::python_multi_cb_function(GiNaC::GiNaCCustomMultiReturnExpressionWrapper(pyoomph::CustomMultiReturnExpressionWrapper(pfunc)), GiNaC::lst(args.begin(), args.end()), GiNaC::ex(numret)); },
		py::keep_alive<0, 1>(), py::keep_alive<1, 0>()); // TODO: Bad hack. Mutual Keep alive will force them to live for ever

	m.def(
		"GiNaC_python_multi_cb_indexed_result", [](const GiNaC::ex &pfunc, const int &index)
		{ return 0 + pyoomph::expressions::python_multi_cb_indexed_result(pfunc, GiNaC::ex(index)); },
		py::keep_alive<0, 1>(), py::keep_alive<1, 0>()); // TODO: Bad hack. Mutual Keep alive will force them to live for ever

	m.def(
		"GiNaC_collect_units", [](const GiNaC::ex &arg)
		{GiNaC::ex factor,units,rest; bool res=pyoomph::expressions::collect_base_units(arg,factor,units,rest); return std::make_tuple(0+factor,0+units,0+rest,res); },
		"Splits an expression into a numerical factor, units, and the rest");

	m.def("GiNaC_TimeSymbol", []()
		  {GiNaC::ex res=0+GiNaC::GiNaCTimeSymbol(pyoomph::TimeSymbol()); return res; });
	m.def(
		"GiNaC_FakeExponentialMode", [](GiNaC::ex arg, bool dual)
		{GiNaC::ex res=0+GiNaC::GiNaCFakeExponentialMode(pyoomph::FakeExponentialMode(arg,dual)); return res; },
		py::arg("mode"), py::arg("dual") = false);

	m.def("GiNaC_unit", [](const std::string name, const std::string shortname)
		  {
		if (!pyoomph::base_units.count(name)) pyoomph::base_units[name]=GiNaC::possymbol(shortname);
		return 0+pyoomph::base_units[name]; });

	m.def("GiNaC_sep_base_units", [](GiNaC::ex in)
		  {
			  std::map<std::string, std::pair<int, unsigned>> occurrences;
			  GiNaC::ex match_exp = GiNaC::wild(0);
			  GiNaC::ex match_fact = GiNaC::wild(1);
			  for (auto &bu : pyoomph::base_units)
			  {
				  GiNaC::lst sublist;
				  for (auto &bu2 : pyoomph::base_units)
				  {
					  if (bu.first != bu2.first)
					  {
						  sublist.append(bu2.second == 1);
					  }
				  }
				  GiNaC::ex simpl = GiNaC::expand(in.subs(sublist)).normal().numer_denom();
				  GiNaC::ex numer = simpl.op(0);
				  GiNaC::ex denom = simpl.op(1);
				  GiNaC::ex inv;
				  int sign;
				  if (GiNaC::has(numer, bu.second))
				  {
					  sign = 1;
					  if (GiNaC::has(denom, bu.second))
					  {
						  std::ostringstream oss;
						  oss << numer << " | " << denom;
						  throw_runtime_error("Has contribution in num and denom: " + oss.str());
					  }
					  inv = numer;
				  }
				  else if (GiNaC::has(denom, bu.second))
				  {
					  sign = -1;
					  inv = denom;
				  }
				  else
				  {
					  continue;
				  }

				  bool matchres = inv.match(bu.second);
				  int pnumer;
				  unsigned pdenom;
				  if (matchres)
				  {
					  pnumer = sign;
					  pdenom = 1;
				  }
				  else
				  {
					  GiNaC::exmap repls;
					  matchres = inv.match(pow(bu.second, match_exp), repls);
					  if (!matchres)
					  {
						  matchres = inv.match(match_fact * pow(bu.second, match_exp), repls);
						  if (!matchres)
						  {
							  throw_runtime_error("Cannot handle this");
						  }
					  }
					  GiNaC::ex p = repls[match_exp];
					  if (!GiNaC::is_a<GiNaC::numeric>(p))
					  {
						  throw_runtime_error("Nonnumeric unit power");
					  }
					  GiNaC::numeric pnum = GiNaC::ex_to<GiNaC::numeric>(p);
					  if (pnum.is_rational())
					  {
						  pnumer = sign * pnum.numer().to_int();
						  pdenom = pnum.denom().to_int();
					  }
					  else
					  {
						  throw_runtime_error("Non-rational unit power. Please use e.g. L=V**rational_num(1,3) instead of L=V**(1/3), i.e. all exponents involving units must use rational_num since float accuracy is truncated and the corresponding units cannot be calculated then");
					  }
				  }
				  occurrences[bu.first] = std::make_pair(pnumer, pdenom);
			  }
			  return occurrences; }

	);

	m.def("GiNaC_subsfields", [](const GiNaC::ex &arg, const std::map<std::string, GiNaC::ex> &fields, const std::map<std::string, GiNaC::ex> &nondimfields, const std::map<std::string, GiNaC::ex> &globalparams)
		  { return 0 + pyoomph::expressions::subs_fields(arg, fields, nondimfields, globalparams); });

	m.def("GiNaC_time_stepper_weight", [](const int &order, const int index, std::string scheme)
		  { 
	  	  if (!pyoomph::__field_name_cache.count(scheme)) pyoomph::__field_name_cache.insert(std::make_pair(scheme,GiNaC::realsymbol(scheme)));
		  return 0 + pyoomph::expressions::time_stepper_weight(order, index,pyoomph::__field_name_cache[scheme]); });

	m.def(
		"GiNaC_general_weak_differential_contribution", [](std::string funcname, std::vector<GiNaC::ex> f, const GiNaC::ex rhs, const GiNaC::ex &ndim, const GiNaC::ex &edim, const GiNaC::ex &coordsys, const GiNaC::ex &flags)
		{
			if (!pyoomph::__field_name_cache.count(funcname))
				pyoomph::__field_name_cache.insert(std::make_pair(funcname, GiNaC::realsymbol(funcname)));
			GiNaC::lst flst(f.begin(), f.end());
			if (coordsys.is_zero())
			{
				return 0 + pyoomph::expressions::general_weak_differential_contribution(pyoomph::__field_name_cache[funcname], flst, rhs, ndim, edim, pyoomph::__no_coordinate_system_wrapper, flags);
			}
			else
			{
				return 0 + pyoomph::expressions::general_weak_differential_contribution(pyoomph::__field_name_cache[funcname], flst, rhs, ndim, edim, coordsys, flags);
			} },
		"Any differential weak contribution that depends on the coordinate system");
	m.def(
		"GiNaC_grad", [](const GiNaC::ex &arg, const GiNaC::ex &ndim, const GiNaC::ex &edim, const GiNaC::ex &coordsys, const GiNaC::ex &withdim)
		{
		if (coordsys.is_zero())
		{
		  return 0+pyoomph::expressions::grad(arg,ndim,edim,pyoomph::__no_coordinate_system_wrapper,withdim);
		}
    else  
		{
			return 0+pyoomph::expressions::grad(arg,ndim,edim,coordsys,withdim);
		} },
		"Calculates the gradient");

	m.def(
		"GiNaC_directional_derivative", [](const GiNaC::ex &f, const GiNaC::ex &d, const GiNaC::ex &ndim, const GiNaC::ex &edim, const GiNaC::ex &coordsys, const GiNaC::ex &withdim)
		{
		if (coordsys.is_zero())
		{
		  return 0+pyoomph::expressions::directional_derivative(f,d,ndim,edim,pyoomph::__no_coordinate_system_wrapper,withdim);
		}
    else  
		{
			return 0+pyoomph::expressions::directional_derivative(f,d,ndim,edim,coordsys,withdim);
		} },
		"Calculates the directional derivative of a scalar, matrix or tensor");

	m.def(
		"GiNaC_div", [](const GiNaC::ex &arg, const GiNaC::ex &ndim, const GiNaC::ex &edim, const GiNaC::ex &coordsys, const GiNaC::ex &withdim)
		{
		if (coordsys.is_zero())
		{
		  return 0+pyoomph::expressions::div(arg,ndim,edim,pyoomph::__no_coordinate_system_wrapper,withdim);
		}
    else  
		{
			return 0+pyoomph::expressions::div(arg,ndim,edim,coordsys,withdim);
		} },
		"Calculates the divergence");

	m.def(
		"GiNaC_dot", [](const GiNaC::ex &arg1, const GiNaC::ex &arg2)
		{ return 0 + pyoomph::expressions::dot(arg1, arg2); },
		"Calculates the dot product");
	m.def(
		"GiNaC_diff", [](const GiNaC::ex &arg1, const GiNaC::ex &arg2)
		{ return 0 + pyoomph::expressions::diff(arg1, arg2); },
		"Calculates the derivative");
	m.def(
		"GiNaC_Diff", [](const GiNaC::ex &arg1, const GiNaC::ex &arg2)
		{ return 0 + pyoomph::expressions::Diff(arg1, arg2); },
		"Derivative, but does not evaluate until code generation");
	m.def(
		"GiNaC_SymSubs", [](const GiNaC::ex &arg1, const GiNaC::ex &what, const GiNaC::ex &by_what)
		{ return 0 + pyoomph::expressions::symbol_subs(arg1, what, by_what); },
		"Call GiNaC::subs, but does not evaluate until code generation");
	m.def(
		"GiNaC_subs", [](const GiNaC::ex &arg1, const GiNaC::ex &what, const GiNaC::ex &by_what)
		{ return 0 + arg1.subs(GiNaC::lst{what}, GiNaC::lst{by_what}); },
		"Call GiNaC::subs, evaluate directly");
	m.def("GiNaC_remove_mode_from_jacobian_or_hessian", [](const GiNaC::ex &expr, const GiNaC::ex &mode, const GiNaC::ex &flag)
		  { return 0 + pyoomph::expressions::remove_mode_from_jacobian_or_hessian(expr, mode, flag); });
	m.def(
		"GiNaC_double_dot", [](const GiNaC::ex &arg1, const GiNaC::ex &arg2)
		{ return 0 + pyoomph::expressions::double_dot(arg1, arg2); },
		"Calculates the double dot product A:B");

	m.def(
		"GiNaC_contract", [](const GiNaC::ex &arg1, const GiNaC::ex &arg2)
		{ return 0 + pyoomph::expressions::contract(arg1, arg2); },
		"Calculates the dot for vectors and double dot for matrices");
	m.def(
		"GiNaC_weak", [](const GiNaC::ex &arg1, const GiNaC::ex &arg2, const GiNaC::ex &flags, const GiNaC::ex &coordsys)
		{ return 0 + pyoomph::expressions::weak(arg1, arg2, flags, coordsys); },
		"(a,b) for weak forms, i.e. integral a*b*dx with flags&1 means lagrangian, flags&2 means dimensional");
	m.def(
		"GiNaC_subexpression", [](const GiNaC::ex &arg1)
		{ return 0 + pyoomph::expressions::subexpression(arg1); },
		"Creates a subexpression");
	m.def(
		"GiNaC_transpose", [](const GiNaC::ex &arg1)
		{ return 0 + pyoomph::expressions::transpose(arg1); },
		"Calculates the transposed matrix");
	m.def(
		"GiNaC_trace", [](const GiNaC::ex &arg1)
		{ return 0 + pyoomph::expressions::trace(arg1); },
		"Calculates the trace of a matrix");
	m.def(
		"GiNaC_determinant", [](const GiNaC::ex &arg,const GiNaC::ex &n)
		{ return 0 + pyoomph::expressions::determinant(arg,n); },
		"Calculates the determinant of an n x n matrix");
	m.def(
		"GiNaC_inverse_matrix", [](const GiNaC::ex &arg,const GiNaC::ex &n,const GiNaC::ex & flags)
		{ return 0 + pyoomph::expressions::inverse_matrix(arg,n,flags); },
		"Calculates the inverse of an n x n matrix");

	m.def(
		"GiNaC_minimize_functional_derivative", [](const GiNaC::ex &F,const std::vector<GiNaC::ex> &only_wrto,const GiNaC::ex &flags,const GiNaC::ex &coordsys)		
		{ return 0 + pyoomph::expressions::minimize_functional_derivative(F,GiNaC::lst(only_wrto.begin(),only_wrto.end()),flags,coordsys); },		
		"Calculates weak formulation of the variation of the functional with integrant F");		
	
	m.def(
		"GiNaC_testfunction", [](const std::string &id, pyoomph::FiniteElementCode *code, std::vector<std::string> tags)
		{
  	  if (!pyoomph::__field_name_cache.count(id)) pyoomph::__field_name_cache.insert(std::make_pair(id,GiNaC::realsymbol(id)));
  	  	  		GiNaC::GiNaCPlaceHolderResolveInfo ri(pyoomph::PlaceHolderResolveInfo(code,tags));
  return 0+pyoomph::expressions::testfunction(pyoomph::__field_name_cache[id],ri); },
		"Symbol which is expanded to the test function of the passed field.");

	m.def(
		"GiNaC_dimtestfunction", [](const std::string &id, pyoomph::FiniteElementCode *code, std::vector<std::string> tags)
		{
  	  if (!pyoomph::__field_name_cache.count(id)) pyoomph::__field_name_cache.insert(std::make_pair(id,GiNaC::realsymbol(id)));
  	  	  		GiNaC::GiNaCPlaceHolderResolveInfo ri(pyoomph::PlaceHolderResolveInfo(code,tags));
  return 0+pyoomph::expressions::dimtestfunction(pyoomph::__field_name_cache[id],ri); },
		"Symbol which is expanded to the test function of the passed field.");

	m.def(
		"GiNaC_testfunction_from_var", [](GiNaC::ex var_or_nondim, bool dimensional)
		{
  	  if (!is_ex_the_function(var_or_nondim,pyoomph::expressions::nondimfield) && !is_ex_the_function(var_or_nondim,pyoomph::expressions::field))
  	  {
  	   throw_runtime_error("Can only be called with var or nondim");
  	  } 
 		GiNaC::GiNaCPlaceHolderResolveInfo ri=GiNaC::ex_to<GiNaC::GiNaCPlaceHolderResolveInfo>(var_or_nondim.op(1));
 		if (dimensional)
 		{
        return 0+pyoomph::expressions::dimtestfunction(var_or_nondim.op(0),ri);
      }
      else
      {
      return 0+pyoomph::expressions::testfunction(var_or_nondim.op(0),ri);
      } },
		"Symbol which is expanded to the test function of the passed field.");

	m.def(
		"GiNaC_dimtestfunction_from_var", [](GiNaC::ex var_or_nondim)
		{
  	  if (!is_ex_the_function(var_or_nondim,pyoomph::expressions::nondimfield) && !is_ex_the_function(var_or_nondim,pyoomph::expressions::field))
  	  {
  	   throw_runtime_error("Can only be called with var or nondim");
  	  } 
 		GiNaC::GiNaCPlaceHolderResolveInfo ri=GiNaC::ex_to<GiNaC::GiNaCPlaceHolderResolveInfo>(var_or_nondim.op(1));
  return 0+pyoomph::expressions::dimtestfunction(var_or_nondim.op(0),ri); },
		"Symbol which is expanded to the test function of the passed field.");

	m.def(
		"GiNaC_scale", [&](const std::string &id, pyoomph::FiniteElementCode *code, std::vector<std::string> tags)
		{
	  if (!pyoomph::__field_name_cache.count(id)) pyoomph::__field_name_cache.insert(std::make_pair(id,GiNaC::realsymbol(id)));
	  	  		GiNaC::GiNaCPlaceHolderResolveInfo ri(pyoomph::PlaceHolderResolveInfo(code,tags));
		return 0+pyoomph::expressions::scale(pyoomph::__field_name_cache[id],ri); },
		"Expands to the scale of this field");

	m.def(
		"GiNaC_testscale", [&](const std::string &id, pyoomph::FiniteElementCode *code, std::vector<std::string> tags)
		{
	  if (!pyoomph::__field_name_cache.count(id)) pyoomph::__field_name_cache.insert(std::make_pair(id,GiNaC::realsymbol(id)));
	  	  		GiNaC::GiNaCPlaceHolderResolveInfo ri(pyoomph::PlaceHolderResolveInfo(code,tags));
		return 0+pyoomph::expressions::test_scale(pyoomph::__field_name_cache[id],ri); },
		"Expands to the scale of the test function");

	m.def(
		"GiNaC_field", [&](const std::string &id, pyoomph::FiniteElementCode *code, const std::vector<std::string> tags)
		{
	  if (!pyoomph::__field_name_cache.count(id)) pyoomph::__field_name_cache.insert(std::make_pair(id,GiNaC::realsymbol(id)));
	  	  		GiNaC::GiNaCPlaceHolderResolveInfo ri(pyoomph::PlaceHolderResolveInfo(code,tags));
		return 0+pyoomph::expressions::field(pyoomph::__field_name_cache[id],ri); },
		"Create a placeholder for a field, used for e.g. properties. Considers scaling");

	m.def(
		"GiNaC_EvalFlag", [&](const std::string &which)
		{
	   if (!pyoomph::__field_name_cache.count(which)) pyoomph::__field_name_cache.insert(std::make_pair(which,GiNaC::realsymbol(which)));
  		return 0+pyoomph::expressions::eval_flag(pyoomph::__field_name_cache[which]); },
		"Evaluate a flag at runtime (e.g. moving_mesh->0,1) or similar to activate or deactivate terms based on this");
	m.def(
		"GiNaC_nondimfield", [&](const std::string &id, pyoomph::FiniteElementCode *code, const std::vector<std::string> tags)
		{
	  if (!pyoomph::__field_name_cache.count(id)) pyoomph::__field_name_cache.insert(std::make_pair(id,GiNaC::realsymbol(id)));
	  		GiNaC::GiNaCPlaceHolderResolveInfo ri(pyoomph::PlaceHolderResolveInfo(code,tags));
			return 0+pyoomph::expressions::nondimfield(pyoomph::__field_name_cache[id],ri); },
		"Create a placeholder for a non-dimensiona field. Opposed to 'field', dimensions are not considerd");
	m.def(
		"GiNaC_eval_in_domain", [&](GiNaC::ex expr, pyoomph::FiniteElementCode *code, const std::vector<std::string> tags)
		{
  		GiNaC::GiNaCPlaceHolderResolveInfo ri(pyoomph::PlaceHolderResolveInfo(code,tags));
		return 0+pyoomph::expressions::eval_in_domain(expr,ri); },
		"Expand vars and nondims in a particular domain");
	m.def(
		"GiNaC_eval_in_past", [&](GiNaC::ex expr, GiNaC::ex offset, GiNaC::ex tstep_action)
		{ return 0 + pyoomph::expressions::eval_in_past(expr, offset, tstep_action); },
		"Expand vars and nondims in a particular domain");
	m.def(
		"GiNaC_eval_at_expansion_mode", [&](GiNaC::ex expr, GiNaC::ex index)
		{ return 0 + pyoomph::expressions::eval_at_expansion_mode(expr, index); },
		"Set the mode index (base or azimuthal mode) for  vars and nondims");

	m.def(
		"GiNaC_internal_function_with_element_arg", [](const std::string &id, const std::vector<GiNaC::ex> &args)
		{ return 0 + pyoomph::expressions::internal_function_with_element_arg(GiNaC::realsymbol(id), GiNaC::lst(args.begin(), args.end())); },
		"Internal functions, used e.g. for elemental functions like element_size etc");
	m.def("GiNaC_vector_dim", []()
		  { return pyoomph::the_vector_dim; });
	m.def(
		"GiNaC_unit_matrix", [](const int &dim)
		{ return 0 + GiNaC::unit_matrix((dim == -1 ? pyoomph::the_vector_dim : dim)); },
		"Creates the identity matrix");
	m.def("GiNaC_Vect", [](const std::vector<GiNaC::ex> &v)
		  { return 0 + GiNaC::matrix(v.size(), 1, GiNaC::lst(v.begin(), v.end())); });
	m.def(
		"GiNaC_delayed_expansion", [](std::function<GiNaC::ex()> func)
		{
	  pyoomph::DelayedPythonCallbackExpansion * cbexpr=new pyoomph::DelayedPythonCallbackExpansion(func);
	  pyoomph::DelayedPythonCallbackExpansionWrapper * wrapped=new pyoomph::DelayedPythonCallbackExpansionWrapper(cbexpr);
	  
	  return 0+GiNaC::GiNaCDelayedPythonCallbackExpansion(*wrapped); },
		py::keep_alive<0, 1>(), py::keep_alive<1, 0>());

	m.def("GiNaC_UnitVect", [](const unsigned &dir, const int &ndim, const int &flags, const GiNaC::ex &coordsys)
		  {
		if (coordsys.is_zero())
		{
		  return 0+pyoomph::expressions::unitvect(dir,ndim,pyoomph::__no_coordinate_system_wrapper,flags);
		}
    else  
		{
			return 0+pyoomph::expressions::unitvect(dir,ndim,coordsys,flags);
		} });
	m.def("GiNaC_Matrix", [](unsigned nd1, const std::vector<GiNaC::ex> &v)
		  { return 0 + GiNaC::matrix(v.size() / nd1, nd1, GiNaC::lst(v.begin(), v.end())); });
	m.def(
		"GiNaC_get_global_symbol", [](const std::string &n)
		{
			if (n == "t")
				return 0 + pyoomph::expressions::t;
			else if (n == "_dt_BDF1")
				return 0 + pyoomph::expressions::_dt_BDF1;
			else if (n == "_dt_BDF2")
				return 0 + pyoomph::expressions::_dt_BDF2;
			else if (n == "_dt_Newmark2")
				return 0 + pyoomph::expressions::_dt_Newmark2;
			else if (n == "x")
				return 0 + pyoomph::expressions::x;
			else if (n == "y")
				return 0 + pyoomph::expressions::y;
			else if (n == "z")
				return 0 + pyoomph::expressions::z;
			else if (n == "nx")
				return 0 + pyoomph::expressions::nx;
			else if (n == "ny")
				return 0 + pyoomph::expressions::ny;
			else if (n == "nz")
				return 0 + pyoomph::expressions::nz;
			else
			{
				throw_runtime_error("Global symbol '" + n + "' not defined");
				return GiNaC::ex(0);
			} },
		"Get the time 't', or coordinates 'x','y','z'");
	m.def("GiNaC_new_symbol", [](const std::string &name)
		  { return 0 + GiNaC::symbol(name); });
}
