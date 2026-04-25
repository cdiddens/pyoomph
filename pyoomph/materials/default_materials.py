#  @file
#  @author Christian Diddens <c.diddens@utwente.nl>
#  @author Duarte Rocha <d.rocha@utwente.nl>
#  
#  @section LICENSE
# 
#  pyoomph - a multi-physics finite element framework based on oomph-lib and GiNaC 
#  Copyright (C) 2021-2026  Christian Diddens & Duarte Rocha
# 
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
# 
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
# 
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <http://www.gnu.org/licenses/>. 
#
#  The authors may be contacted at c.diddens@utwente.nl and d.rocha@utwente.nl
#
# ========================================================================
 
from .generic import *
from ..expressions import *
from ..expressions.units import *
from .UNIFAC import *


@MaterialProperties.register()
class PureLiquidWater(PureLiquidProperties):
    name = "water"

    def __init__(self):
        super().__init__()
        # https://en.wikipedia.org/wiki/Water
        self.molar_mass = 18.01528 * gram / mol

        TKelvin = var("temperature") / kelvin

        # https://www.engineersedge.com/physics/water__density_viscosity_specific_weight_13146.htm
        self.dynamic_viscosity = subexpression(
            2.414e-5 * (10 ** (247.8 / (TKelvin - 140))) * pascal * second
        )

        # https://www.thecalculator.co/others/Water-Density-Calculator-629.html
        self.mass_density = subexpression(
            (
                1000
                * (
                    1.0
                    - (TKelvin + 15.7914)
                    / (508929 * (TKelvin - 205.02037))
                    * (TKelvin - 277.1363) ** 2
                )
            )
            * kilogram
            / meter**3
        )

        # https://en.wikipedia.org/wiki/E%C3%B6tv%C3%B6s_rule
        self.default_surface_tension["gas"] = subexpression(
            0.07275 * (1.0 - 0.002 * (TKelvin - 291.0)) * newton / meter
        )

        # https://en.wikipedia.org/wiki/Antoine_equation
        self.set_vapor_pressure_by_Antoine_coeffs(8.07131, 1730.63, 233.426)

        # For activity coefficients for mixtures, we also need the UNIFAC groups
        self.set_unifac_groups({"H2O": 1})  # Just one H2O group here

        # https://digitalcommons.memphis.edu/cgi/viewcontent.cgi?article=4393&context=etd
        self.thermal_conductivity = (
            (-752.249 + 7.43735 * TKelvin - 0.00967398 * TKelvin * TKelvin)
            * milli
            * watt
            / (meter * kelvin)
        )

        # https://www.engineeringtoolbox.com/water-thermal-properties-d_162.html
        self.specific_heat_capacity = 4.187 * kilo * joule / (kilogram * kelvin)

        # https://www.engineeringtoolbox.com/water-properties-d_1573.html
        # TODO: according to Wikipedia, it is 2257 J/g, but likely at T=100°C ???
        self.latent_heat_of_evaporation = 2437.7 * kilo * joule / kilogram


@MaterialProperties.register()
class PureLiquidGlycerol(PureLiquidProperties):
    name = "glycerol"

    def __init__(self):
        super().__init__()
        # https://en.wikipedia.org/wiki/Glycerol
        self.molar_mass = 92.094 * gram / mol

        self.dynamic_viscosity = 1 * pascal * second  ##TODO Correct ones here

        # https://en.wikipedia.org/wiki/Glycerol
        self.mass_density = 1261 * kilogram / meter**3

        # https://en.wikipedia.org/wiki/Glycerol_(data_page)
        self.default_surface_tension["gas"] = 63.4 * milli * newton / meter

        # https://www.matweb.com/search/datasheet.aspx?matguid=015b4c540c454ad7b944980dfa9438c8&ckck=1
        self.thermal_conductivity = 0.292 * watt / (meter * kelvin)

        # https://en.wikipedia.org/wiki/Glycerol_(data_page)
        self.specific_heat_capacity = 221.9 * joule / (mol * kelvin) / self.molar_mass

        # Different UNIFAC models have different decompositions of glycerol
        self.set_unifac_groups({"CH2": 2, "CH1": 1, "OH": 3}, only_for="Original")
        self.set_unifac_groups(
            {"CH2": 2, "CH1": 1, "OH (P)": 2, "OH (S)": 1}, only_for="Dortmund"
        )
        self.set_unifac_groups(
            {"CH2(hydroxy)": 2, "CH(hydroxy)": 1, "OH(new)": 3}, only_for="AIOMFAC"
        )


@MaterialProperties.register()
class PureLiquidEthanol(PureLiquidProperties):
    name = "ethanol"

    def __init__(self):
        super().__init__()
        # https://en.wikipedia.org/wiki/Ethanol
        self.molar_mass = 46.07 * gram / mol

        # https://en.wikipedia.org/wiki/Ethanol
        self.dynamic_viscosity = 1.2 * milli * pascal * second

        # https://en.wikipedia.org/wiki/Ethanol
        self.mass_density = 789.45 * kilogram / meter**3

        # http://ddbonline.ddbst.de/DIPPR106SFTCalculation/DIPPR106SFTCalculationCGI.exe?component=Ethanol
        A = 131.38
        B = 5.5437
        C = -8.4826
        D = 4.3164
        Tc = 516.2
        TK = var("temperature") / kelvin
        Tr = TK / Tc
        sigma = A * maximum(0.00000000001, 1 - Tr) ** maximum(
            0.00001, B + C * Tr + D * Tr**2
        )
        self.default_surface_tension["gas"] = sigma * milli * newton / meter

        # https://en.wikipedia.org/wiki/Antoine_equation
        self.set_vapor_pressure_by_Antoine_coeffs(8.20417, 1642.89, 230.300)

        # https://www.engineeringtoolbox.com/fluids-evaporation-latent-heat-d_147.html
        self.latent_heat_of_evaporation = 846 * kilo * joule / kilogram

        self.set_unifac_groups({"CH3": 1, "CH2": 1, "OH": 1}, only_for="Original")
        self.set_unifac_groups({"CH3": 1, "CH2": 1, "OH (P)": 1}, only_for="Dortmund")
        self.set_unifac_groups(
            {"CH3(long-chain)": 1, "CH2(hydroxy)": 1, "OH(new)": 1}, only_for="AIOMFAC"
        )
        
        # https://www.engineeringtoolbox.com/ethanol-ethyl-alcohol-properties-C2H6O-d_2027.html
        self.thermal_conductivity=0.167*watt/(meter*kelvin)
        self.specific_heat_capacity=2.57*kilo*joule/(kilogram*kelvin)


@MaterialProperties.register()
class PureLiquid12Hexanediol(PureLiquidProperties):
    name = "12hexanediol"  # we may not use a comma or the dash here

    def __init__(self):
        super().__init__()
        # https://www.sigmaaldrich.com/DE/de/product/aldrich/213691
        self.molar_mass = 118.17 * gram / mol
        self.mass_density = 951 * kilogram / (meter**3)

        # https://dx.doi.org/10.1515/zna-2004-0905
        self.dynamic_viscosity = 0.080 * pascal * second

        self.set_unifac_groups(
            {"CH3": 1, "CH2": 4, "CH": 1, "OH": 2}, only_for="Original"
        )
        self.set_unifac_groups(
            {"CH3": 1, "CH2": 4, "CH": 1, "OH (P)": 1, "OH (S)": 1}, only_for="Dortmund"
        )
        self.set_unifac_groups(
            {
                "CH3(alk)": 1,
                "CH2(alk)": 3,
                "CH2(hydroxy)": 1,
                "CH(hydroxy)": 1,
                "OH(new)": 2,
            },
            only_for="AIOMFAC",
        )


#### Liquid mixtures ################################


@MaterialProperties.register()
class MixtureLiquidGlycerolWater(MixtureLiquidProperties):
    components = {"water", "glycerol"}
    passive_field = "water"

    def __init__(self, pure_properties: Dict[str, MaterialProperties]):
        super().__init__(pure_properties)

        # Just assumptions, for mass density reasonable according to the data from https://doi.org/10.1016/j.petrol.2012.09.003
        self.set_by_weighted_average("mass_density")
        self.set_by_weighted_average("thermal_conductivity")
        self.set_by_weighted_average("specific_heat_capacity")

        yG = self.get_mass_fraction_field("glycerol")

        TCelsius = subexpression(var("temperature") / kelvin - 273.15)

        # https://dx.doi.org/10.1021/ie071349z
        a = 0.705 - 0.0017 * TCelsius
        b = (4.9 + 0.036 * TCelsius) * a**2.5
        muG = 12100 * exp((-1233 + TCelsius) * TCelsius / (9900 + 70 * TCelsius))
        muW = 1.790 * exp((-1230 - TCelsius) * TCelsius / (36100 + 360 * TCelsius))
        # alpha = subexpression(1 - yG + a * b * yG * (1 - yG) / (a * yG + b * (1 - yG)))
        # self.dynamic_viscosity= subexpression((muW) ** alpha * ((muG) ** (1.0 - alpha))* 0.001*pascal * second)
        alpha = subexpression(1 - yG + a * b * yG * (1 - yG) / (a * yG + b * (1 - yG)))
        self.dynamic_viscosity = subexpression(
            muW * (muG / muW) ** (1 - alpha) * 0.001 * pascal * second
        )

        # Fit of data from https://doi.org/10.1016/j.petrol.2012.09.003
        #self.default_surface_tension["gas"] = subexpression(
        #    72.45e-3
        #    * (
        #        (
        #            1.0
        #            - 0.1214690683 * yG
        #            + 0.4874796412 * yG**2
        #            - 2.208295376 * yG**3
        #            + 3.412242927 * yG**4
        #            - 1.698619738 * yG**5
        #        )
        #        - (0.0001455 * (1 - yG) + 0.00008845 * yG) * (TCelsius)
        #    )
        #    * newton
        #    / meter
        #)
                
        T_minus_T0_in_K=(var("temperature")-25*celsius)/kelvin
        self.default_surface_tension["gas"]=subexpression((72.45
            * (
                (
                    1.0
                    - 0.1214690683 * yG
                    + 0.4874796412 * yG**2
                    - 2.208295376 * yG**3
                    + 3.412242927 * yG**4
                    - 1.698619738 * yG**5
                )
               
            )            
         - (72.75 * 0.002 * (1 - yG) + 0.0598 * yG) * (T_minus_T0_in_K) ) *milli*newton/meter)

        # Fit of data from https://doi.org/10.1021/je049917u
        D = (
            1.024e-11
            * (-0.721 * yG + 0.7368)
            / (0.49311e-2 * yG + 0.7368e-2)
            * meter**2
            / second
        )
        self.set_diffusion_coefficient(D)

        self.set_activity_coefficients_by_unifac("AIOMFAC")
        
    
        
        


@MaterialProperties.register()
class MixtureLiquidEthanolWater(MixtureLiquidProperties):
    components = {"water", "ethanol"}
    passive_field = "water"

    def __init__(self, pure_properties: Dict[str, MaterialProperties]):
        super().__init__(pure_properties)
        TKelvin = var("temperature") / kelvin
        yE = self.get_mass_fraction_field("ethanol")
        se: Callable[[Expression], Expression] = lambda x: subexpression(x)

        def clamp_0_1(x: Expression) -> Expression:
            return heaviside(x) * x - heaviside(x - 1) * (x - 1)

        # def smooth_clamp(x:Expression,mini:ExpressionOrNum,maxi:ExpressionOrNum):
        # x=subexpression(clamp_0_1(subexpression((x-mini)/(maxi-mini))))
        # return x*x*x*(x*(x*6-15)+10)

        yE = se(clamp_0_1(yE))
        # yE = clamp_0_1(yE)

        # Fit from data given here: http://dx.doi.org/10.1016/j.jct.2007.05.004
        self.dynamic_viscosity = se(
            (
                (0.000834378 - 1.77674e-5 * (TKelvin - 298.15))
                + (0.00670473 - 0.00030053 * (TKelvin - 298.15)) * yE
                + (-0.00884472 + 0.000451734 * (TKelvin - 298.15)) * yE**2
                + (0.00237477 - 0.000152765 * (TKelvin - 298.15)) * yE**3
            )
            * pascal
            * second
        )
        self.mass_density = se(
            (997.0479 + (789.0 - 997.0479) * (0.65951 * yE + (1.0 - 0.65951) * yE**2))
            * kilogram
            / (meter**3)
        )

        # Fit from data given here: http://dx.doi.org/10.1039/c3cp43785j
        self.set_diffusion_coefficient(
            se(1.25477e-9 * (1.0 - 2.7794 * yE + 2.72277 * yE**2) * meter**2 / second)
        )

        # Fit from data given here: http://dx.doi.org/10.1021/je00019a016
        self.default_surface_tension["gas"] = se(
            (
                (0.016045 - 6.43279e-5 * (TKelvin - 298.15))
                + (0.730407 - 0.00189571 * (TKelvin - 298.15))
                / (100.0 * yE + (13.1225 - 0.011996 * (TKelvin - 298.15)))
            )
            * newton
            / meter
        )

        # http://dx.doi.org/10.1016/0378-3812(81)85011-X
        yW = 1 - yE
        self.specific_heat_capacity = (
            se(
                2444.61
                + 3369.15 * yW
                - 3136.31 * yW**2
                + 6520.2 * yW**3
                - 5136.02 * yW**4
            )
            * joule
            / (kilogram * kelvin)
        )
        # http://dx.doi.org/10.1016/0301-0104(88)87161-1
        self.thermal_conductivity = (
            se(0.584425 - 0.599043 * yE + 0.18041 * yE**2) * watt / (meter * kelvin)
        )

        self.set_activity_coefficients_by_unifac("AIOMFAC")


@MaterialProperties.register()
class MixtureLiquid12HexanediolWater(MixtureLiquidProperties):
    components = {"12hexanediol", "water"}
    passive_field = "water"

    def __init__(self, pure_properties: Dict[str, MaterialProperties]):
        super().__init__(pure_properties)
        yH = self.get_mass_fraction_field("12hexanediol")

        # Fit of the data from https://doi.org/10.1515/zna-2004-0905
        self.dynamic_viscosity = subexpression(
            1.002
            * exp(4.24996 * yH - 0.224339 * yH**2 - 3.45588 * yH**3 + 3.84868 * yH**4)
            * milli
            * pascal
            * second
        )

        # Using Stokes-Einstein assumption. We do not have the data for this mixture
        self.set_diffusion_coefficient(
            subexpression(
                1.37853e-09
                / (
                    exp(
                        4.24996 * yH
                        - 0.224339 * yH**2
                        - 3.45588 * yH**3
                        + 3.84868 * yH**4
                    )
                )
                * meter**2
                / second
            )
        )

        # Just an assumption:
        self.set_by_weighted_average("mass_density", fraction_type="mass_fraction")

        # Fit of the data from https://doi.org/10.1016/j.fluid.2007.05.029
        # XXX Warning: This is a monotonic fit, we know it is not!
        self.default_surface_tension["gas"] = subexpression(
            (23.8436 + 48.6529 / (1 + 89.9309 * yH + 758.768 * yH**2))
            * milli
            * newton
            / meter
        )

        self.set_activity_coefficients_by_unifac("AIOMFAC")


    def use_interpolated_experimental_surface_tension(self,pendant_drop_method:bool=True,C_interpolator:bool=True,spline_order:int=3):
        from pyoomph.expressions.interpol import InterpolateSpline1d,CSplineInterpolator
        import pyoomph
        data_file= os.path.join(os.path.dirname(__file__),"csv_data","surftens_water_12hexanediol_pendant.csv" if pendant_drop_method else "surftens_water_12hexanediol_ring.csv")
        inter=CSplineInterpolator(data_file,k=spline_order) if C_interpolator else InterpolateSpline1d(data_file,k=spline_order)
        wHD=var("massfrac_12hexanediol")        
        self.default_surface_tension["gas"]=inter(wHD)*milli*newton/meter

############## GASES ###################


@MaterialProperties.register()
class PureGasAir(PureGasProperties):
    name = "air"

    def __init__(self):
        super().__init__()
        # https://www.engineeringtoolbox.com/molecular-mass-air-d_679.html
        self.molar_mass = 28.9647 * gram / mol

        TKelvin = var("temperature") / kelvin

        self.set_mass_density_from_ideal_gas_law()

        # Fit from data at https://www.engineeringtoolbox.com/dry-air-properties-d_973.html
        self.dynamic_viscosity = (
            (0.0409424 + 0.00725803 * TKelvin - 4.12727e-06 * (TKelvin) ** 2)
            * 1e-5
            * pascal
            * second
        )
        self.thermal_conductivity = (
            (-0.0217506 + 0.00984373 * TKelvin - 3.4318e-06 * TKelvin * TKelvin)
            * 1e-5
            * kilo
            * watt
            / (meter * kelvin)
        )
        self.specific_heat_capacity = 1.005 * kilo * joule / (kilogram * kelvin)

        # https://dx.doi.org/10.5194/acp-14-9233-2014
        self.diffusion_volume_for_Fuller_eq = 19.7


@MaterialProperties.register()
class PureGasWater(PureGasProperties):
    name = "water"

    def __init__(self):
        super().__init__()
        # https://en.wikipedia.org/wiki/Water
        self.molar_mass = 18.01528 * gram / mol

        # Fit from data at https://en.wikipedia.org/wiki/List_of_viscosities
        TKelvin = var("temperature") / kelvin
        self.dynamic_viscosity = (
            (0.0409424 + 0.00725803 * TKelvin - 4.12727e-06 * (TKelvin) ** 2)
            * 1e-5
            * pascal
            * second
        )  # TODO

        self.set_mass_density_from_ideal_gas_law()

        # https://www.engineeringtoolbox.com/water-liquid-gas-thermal-conductivity-temperature-pressure-d_2012.html
        self.thermal_conductivity = (
            24.57 * milli * watt / (meter * kelvin)
        )  # TODO: This is at T = 100

        # https://www.engineeringtoolbox.com/water-vapor-d_979.html
        self.specific_heat_capacity = (
            1.864 * joule / (gram * kelvin)
        )  ##TODO: This is at T = 100 C

        # https://dx.doi.org/10.5194/acp-15-5585-2015
        self.diffusion_volume_for_Fuller_eq = 13.1


@MaterialProperties.register()
class MixtureGasAirWater(MixtureGasProperties):
    components = {"water", "air"}
    passive_field = "air"

    def __init__(self, pure_properties: Dict[str, MaterialProperties]):
        super().__init__(pure_properties)
        TKelvin = var("temperature") / kelvin

        # TODO: Improve: This is the one for air only
        self.dynamic_viscosity = self.pure_properties["air"].dynamic_viscosity
        self.thermal_conductivity = self.pure_properties["air"].thermal_conductivity
        self.specific_heat_capacity = self.pure_properties["air"].specific_heat_capacity

        # Mason, Edward Allen, and MALINAUSKAS AP. "Gas transportin porous media: the dusty-gas model." (1983).
        self.set_diffusion_coefficient(
            1.87e-10 * maximum(TKelvin, 0.0) ** 2.072 * meter**2 / second
        )

        self.set_mass_density_from_ideal_gas_law()


@MaterialProperties.register()
class PureGasEthanol(PureGasProperties):
    name = "ethanol"

    def __init__(self):
        super().__init__()
        # https://en.wikipedia.org/wiki/Ethanol
        self.molar_mass = 0.4607e-01 * kilogram / mol

        # Just an assumption
        self.dynamic_viscosity = 5 * micro * pascal * second  # TODO
        self.set_mass_density_from_ideal_gas_law()

        self.thermal_conductivity = 0.02356 * watt / (meter * kelvin)
        # https://www.engineeringtoolbox.com/specific-heat-capacity-ethanol-Cp-Cv-isobaric-isochoric-ethyl-alcohol-d_2030.html
        self.specific_heat_capacity = (
            1.804 * kilo * joule / (kilogram * kelvin)
        )  # TODO: This is at T = 351.1 K


@MaterialProperties.register()
class MixtureGasAirEthanolWater(MixtureGasProperties):
    components = {"ethanol", "water", "air"}
    passive_field = "air"

    def __init__(self, pure_properties: Dict[str, MaterialProperties]):
        super().__init__(pure_properties)
        TKelvin = var("temperature") / kelvin
        pAtm = var("absolute_pressure") / atm

        # TODO: Improve: This is the one for air only
        self.dynamic_viscosity = self.pure_properties["air"].dynamic_viscosity

        # Mason, Edward Allen, and MALINAUSKAS AP. "Gas transportin porous media: the dusty-gas model." (1983).
        self.set_diffusion_coefficient(
            "water", 1.87e-10 * TKelvin**2.072 / pAtm * meter**2 / second
        )

        # https://dx.doi.org/10.1021/ie50539a046
        self.set_diffusion_coefficient(
            "ethanol", 0.135e-04 * meter**2 / second
        )  ###TODO Temperature dep of this

        self.set_mass_density_from_ideal_gas_law()
        self.set_by_weighted_average("specific_heat_capacity")
        self.set_by_weighted_average("thermal_conductivity")



@MaterialProperties.register()
class MixtureGasAirEthanol(MixtureGasProperties):
    components = {"ethanol", "air"}
    passive_field = "air"

    def __init__(self, pure_properties: Dict[str, MaterialProperties]):
        super().__init__(pure_properties)        

        # TODO: Improve: This is the one for air only
        self.dynamic_viscosity = self.pure_properties["air"].dynamic_viscosity

        # https://dx.doi.org/10.1021/ie50539a046
        self.set_diffusion_coefficient(
            "ethanol", 0.135e-04 * meter**2 / second
        )  ###TODO Temperature dep of this

        self.set_mass_density_from_ideal_gas_law()
        self.set_by_weighted_average("specific_heat_capacity")
        self.set_by_weighted_average("thermal_conductivity")

############## SOLIDS ###################


@MaterialProperties.register()
class PureSolidAluminium(PureSolidProperties):
    name = "aluminium"

    def __init__(self):
        super().__init__()
        # https://en.wikipedia.org/wiki/Aluminium
        self.molar_mass = 26.98 * gram / mol
        self.heating_capacity = 24.20 * joule / (mol * kelvin) / self.molar_mass
        self.thermal_conductivity = 237 * watt / (meter * kelvin)
        self.mass_density = 2.699 * gram / (centi * meter) ** 3


@MaterialProperties.register()
class PureSolidPaper(PureSolidProperties):
    name = "paper"

    def __init__(self):
        super().__init__()
        self.molar_mass = 26.98 * gram / mol  # Makes not sense at all
        # https://www.aqua-calc.com/calculate/volume-to-weight/substance/paper-coma-and-blank-standard
        self.mass_density = 1201 * kilogram / meter**3
        # https://sciencing.com/thermal-properties-paper-6893512.html
        self.thermal_conductivity = 0.05 * watt / (meter * kelvin)
        self.specific_heat_capacity = 1.4 * kilo * joule / (kilogram * kelvin)


@MaterialProperties.register()
class PureSolidBorosilicate(PureSolidProperties):
    name = "borosilicate"

    def __init__(self):
        super().__init__()
        self.molar_mass = 122.0632 * gram / mol  ##TODO: correct value
        # https://en.wikipedia.org/wiki/Borosilicate_glass
        self.mass_density = 2230.0 * kilogram / (meter**3)        
        self.specific_heat_capacity = 830 * joule / (kilogram * kelvin)
        # https://www.design1st.com/Design-Resource-Library/engineering_data/MaterialPropertiesTables.pdf
        self.thermal_conductivity = 1.13 * watt / (meter * kelvin)

@MaterialProperties.register()
class PureSolidStainlessSteel(PureSolidProperties):
    name="stainless_steel"
    def __init__(self):
        super().__init__()
        self.molar_mass = 55.845 * gram / mol
        self.mass_density = 7.8 * gram / (centi*meter)**3
        self.thermal_conductivity=16.3*watt/(meter*kelvin)
        self.specific_heat_capacity=0.5*kilo*joule/(kilogram*kelvin)