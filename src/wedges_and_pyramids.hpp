#pragma once

#include "oomph_lib.hpp"
#include "exception.hpp"
// Wedges and pyramids are not supported by oomph-lib, but the basic Element classes are defined in the oomph namespace here,
// they are adjusted for pyoomph in the elements.{cpp,hpp} files.
namespace oomph
{

// Gauss-Legendre quadrature for the C1 wedge element, with 6 points (2 in each direction)
class WedgeGaussC1 : public Integral
  {
  private:    
    static const unsigned Npts = 6;    
    static const double Knot[6][3], Weight[6];
  public:    
    WedgeGaussC1(){};
    WedgeGaussC1(const WedgeGaussC1& dummy) = delete;
    void operator=(const WedgeGaussC1&) = delete;
    unsigned nweight() const    {   return Npts;   }
    double knot(const unsigned& i, const unsigned& j) const   {    return Knot[i][j];  }
    double weight(const unsigned& i) const   {   return Weight[i];   }
 };

 // Gauss-Legendre quadrature for the C2 wedge element, with 18 points (3 in each direction)
class WedgeGaussC2 : public Integral
{
private:
    static const unsigned Npts = 18;
    static const double Knot[18][3], Weight[18];

public:
    WedgeGaussC2() {}
    WedgeGaussC2(const WedgeGaussC2&) = delete;
    void operator=(const WedgeGaussC2&) = delete;

    unsigned nweight() const { return Npts; }
    double knot(const unsigned& i, const unsigned& j) const { return Knot[i][j]; }
    double weight(const unsigned& i) const { return Weight[i]; }
};

// Gauss-Legendre quadrature for the C1 pyramid element, with 5 points (2 in each direction)

class PyramidGaussC1 : public Integral
  {
  private:    
    static const unsigned Npts = 12;
    static const double Knot[12][3], Weight[12];
  public:    
    PyramidGaussC1(){};
    PyramidGaussC1(const PyramidGaussC1& dummy) = delete;
    void operator=(const PyramidGaussC1&) = delete;
    unsigned nweight() const    {   return Npts;   }
    double knot(const unsigned& i, const unsigned& j) const   {    return Knot[i][j];  }
    double weight(const unsigned& i) const   {   return Weight[i];   }
 };

  // Gauss-Legendre quadrature for the C2 pyramid element, with 14 points (3 in each direction)
class PyramidGaussC2 : public Integral
{
private:
    static const unsigned Npts = 27;
    static const double Knot[27][3], Weight[27];

public:
    PyramidGaussC2() {}
    PyramidGaussC2(const PyramidGaussC2&) = delete;
    void operator=(const PyramidGaussC2&) = delete;

    unsigned nweight() const { return Npts; }
    double knot(const unsigned& i, const unsigned& j) const { return Knot[i][j]; }
    double weight(const unsigned& i) const { return Weight[i]; }
};

class WedgeElementShapeC1
{
 public:
    static void shape(const Vector<double>& s, Shape& psi) 
    {
        const double s0 = s[0];
        const double s1 = s[1];
        const double s2 = s[2];
        const double l1 = 1.0 - s0 - s1;
        psi[0] = l1 * (1.0 - s2);
        psi[1] = s0 * (1.0 - s2);
        psi[2] = s1 * (1.0 - s2);
        psi[3] = l1 * s2;
        psi[4] = s0 * s2;
        psi[5] = s1 * s2;
    }

    
    static void dshape_local(const Vector<double>& s,Shape& psi,DShape& dpsids) 
    {
        const double s0 = s[0];
        const double s1 = s[1];
        const double s2 = s[2];
        const double l1 = 1.0 - s0 - s1;

        // Shape functions
        psi[0] = l1 * (1.0 - s2);
        psi[1] = s0 * (1.0 - s2);
        psi[2] = s1 * (1.0 - s2);

        psi[3] = l1 * s2;
        psi[4] = s0 * s2;
        psi[5] = s1 * s2;

        // Derivatives wrt s0
        dpsids(0,0) = -(1.0 - s2);
        dpsids(1,0) =  (1.0 - s2);
        dpsids(2,0) =  0.0;
        dpsids(3,0) = -s2;
        dpsids(4,0) =  s2;
        dpsids(5,0) =  0.0;

        // Derivatives wrt s1
        dpsids(0,1) = -(1.0 - s2);
        dpsids(1,1) =  0.0;
        dpsids(2,1) =  (1.0 - s2);
        dpsids(3,1) = -s2;
        dpsids(4,1) =  0.0;
        dpsids(5,1) =  s2;

        // Derivatives wrt s2
        dpsids(0,2) = -l1;
        dpsids(1,2) = -s0;
        dpsids(2,2) = -s1;
        dpsids(3,2) =  l1;
        dpsids(4,2) =  s0;
        dpsids(5,2) =  s1;
    }
};




class PyramidElementShapeC1
{
 public:
    static void shape(const Vector<double>& s, Shape& psi) 
    {
        const double s0 = s[0];
        const double s1 = s[1];
        const double s2 = s[2];

        const double w  = 1.0 - s2;
        const double iw = 1.0 / w;

        const double w_s0 = w - s0;
        const double w_s1 = w - s1;

        // shape functions
        psi[0] = w_s0 * w_s1 * iw;
        psi[1] = s0    * w_s1 * iw;
        psi[2] = s0    * s1  * iw;
        psi[3] = w_s0  * s1  * iw;
        psi[4] = s2;
    }


    
    static void dshape_local(const Vector<double>& s,Shape& psi,DShape& dpsids) 
    {
        const double s0 = s[0];
        const double s1 = s[1];
        const double s2 = s[2];

        const double w  = 1.0 - s2;
        const double iw = 1.0 / w;
        const double iw2 = iw * iw;

        const double w_s0 = w - s0;
        const double w_s1 = w - s1;

        psi[0] = w_s0 * w_s1 * iw;
        psi[1] = s0   * w_s1 * iw;
        psi[2] = s0   * s1   * iw;
        psi[3] = w_s0 * s1   * iw;
        psi[4] = s2;

        // derivatives wrt s0
        dpsids(0,0) = -(w_s1) * iw;
        dpsids(1,0) =  (w_s1) * iw;
        dpsids(2,0) =  s1     * iw;
        dpsids(3,0) = -s1     * iw;
        dpsids(4,0) = 0.0;

        // derivatives wrt s1
        dpsids(0,1) = -(w_s0) * iw;
        dpsids(1,1) = -s0     * iw;
        dpsids(2,1) =  s0     * iw;
        dpsids(3,1) =  w_s0   * iw;
        dpsids(4,1) = 0.0;


        // derivatives wrt s2
        const double s0s1 = s0 * s1;

        dpsids(0,2) = s0s1 * iw2 - 1.0;
        dpsids(1,2) = - s0s1 * iw2;
        dpsids(2,2) = s0s1 * iw2;
        dpsids(3,2) = - s0s1 * iw2;
        dpsids(4,2) = 1.0;
    }
};

class WedgeElementShapeC2
{
 public:
    // ---------------------------------------------------------------
    // shape
    // ---------------------------------------------------------------
    // Evaluate all 18 shape functions at local coordinate s=(s0,s1,s2).
    //
    //   psi[layer*6 + k] = T_k(s0,s1) * L_layer(s2)
    //
    // Triangle bases  (lambda = 1 - s0 - s1):
    //   T0 = lambda*(2*lambda - 1)
    //   T1 = s0*(2*s0 - 1)
    //   T2 = s1*(2*s1 - 1)
    //   T3 = 4*s0*lambda
    //   T4 = 4*s1*lambda
    //   T5 = 4*s0*s1
    //
    // 1-D Lagrange bases on [0,1]:
    //   L0 = (1-s2)*(1-2*s2)
    //   L1 = 4*s2*(1-s2)
    //   L2 = s2*(2*s2-1)
    // ---------------------------------------------------------------
    static void shape(const Vector<double>& s, Shape& psi) 
    {
        const double s0 = s[0];
        const double s1 = s[1];
        const double s2 = s[2];
        const double l  = 1.0 - s0 - s1;   // barycentric lambda

        // Quadratic triangle bases
        const double T0 = l  * (2.0*l  - 1.0);
        const double T1 = s0 * (2.0*s0 - 1.0);
        const double T2 = s1 * (2.0*s1 - 1.0);
        const double T3 = 4.0 * s0 * l;
        const double T4 = 4.0 * s1 * l;
        const double T5 = 4.0 * s0 * s1;

        // Quadratic 1-D Lagrange bases along s2
        const double L0 = (1.0 - s2) * (1.0 - 2.0*s2);
        const double L1 = 4.0 * s2 * (1.0 - s2);
        const double L2 = s2 * (2.0*s2 - 1.0);

        // Layer s2 = 0  (nodes 0-5)
        psi[0]  = T0 * L0;
        psi[1]  = T1 * L0;
        psi[2]  = T2 * L0;
        psi[3]  = T3 * L0;
        psi[4]  = T4 * L0;
        psi[5]  = T5 * L0;

        // Layer s2 = 1/2  (nodes 6-11)
        psi[6]  = T0 * L1;
        psi[7]  = T1 * L1;
        psi[8]  = T2 * L1;
        psi[9]  = T3 * L1;
        psi[10] = T4 * L1;
        psi[11] = T5 * L1;

        // Layer s2 = 1  (nodes 12-17)
        psi[12] = T0 * L2;
        psi[13] = T1 * L2;
        psi[14] = T2 * L2;
        psi[15] = T3 * L2;
        psi[16] = T4 * L2;
        psi[17] = T5 * L2;
    }

    // ---------------------------------------------------------------
    // dshape_local
    // ---------------------------------------------------------------
    // Evaluate shape functions and their local-coordinate derivatives.
    //
    //   dpsids(i,j) = d(psi_i)/d(s_j),   j in {0,1,2}
    //
    // By the product rule:
    //   d/ds0 [T_k * L_m] = (dT_k/ds0) * L_m
    //   d/ds1 [T_k * L_m] = (dT_k/ds1) * L_m
    //   d/ds2 [T_k * L_m] = T_k * (dL_m/ds2)
    //
    // Triangle basis derivatives (lambda = 1 - s0 - s1):
    //   dT0/ds0 = 1 - 4*lambda           dT0/ds1 = 1 - 4*lambda
    //   dT1/ds0 = 4*s0 - 1               dT1/ds1 = 0
    //   dT2/ds0 = 0                       dT2/ds1 = 4*s1 - 1
    //   dT3/ds0 = 4*(1 - 2*s0 - s1)      dT3/ds1 = -4*s0
    //   dT4/ds0 = -4*s1                   dT4/ds1 = 4*(1 - s0 - 2*s1)
    //   dT5/ds0 = 4*s1                    dT5/ds1 = 4*s0
    //
    // 1-D Lagrange derivatives:
    //   dL0/ds2 = -3 + 4*s2
    //   dL1/ds2 =  4*(1 - 2*s2)
    //   dL2/ds2 =  4*s2 - 1
    // ---------------------------------------------------------------
    static void dshape_local(const Vector<double>& s,
                              Shape&  psi,
                              DShape& dpsids)
    {
        const double s0 = s[0];
        const double s1 = s[1];
        const double s2 = s[2];
        const double l  = 1.0 - s0 - s1;

        // ---- Triangle bases ----
        const double T0 = l  * (2.0*l  - 1.0);
        const double T1 = s0 * (2.0*s0 - 1.0);
        const double T2 = s1 * (2.0*s1 - 1.0);
        const double T3 = 4.0 * s0 * l;
        const double T4 = 4.0 * s1 * l;
        const double T5 = 4.0 * s0 * s1;

        // ---- Triangle basis derivatives wrt s0 ----
        const double dT0ds0 = 1.0 - 4.0*l;
        const double dT1ds0 = 4.0*s0 - 1.0;
        const double dT2ds0 = 0.0;
        const double dT3ds0 = 4.0*(1.0 - 2.0*s0 - s1);
        const double dT4ds0 = -4.0*s1;
        const double dT5ds0 = 4.0*s1;

        // ---- Triangle basis derivatives wrt s1 ----
        const double dT0ds1 = 1.0 - 4.0*l;
        const double dT1ds1 = 0.0;
        const double dT2ds1 = 4.0*s1 - 1.0;
        const double dT3ds1 = -4.0*s0;
        const double dT4ds1 = 4.0*(1.0 - s0 - 2.0*s1);
        const double dT5ds1 = 4.0*s0;

        // ---- 1-D Lagrange bases ----
        const double L0 = (1.0 - s2) * (1.0 - 2.0*s2);
        const double L1 = 4.0 * s2 * (1.0 - s2);
        const double L2 = s2 * (2.0*s2 - 1.0);

        // ---- 1-D Lagrange derivatives wrt s2 ----
        const double dL0ds2 = -3.0 + 4.0*s2;
        const double dL1ds2 =  4.0 * (1.0 - 2.0*s2);
        const double dL2ds2 =  4.0*s2 - 1.0;

        // ---- Shape functions ----
        // Layer 0
        psi[0]  = T0*L0;  psi[1]  = T1*L0;  psi[2]  = T2*L0;
        psi[3]  = T3*L0;  psi[4]  = T4*L0;  psi[5]  = T5*L0;
        // Layer 1
        psi[6]  = T0*L1;  psi[7]  = T1*L1;  psi[8]  = T2*L1;
        psi[9]  = T3*L1;  psi[10] = T4*L1;  psi[11] = T5*L1;
        // Layer 2
        psi[12] = T0*L2;  psi[13] = T1*L2;  psi[14] = T2*L2;
        psi[15] = T3*L2;  psi[16] = T4*L2;  psi[17] = T5*L2;

        // ---- d/ds0 ----
        // Layer 0
        dpsids(0,0)  = dT0ds0*L0;  dpsids(1,0)  = dT1ds0*L0;
        dpsids(2,0)  = dT2ds0*L0;  dpsids(3,0)  = dT3ds0*L0;
        dpsids(4,0)  = dT4ds0*L0;  dpsids(5,0)  = dT5ds0*L0;
        // Layer 1
        dpsids(6,0)  = dT0ds0*L1;  dpsids(7,0)  = dT1ds0*L1;
        dpsids(8,0)  = dT2ds0*L1;  dpsids(9,0)  = dT3ds0*L1;
        dpsids(10,0) = dT4ds0*L1;  dpsids(11,0) = dT5ds0*L1;
        // Layer 2
        dpsids(12,0) = dT0ds0*L2;  dpsids(13,0) = dT1ds0*L2;
        dpsids(14,0) = dT2ds0*L2;  dpsids(15,0) = dT3ds0*L2;
        dpsids(16,0) = dT4ds0*L2;  dpsids(17,0) = dT5ds0*L2;

        // ---- d/ds1 ----
        // Layer 0
        dpsids(0,1)  = dT0ds1*L0;  dpsids(1,1)  = dT1ds1*L0;
        dpsids(2,1)  = dT2ds1*L0;  dpsids(3,1)  = dT3ds1*L0;
        dpsids(4,1)  = dT4ds1*L0;  dpsids(5,1)  = dT5ds1*L0;
        // Layer 1
        dpsids(6,1)  = dT0ds1*L1;  dpsids(7,1)  = dT1ds1*L1;
        dpsids(8,1)  = dT2ds1*L1;  dpsids(9,1)  = dT3ds1*L1;
        dpsids(10,1) = dT4ds1*L1;  dpsids(11,1) = dT5ds1*L1;
        // Layer 2
        dpsids(12,1) = dT0ds1*L2;  dpsids(13,1) = dT1ds1*L2;
        dpsids(14,1) = dT2ds1*L2;  dpsids(15,1) = dT3ds1*L2;
        dpsids(16,1) = dT4ds1*L2;  dpsids(17,1) = dT5ds1*L2;

        // ---- d/ds2 ----
        // Layer 0
        dpsids(0,2)  = T0*dL0ds2;  dpsids(1,2)  = T1*dL0ds2;
        dpsids(2,2)  = T2*dL0ds2;  dpsids(3,2)  = T3*dL0ds2;
        dpsids(4,2)  = T4*dL0ds2;  dpsids(5,2)  = T5*dL0ds2;
        // Layer 1
        dpsids(6,2)  = T0*dL1ds2;  dpsids(7,2)  = T1*dL1ds2;
        dpsids(8,2)  = T2*dL1ds2;  dpsids(9,2)  = T3*dL1ds2;
        dpsids(10,2) = T4*dL1ds2;  dpsids(11,2) = T5*dL1ds2;
        // Layer 2
        dpsids(12,2) = T0*dL2ds2;  dpsids(13,2) = T1*dL2ds2;
        dpsids(14,2) = T2*dL2ds2;  dpsids(15,2) = T3*dL2ds2;
        dpsids(16,2) = T4*dL2ds2;  dpsids(17,2) = T5*dL2ds2;
    } 
};

class PyramidElementShapeC2
{
 public:
    // ---------------------------------------------------------------
    // shape
    // ---------------------------------------------------------------
    // Evaluate all 14 shape functions at local coordinate s=(s0,s1,s2),
    // as defined at:
    //   https://defelement.org/elements/examples/pyramid-lagrange-equispaced-2.html
    //
    // Unlike the wedge, this element is NOT a polynomial tensor product:
    // the pyramid's cross-section shrinks to a point at the apex (s2=1),
    // so the (s0,s1)-dependence is rational in s2. All shape functions
    // are built from:
    //
    //   px = s0 + s2 - 1        py = s1 + s2 - 1
    //   qx = 2*s0 + s2 - 1      qy = 2*s1 + s2 - 1
    //
    //   C0 = 4*s0*s1 + 2*s0*s2 - 2*s0 + 2*s1*s2 - 2*s1 + 2*s2^2 - 3*s2 + 1
    //   C1 = 4*s0*s1 + 2*s0*s2 - 2*s0 + 2*s1*s2 - 2*s1 - s2 + 1
    //
    // with denominators den1 = (s2-1), den2 = (s2-1)^2, which cancel
    // exactly at the nodes (psi[i](node_j) = delta_ij, sum_i psi[i] = 1).
    //
    //   Base vertices (s2 = 0):
    //     psi[0] = px*py*C0 / den2         (node 0: s0=0,s1=0,s2=0)
    //     psi[1] = s0*py*C1 / den2         (node 1: s0=1,s1=0,s2=0)
    //     psi[2] = s0*s1*C0 / den2         (node 2: s0=1,s1=1,s2=0)
    //     psi[3] = s1*px*C1 / den2         (node 3: s0=0,s1=1,s2=0)
    //
    //   Apex vertex:
    //     psi[4] = s2*(2*s2-1)             (node 4: s0=0,s1=0,s2=1)
    //
    //   Base-edge midpoints (s2 = 0):
    //     psi[5] = -4*s0*px*py*qy / den2   (edge 0-1)
    //     psi[6] = -4*s0*s1*qx*py / den2   (edge 1-2)
    //     psi[7] = -4*s0*s1*px*qy / den2   (edge 2-3)
    //     psi[8] = -4*s1*px*qx*py / den2   (edge 0-3)
    //
    //   Apex-edge midpoints (s2 = 1/2):
    //     psi[9]  = -4*s2*px*py   / den1   (edge 0-4)
    //     psi[10] =  4*s0*s2*py   / den1   (edge 1-4)
    //     psi[11] = -4*s0*s1*s2   / den1   (edge 2-4)
    //     psi[12] =  4*s1*s2*px   / den1   (edge 3-4)
    //
    //   Base-face centre:
    //     psi[13] = 16*s0*s1*px*py / den2
    // ---------------------------------------------------------------
    static void shape(const Vector<double>& s, Shape& psi)
    {
        const double s0 = s[0];
        const double s1 = s[1];
        const double s2 = s[2];

        // Recurring linear combinations
        const double px = s0 + s2 - 1.0;
        const double py = s1 + s2 - 1.0;
        const double qx = 2.0*s0 + s2 - 1.0;
        const double qy = 2.0*s1 + s2 - 1.0;

        // Recurring quadratic combinations
        const double C0 = 4.0*s0*s1 + 2.0*s0*s2 - 2.0*s0 + 2.0*s1*s2
                         - 2.0*s1 + 2.0*s2*s2 - 3.0*s2 + 1.0;
        const double C1 = 4.0*s0*s1 + 2.0*s0*s2 - 2.0*s0 + 2.0*s1*s2
                         - 2.0*s1 - s2 + 1.0;

        // Denominators (rational element: these cancel exactly at nodes)
        const double den1 = s2 - 1.0;
        const double den2 = den1 * den1;

        // Base vertices (s2 = 0)
        psi[0] = px * py * C0 / den2;
        psi[1] = s0 * py * C1 / den2;
        psi[2] = s0 * s1 * C0 / den2;
        psi[3] = s1 * px * C1 / den2;

        // Apex vertex
        psi[4] = s2 * (2.0*s2 - 1.0);

        // Base-edge midpoints (s2 = 0)
        psi[5] = -4.0 * s0 * px * py * qy / den2;   // edge 0-1
        psi[6] = -4.0 * s0 * s1 * qx * py / den2;   // edge 1-2
        psi[7] = -4.0 * s0 * s1 * px * qy / den2;   // edge 2-3
        psi[8] = -4.0 * s1 * px * qx * py / den2;   // edge 0-3

        // Apex-edge midpoints (s2 = 1/2)
        psi[9]  = -4.0 * s2 * px * py / den1;       // edge 0-4
        psi[10] =  4.0 * s0 * s2 * py / den1;       // edge 1-4
        psi[11] = -4.0 * s0 * s1 * s2 / den1;       // edge 2-4
        psi[12] =  4.0 * s1 * s2 * px / den1;       // edge 3-4

        // Base-face centre
        psi[13] = 16.0 * s0 * s1 * px * py / den2;
    }

    // ---------------------------------------------------------------
    // dshape_local
    // ---------------------------------------------------------------
    // Evaluate shape functions and their local-coordinate derivatives
    // for the 14-node rational pyramid, as defined at:
    //   https://defelement.org/elements/examples/pyramid-lagrange-equispaced-2.html
    //
    //   dpsids(i,j) = d(psi_i)/d(s_j),   j in {0,1,2}
    //
    // Each psi_i = N_i / den1  or  N_i / den2  (den1 = s2-1, den2 = den1^2),
    // where N_i is a polynomial built from:
    //
    //   px = s0+s2-1   py = s1+s2-1   qx = 2*s0+s2-1   qy = 2*s1+s2-1
    //   C0 = 4*s0*s1 + 2*s0*s2 - 2*s0 + 2*s1*s2 - 2*s1 + 2*s2^2 - 3*s2 + 1
    //   C1 = 4*s0*s1 + 2*s0*s2 - 2*s0 + 2*s1*s2 - 2*s1 - s2 + 1
    //
    // Derivatives of these building blocks:
    //   dpx/ds0=1, dpx/ds1=0, dpx/ds2=1        dpy/ds0=0, dpy/ds1=1, dpy/ds2=1
    //   dqx/ds0=2, dqx/ds1=0, dqx/ds2=1        dqy/ds0=0, dqy/ds1=2, dqy/ds2=1
    //   dC0/ds0 = dC1/ds0 = D0 = 4*s1+2*s2-2
    //   dC0/ds1 = dC1/ds1 = D1 = 4*s0+2*s2-2
    //   dC0/ds2 = E0 = 2*s0+2*s1+4*s2-3        dC1/ds2 = E1 = 2*s0+2*s1-1
    //
    // For each shape function N_i/den1^p (p=1 or 2) with dN_i/ds_j computed
    // via the product rule above:
    //   d/ds0, d/ds1 :  (den1 is independent of s0,s1)
    //     dpsi/ds0 = dN/ds0 / den1^p         dpsi/ds1 = dN/ds1 / den1^p
    //   d/ds2 (quotient rule, d(den1)/ds2 = 1):
    //     p=1:  dpsi/ds2 = dN/ds2/den1 - N/den1^2
    //     p=2:  dpsi/ds2 = (dN/ds2*den1 - 2*N) / den1^3
    // ---------------------------------------------------------------
    static void dshape_local(const Vector<double>& s,
                              Shape&  psi,
                              DShape& dpsids)
    {
        const double s0 = s[0];
        const double s1 = s[1];
        const double s2 = s[2];

        // ---- Recurring linear combinations and their derivatives ----
        const double px = s0 + s2 - 1.0;
        const double py = s1 + s2 - 1.0;
        const double qx = 2.0*s0 + s2 - 1.0;
        const double qy = 2.0*s1 + s2 - 1.0;

        // ---- Recurring quadratic combinations ----
        const double C0 = 4.0*s0*s1 + 2.0*s0*s2 - 2.0*s0 + 2.0*s1*s2
                         - 2.0*s1 + 2.0*s2*s2 - 3.0*s2 + 1.0;
        const double C1 = 4.0*s0*s1 + 2.0*s0*s2 - 2.0*s0 + 2.0*s1*s2
                         - 2.0*s1 - s2 + 1.0;

        // ---- Derivatives of C0, C1 ----
        const double D0 = 4.0*s1 + 2.0*s2 - 2.0;   // dC0/ds0 = dC1/ds0
        const double D1 = 4.0*s0 + 2.0*s2 - 2.0;   // dC0/ds1 = dC1/ds1
        const double E0 = 2.0*s0 + 2.0*s1 + 4.0*s2 - 3.0;  // dC0/ds2
        const double E1 = 2.0*s0 + 2.0*s1 - 1.0;           // dC1/ds2

        // ---- Denominators ----
        const double den1  = s2 - 1.0;
        const double den2  = den1 * den1;
        const double den1c = den1 * den2;   // den1^3

        // =================================================================
        // Base vertices (s2 = 0 layer), denominator den2
        // =================================================================
        {
            // Node 0: N0 = px*py*C0
            const double N0    = px * py * C0;
            const double dN0d0 = py * (C0 + px * D0);
            const double dN0d1 = px * (C0 + py * D1);
            const double dN0d2 = C0 * (px + py) + px * py * E0;

            psi[0]        = N0 / den2;
            dpsids(0, 0)  = dN0d0 / den2;
            dpsids(0, 1)  = dN0d1 / den2;
            dpsids(0, 2)  = (dN0d2 * den1 - 2.0 * N0) / den1c;
        }
        {
            // Node 1: N1 = s0*py*C1
            const double N1    = s0 * py * C1;
            const double dN1d0 = py * (C1 + s0 * D0);
            const double dN1d1 = s0 * (C1 + py * D1);
            const double dN1d2 = s0 * (C1 + py * E1);

            psi[1]        = N1 / den2;
            dpsids(1, 0)  = dN1d0 / den2;
            dpsids(1, 1)  = dN1d1 / den2;
            dpsids(1, 2)  = (dN1d2 * den1 - 2.0 * N1) / den1c;
        }
        {
            // Node 2: N2 = s0*s1*C0
            const double N2    = s0 * s1 * C0;
            const double dN2d0 = s1 * (C0 + s0 * D0);
            const double dN2d1 = s0 * (C0 + s1 * D1);
            const double dN2d2 = s0 * s1 * E0;

            psi[2]        = N2 / den2;
            dpsids(2, 0)  = dN2d0 / den2;
            dpsids(2, 1)  = dN2d1 / den2;
            dpsids(2, 2)  = (dN2d2 * den1 - 2.0 * N2) / den1c;
        }
        {
            // Node 3: N3 = s1*px*C1
            const double N3    = s1 * px * C1;
            const double dN3d0 = s1 * (C1 + px * D0);
            const double dN3d1 = px * (C1 + s1 * D1);
            const double dN3d2 = s1 * (C1 + px * E1);

            psi[3]        = N3 / den2;
            dpsids(3, 0)  = dN3d0 / den2;
            dpsids(3, 1)  = dN3d1 / den2;
            dpsids(3, 2)  = (dN3d2 * den1 - 2.0 * N3) / den1c;
        }

        // =================================================================
        // Apex vertex -- plain polynomial, no quotient rule needed
        // =================================================================
        psi[4]       = s2 * (2.0 * s2 - 1.0);
        dpsids(4, 0) = 0.0;
        dpsids(4, 1) = 0.0;
        dpsids(4, 2) = 4.0 * s2 - 1.0;

        // =================================================================
        // Base-edge midpoints (s2 = 0 layer), denominator den2
        // =================================================================
        {
            // Node 5 (edge 0-1): N5 = -4*s0*px*py*qy
            const double N5    = -4.0 * s0 * px * py * qy;
            const double dN5d0 = -4.0 * py * qy * (px + s0);
            const double dN5d1 = -4.0 * s0 * px * (qy + 2.0 * py);
            const double dN5d2 = -4.0 * s0 * (py * qy + px * qy + px * py);

            psi[5]        = N5 / den2;
            dpsids(5, 0)  = dN5d0 / den2;
            dpsids(5, 1)  = dN5d1 / den2;
            dpsids(5, 2)  = (dN5d2 * den1 - 2.0 * N5) / den1c;
        }
        {
            // Node 6 (edge 1-2): N6 = -4*s0*s1*qx*py
            const double N6    = -4.0 * s0 * s1 * qx * py;
            const double dN6d0 = -4.0 * s1 * py * (qx + 2.0 * s0);
            const double dN6d1 = -4.0 * s0 * qx * (py + s1);
            const double dN6d2 = -4.0 * s0 * s1 * (py + qx);

            psi[6]        = N6 / den2;
            dpsids(6, 0)  = dN6d0 / den2;
            dpsids(6, 1)  = dN6d1 / den2;
            dpsids(6, 2)  = (dN6d2 * den1 - 2.0 * N6) / den1c;
        }
        {
            // Node 7 (edge 2-3): N7 = -4*s0*s1*px*qy
            const double N7    = -4.0 * s0 * s1 * px * qy;
            const double dN7d0 = -4.0 * s1 * qy * (px + s0);
            const double dN7d1 = -4.0 * s0 * px * (qy + 2.0 * s1);
            const double dN7d2 = -4.0 * s0 * s1 * (qy + px);

            psi[7]        = N7 / den2;
            dpsids(7, 0)  = dN7d0 / den2;
            dpsids(7, 1)  = dN7d1 / den2;
            dpsids(7, 2)  = (dN7d2 * den1 - 2.0 * N7) / den1c;
        }
        {
            // Node 8 (edge 0-3): N8 = -4*s1*px*qx*py
            const double N8    = -4.0 * s1 * px * qx * py;
            const double dN8d0 = -4.0 * s1 * py * (qx + 2.0 * px);
            const double dN8d1 = -4.0 * px * qx * (py + s1);
            const double dN8d2 = -4.0 * s1 * (qx * py + px * py + px * qx);

            psi[8]        = N8 / den2;
            dpsids(8, 0)  = dN8d0 / den2;
            dpsids(8, 1)  = dN8d1 / den2;
            dpsids(8, 2)  = (dN8d2 * den1 - 2.0 * N8) / den1c;
        }

        // =================================================================
        // Apex-edge midpoints (s2 = 1/2 layer), denominator den1
        // =================================================================
        {
            // Node 9 (edge 0-4): N9 = -4*s2*px*py
            const double N9    = -4.0 * s2 * px * py;
            const double dN9d0 = -4.0 * s2 * py;
            const double dN9d1 = -4.0 * s2 * px;
            const double dN9d2 = -4.0 * (px * py + s2 * py + s2 * px);

            psi[9]        = N9 / den1;
            dpsids(9, 0)  = dN9d0 / den1;
            dpsids(9, 1)  = dN9d1 / den1;
            dpsids(9, 2)  = (dN9d2 * den1 - N9) / den2;
        }
        {
            // Node 10 (edge 1-4): N10 = 4*s0*s2*py
            const double N10    = 4.0 * s0 * s2 * py;
            const double dN10d0 = 4.0 * s2 * py;
            const double dN10d1 = 4.0 * s0 * s2;
            const double dN10d2 = 4.0 * s0 * (py + s2);

            psi[10]       = N10 / den1;
            dpsids(10, 0) = dN10d0 / den1;
            dpsids(10, 1) = dN10d1 / den1;
            dpsids(10, 2) = (dN10d2 * den1 - N10) / den2;
        }
        {
            // Node 11 (edge 2-4): N11 = -4*s0*s1*s2
            const double N11    = -4.0 * s0 * s1 * s2;
            const double dN11d0 = -4.0 * s1 * s2;
            const double dN11d1 = -4.0 * s0 * s2;
            const double dN11d2 = -4.0 * s0 * s1;

            psi[11]       = N11 / den1;
            dpsids(11, 0) = dN11d0 / den1;
            dpsids(11, 1) = dN11d1 / den1;
            dpsids(11, 2) = (dN11d2 * den1 - N11) / den2;
        }
        {
            // Node 12 (edge 3-4): N12 = 4*s1*s2*px
            const double N12    = 4.0 * s1 * s2 * px;
            const double dN12d0 = 4.0 * s1 * s2;
            const double dN12d1 = 4.0 * s2 * px;
            const double dN12d2 = 4.0 * s1 * (px + s2);

            psi[12]       = N12 / den1;
            dpsids(12, 0) = dN12d0 / den1;
            dpsids(12, 1) = dN12d1 / den1;
            dpsids(12, 2) = (dN12d2 * den1 - N12) / den2;
        }

        // =================================================================
        // Base-face centre, denominator den2
        // =================================================================
        {
            // Node 13: N13 = 16*s0*s1*px*py
            const double N13    = 16.0 * s0 * s1 * px * py;
            const double dN13d0 = 16.0 * s1 * py * (px + s0);
            const double dN13d1 = 16.0 * s0 * px * (py + s1);
            const double dN13d2 = 16.0 * s0 * s1 * (px + py);

            psi[13]       = N13 / den2;
            dpsids(13, 0) = dN13d0 / den2;
            dpsids(13, 1) = dN13d1 / den2;
            dpsids(13, 2) = (dN13d2 * den1 - 2.0 * N13) / den1c;
        }
    }
};

class WedgeElementBase :  public virtual FiniteElement
{
  public:
    void build_face_element(const int& face_index,FaceElement* face_element_pt) override;
    CoordinateMappingFctPt face_to_bulk_coordinate_fct_pt(const int& face_index) const override;
    BulkCoordinateDerivativesFctPt bulk_coordinate_derivatives_fct_pt(const int& face_index) const override ;
    int face_outer_unit_normal_sign(const int&) const override;
    double s_min() const    { return 0.0; }
    double s_max() const    { return 1.0; }
    unsigned nvertex_node() const { return 6; }
    Node* vertex_node_pt(const unsigned& j) const override
    {           
      if (j > 5)
      {
          std::ostringstream error_message;
          error_message  << "Element only has six vertex nodes; called with node number " << j << std::endl;
          throw OomphLibError(error_message.str(),OOMPH_CURRENT_FUNCTION,OOMPH_EXCEPTION_LOCATION);
      }
      return node_pt(j);
    }
    unsigned nnode_on_face() const override { throw_runtime_error("nnode_on_face cannot be implemented for a Wedge, damn."); }
    virtual unsigned nnode_on_face_by_index(const int& face_index) const  { return (face_index<2) ? 3 : 4; }
};

class RefineableWedgeElement : public virtual RefineableElement, public virtual WedgeElementBase
  {

  public:
    /// \short Shorthand for pointer to an argument-free void member
    /// function of the refineable element
    typedef void (RefineableWedgeElement::*VoidMemberFctPt)();

    /// Constructor: Pass refinement level (default 0 = root)
    RefineableWedgeElement() : RefineableElement()
    {
    }

    /// Broken copy constructor
    RefineableWedgeElement(const RefineableWedgeElement &dummy)
    {
      BrokenCopy::broken_copy("RefineableWedgeElement");
    }

    virtual ~RefineableWedgeElement()
    {
    }

    unsigned required_nsons() const
    {
      throw_runtime_error("TODO");
      return 4;
    }

    virtual Node *node_created_by_neighbour(const Vector<double> &s_fraction, bool &is_periodic);

    virtual Node *node_created_by_son_of_neighbour(const Vector<double> &s_fraction, bool &is_periodic)
    {
      return 0;
    }

    virtual void build(Mesh *&mesh_pt, Vector<Node *> &new_node_pt, bool &was_already_built, std::ofstream &new_nodes_file);

    void check_integrity(double &max_error);

    void output_corners(std::ostream &outfile, const std::string &colour) const;

    OcTree *octree_pt() { return dynamic_cast<OcTree *>(Tree_pt); }

    OcTree *octree_pt() const { return dynamic_cast<OcTree *>(Tree_pt); }

    void setup_hanging_nodes(Vector<std::ofstream *> &output_stream);

    virtual void further_setup_hanging_nodes() = 0;

  protected:
    static std::map<unsigned, DenseMatrix<int>> Father_bound;

    void setup_father_bounds();

    void get_edge_bcs(const int &edge, Vector<int> &bound_cons) const;

  public:
    void get_boundaries(const int &edge, std::set<unsigned> &boundaries) const;

    void get_bcs(int bound, Vector<int> &bound_cons) const;
    void interpolated_zeta_on_edge(const unsigned &boundary, const int &edge, const Vector<double> &s, Vector<double> &zeta);

  protected:
    void setup_hang_for_value(const int &value_id);

    virtual void quad_hang_helper(const int &value_id, const int &my_edge, std::ofstream &output_hangfile);
  };


class PyramidElementBase :  public virtual FiniteElement
{
  public:
    void build_face_element(const int& face_index,FaceElement* face_element_pt) override;
    CoordinateMappingFctPt face_to_bulk_coordinate_fct_pt(const int& face_index) const override;
    BulkCoordinateDerivativesFctPt bulk_coordinate_derivatives_fct_pt(const int& face_index) const override ;
    int face_outer_unit_normal_sign(const int&) const override;
    double s_min() const    { return 0.0; }
    double s_max() const    { return 1.0; }
    unsigned nvertex_node() const   { return 5; }
    Node* vertex_node_pt(const unsigned& j) const override
    {       
      if (j > 4)
      {
          std::ostringstream error_message;
          error_message  << "Element only has five vertex nodes; called with node number " << j << std::endl;
          throw OomphLibError(error_message.str(),OOMPH_CURRENT_FUNCTION,OOMPH_EXCEPTION_LOCATION);
      }
      return node_pt(j);    
    }
    unsigned nnode_on_face() const override { throw_runtime_error("nnode_on_face cannot be implemented for a Pyramid, damn."); }
    virtual unsigned nnode_on_face_by_index(const int& face_index) const  { return (face_index==4) ? 4 : 3; } 
};


class RefineablePyramidElement : public virtual RefineableElement, public virtual PyramidElementBase
  {

  public:
    /// \short Shorthand for pointer to an argument-free void member
    /// function of the refineable element
    typedef void (RefineablePyramidElement::*VoidMemberFctPt)();

    /// Constructor: Pass refinement level (default 0 = root)
    RefineablePyramidElement() : RefineableElement()
    {
    }

    /// Broken copy constructor
    RefineablePyramidElement(const RefineablePyramidElement &dummy)
    {
      BrokenCopy::broken_copy("RefineablePyramidElement");
    }

    virtual ~RefineablePyramidElement()
    {
    }

    unsigned required_nsons() const
    {
      throw_runtime_error("TODO"); // Here, nothing is do be done for now
      return 4;
    }

    virtual Node *node_created_by_neighbour(const Vector<double> &s_fraction, bool &is_periodic);

    virtual Node *node_created_by_son_of_neighbour(const Vector<double> &s_fraction, bool &is_periodic)
    {
      return 0;
    }

    virtual void build(Mesh *&mesh_pt, Vector<Node *> &new_node_pt, bool &was_already_built, std::ofstream &new_nodes_file);

    void check_integrity(double &max_error);

    void output_corners(std::ostream &outfile, const std::string &colour) const;

    OcTree *octree_pt() { return dynamic_cast<OcTree *>(Tree_pt); }

    OcTree *octree_pt() const { return dynamic_cast<OcTree *>(Tree_pt); }

    void setup_hanging_nodes(Vector<std::ofstream *> &output_stream);

    virtual void further_setup_hanging_nodes() = 0;

  protected:
    static std::map<unsigned, DenseMatrix<int>> Father_bound;

    void setup_father_bounds();

    void get_edge_bcs(const int &edge, Vector<int> &bound_cons) const;

  public:
    void get_boundaries(const int &edge, std::set<unsigned> &boundaries) const;

    void get_bcs(int bound, Vector<int> &bound_cons) const;
    void interpolated_zeta_on_edge(const unsigned &boundary, const int &edge, const Vector<double> &s, Vector<double> &zeta);

  protected:
    void setup_hang_for_value(const int &value_id);

    virtual void quad_hang_helper(const int &value_id, const int &my_edge, std::ofstream &output_hangfile);
  };





class WedgeElementC1 :  public virtual RefineableWedgeElement
{
 // A first order wedge element 
 // Nodes are ordered as follows:
 /*
 Index : Local coordinates (s0,s1,s2)
   0: (0,0,0)
   1: (1,0,0)
   2: (0,1,0)

   3: (0,0,1)
   4: (1,0,1)
   5: (0,1,1)


              5 o
               /\
              /  \
             /    \
            / [1]  \
         3 o--------o 4
           |  [3]   |
   [2]     |  2 o   |    [4]
           |   /\   |
           |  /  \  |
           | /    \ |
           |/  [0] \|
         0 o--------o 1

    s[i] runs from 0 to 1, but with the constraint that s[0]+s[1] <= 1

    facets are numbered as follows:
    facet 0: s[2] = 0, nodes 0,1,2
    facet 1: s[2] = 1, nodes 3,4,5
    facet 2: s[0] = 0, nodes 0,2,3,5
    facet 3: s[1] = 0, nodes 0,1,3,4
    facet 4: s[0]+s[1] = 1, nodes 1,2,4,5
 */

 private:
    static WedgeGaussC1 Default_integration_scheme;
 public:    
    unsigned nnode_1d() const { return 2;}

    WedgeElementC1() : WedgeElementBase() 
    {
        set_n_node(6);
        set_dimension(3);
        set_integration_scheme(&Default_integration_scheme);
    }
    WedgeElementC1(const WedgeElementC1&) = delete;
    ~WedgeElementC1() {}

    unsigned int get_bulk_node_number(const int & face_index, const unsigned int& i) const override;
        
    void shape(const Vector<double>& s, Shape& psi) const override
    {
        WedgeElementShapeC1::shape(s, psi);
    }

    void dshape_local(const Vector<double>& s,Shape& psi,DShape& dpsids) const override
    {
        WedgeElementShapeC1::dshape_local(s, psi, dpsids);
    }

    inline void local_coordinate_of_node(const unsigned& j,Vector<double>& s) const
    {
      if (j==0) { s[0] = 0.0; s[1] = 0.0; s[2] = 0.0;}
      else if (j==1) { s[0] = 1.0; s[1] = 0.0; s[2] = 0.0;}
      else if (j==2) { s[0] = 0.0; s[1] = 1.0; s[2] = 0.0;}
      else if (j==3) { s[0] = 0.0; s[1] = 0.0; s[2] = 1.0;}
      else if (j==4) { s[0] = 1.0; s[1] = 0.0; s[2] = 1.0;}
      else if (j==5) { s[0] = 0.0; s[1] = 1.0; s[2] = 1.0;}
      else
      {
          std::ostringstream error_message;
          error_message  << "Element only has six nodes; called with node number " << j << std::endl;
          throw OomphLibError(error_message.str(),OOMPH_CURRENT_FUNCTION,OOMPH_EXCEPTION_LOCATION);
      }
    }


};



class PyramidElementC1 :  public virtual RefineablePyramidElement
// A first order pyramid element
// Nodes are ordered as follows:
/*
Index : Local coordinates (s0,s1,s2)
  0: (0,0,0)
  1: (1,0,0)
  2: (1,1,0)
  3: (0,1,0)

  4: (0,0,1)  <- apex (degenerate point, s2=1 collapses to a point at s0=s1=0)


              4 o
               /|\
              / | \
             /  |  \
            / [2]|[3]\
           /    |    \
        3 o-----|-----o 2
          |  [4]|     |
  [1]     |     |     |   [0]
          |   [4]     |
        0 o-----------o 1

  s[0] and s[1] run from 0 to 1-s[2], s[2] runs from 0 to 1

  facets are numbered as follows:
  facet 0: s[1] = 0,        nodes 0,1,4     (triangle)
  facet 1: s[0] = 1-s[2],   nodes 1,2,4     (triangle)
  facet 2: s[1] = 1-s[2],   nodes 2,3,4     (triangle)
  facet 3: s[0] = 0,        nodes 0,3,4     (triangle)
  facet 4: s[2] = 0,        nodes 0,1,2,3   (quad base)
*/
{ 
 private:
    static PyramidGaussC1 Default_integration_scheme; 
 public:    
    unsigned nnode_1d() const { return 2;}

    PyramidElementC1() : PyramidElementBase() 
    {
        set_n_node(5);
        set_dimension(3);
        set_integration_scheme(&Default_integration_scheme);
    }
    PyramidElementC1(const PyramidElementC1&) = delete;
    ~PyramidElementC1() {}

    unsigned int get_bulk_node_number(const int & face_index, const unsigned int& i) const override;
        
    void shape(const Vector<double>& s, Shape& psi) const override
    {
        PyramidElementShapeC1::shape(s, psi);
    }

    void dshape_local(const Vector<double>& s,Shape& psi,DShape& dpsids) const override
    {
        PyramidElementShapeC1::dshape_local(s, psi, dpsids);
    }

    inline void local_coordinate_of_node(const unsigned& j,Vector<double>& s) const
    {
      if (j==0) { s[0] = 0.0; s[1] = 0.0; s[2] = 0.0;}
      else if (j==1) { s[0] = 1.0; s[1] = 0.0; s[2] = 0.0;}
      else if (j==2) { s[0] = 1.0; s[1] = 1.0; s[2] = 0.0;}
      else if (j==3) { s[0] = 0.0; s[1] = 1.0; s[2] = 0.0;}
      else if (j==4) { s[0] = 0.0; s[1] = 0.0; s[2] = 1.0;}
      else
      {
          std::ostringstream error_message;
          error_message  << "Element only has five nodes; called with node number " << j << std::endl;
          throw OomphLibError(error_message.str(),OOMPH_CURRENT_FUNCTION,OOMPH_EXCEPTION_LOCATION);
      }
    }


};




class WedgeElementC2 : public virtual RefineableWedgeElement
{
 /* Admittably, with the help of Claude.ai ...
 // An 18-node quadratic (serendipity-style) wedge / triangular prism element.
 //
 // Node numbering
 // ==============
 // Nodes are arranged in three triangular layers along s2:
 //
 //   Layer s2 = 0  : nodes  0 –  5
 //   Layer s2 = 1/2: nodes  6 – 11
 //   Layer s2 = 1  : nodes 12 – 17
 //
 // Within each layer the six nodes follow the quadratic triangle pattern
 // (corners first, then edge midpoints):
 //
 //   +0 : corner   at (s0,s1) = (0,   0  )
 //   +1 : corner   at (s0,s1) = (1,   0  )
 //   +2 : corner   at (s0,s1) = (0,   1  )
 //   +3 : midpoint of edge 0–1, (s0,s1) = (1/2, 0  )
 //   +4 : midpoint of edge 0–2, (s0,s1) = (0,   1/2)
 //   +5 : midpoint of edge 1–2, (s0,s1) = (1/2, 1/2)
 //
 // Full coordinate table
 // ---------------------
 //   Node  s0    s1    s2
 //    0    0     0     0
 //    1    1     0     0
 //    2    0     1     0
 //    3    1/2   0     0
 //    4    0     1/2   0
 //    5    1/2   1/2   0

 //    6    0     0     1/2
 //    7    1     0     1/2
 //    8    0     1     1/2
 //    9    1/2   0     1/2
 //   10    0     1/2   1/2
 //   11    1/2   1/2   1/2
 
 //   12    0     0     1
 //   13    1     0     1
 //   14    0     1     1
 //   15    1/2   0     1
 //   16    0     1/2   1
 //   17    1/2   1/2   1
 //
 // Schematic (face labels match WedgeElementC1)
 //
 //              14 o
 //               / \
 //            16o   o17
 //             /     \
 //          12o--15---o13
 //            |  [1]  |
 //   [2]      |       |  [4]       faces:
 //            8o  o11 o              0 : s2 = 0        (triangular, 6 nodes)
 //            |  [3]  |              1 : s2 = 1        (triangular, 6 nodes)
 //            |  2 o  |              2 : s0 = 0        (quad, 9 nodes)
 //            |4o   o5|              3 : s1 = 0        (quad, 9 nodes)
 //            |/  [0]\|              4 : s0+s1 = 1     (quad, 9 nodes)
 //           6o---9---o7
 //            |       |
 //           0o---3---o1
 //
 //   s[i] runs from 0 to 1, with the constraint s[0]+s[1] <= 1
 //
 // Shape functions
 // ===============
 // psi[layer*6 + k] = T_k(s0,s1) * L_layer(s2)
 //
 // Quadratic triangle bases  (lambda = 1 - s0 - s1):
 //   T0 = lambda*(2*lambda - 1)        vertex (0,0)
 //   T1 = s0*(2*s0 - 1)                vertex (1,0)
 //   T2 = s1*(2*s1 - 1)                vertex (0,1)
 //   T3 = 4*s0*lambda                  midpoint (1/2, 0)
 //   T4 = 4*s1*lambda                  midpoint (0, 1/2)
 //   T5 = 4*s0*s1                      midpoint (1/2, 1/2)
 //
 // Quadratic 1-D Lagrange bases on [0,1]:
 //   L0(s2) = (1-s2)*(1-2*s2)          node at s2 = 0
 //   L1(s2) = 4*s2*(1-s2)              node at s2 = 1/2
 //   L2(s2) = s2*(2*s2-1)              node at s2 = 1
 //
 // Quadrilateral face node ordering (9 nodes)
 // ===========================================
 // Convention: 4 corners -> 4 edge midpoints -> 1 centre.
 // This is the natural extension of the C1 bilinear quad face ordering.
 //
 //   Face 2 (s0=0):  parametric coords (s1, s2)
 //     i   bulk node   s1   s2   role
 //     0       0       0    0    corner
 //     1      12       0    1    corner
 //     2       2       1    0    corner
 //     3      14       1    1    corner
 //     4       6       0   1/2   s2-edge mid (between corners i=0 & i=1)
 //     5       8       1   1/2   s2-edge mid (between corners i=2 & i=3)
 //     6       4      1/2   0    s1-edge mid (between corners i=0 & i=2)
 //     7      16      1/2   1    s1-edge mid (between corners i=1 & i=3)
 //     8      10      1/2  1/2   centre
 //
 //   Face 3 (s1=0):  parametric coords (s0, s2)
 //     i   bulk node   s0   s2   role
 //     0       0       0    0    corner
 //     1      12       0    1    corner
 //     2       1       1    0    corner
 //     3      13       1    1    corner
 //     4       6       0   1/2   s2-edge mid (between corners i=0 & i=1)
 //     5       7       1   1/2   s2-edge mid (between corners i=2 & i=3)
 //     6       3      1/2   0    s0-edge mid (between corners i=0 & i=2)
 //     7      15      1/2   1    s0-edge mid (between corners i=1 & i=3)
 //     8       9      1/2  1/2   centre
 //
 //   Face 4 (s0+s1=1): parametric coords (t, s2),
 //                      t=0 at (s0=1,s1=0), t=1 at (s0=0,s1=1)
 //     i   bulk node   t    s2   role
 //     0       1       0    0    corner
 //     1      13       0    1    corner
 //     2       2       1    0    corner
 //     3      14       1    1    corner
 //     4       7       0   1/2   s2-edge mid (between corners i=0 & i=1)
 //     5       8       1   1/2   s2-edge mid (between corners i=2 & i=3)
 //     6       5      1/2   0    t-edge  mid (between corners i=0 & i=2)
 //     7      17      1/2   1    t-edge  mid (between corners i=1 & i=3)
 //     8      11      1/2  1/2   centre
 */
 private:
    static WedgeGaussC2 Default_integration_scheme;

 public:
    unsigned nnode_1d() const { return 3; }

    WedgeElementC2() : WedgeElementBase()
    {
        set_n_node(18);
        set_dimension(3);
        set_integration_scheme(&Default_integration_scheme);
    }
    WedgeElementC2(const WedgeElementC2&) = delete;
    ~WedgeElementC2() {}

    Node* vertex_node_pt(const unsigned& j) const override
    {           
      if (j > 5)
      {
          std::ostringstream error_message;
          error_message  << "Element only has six vertex nodes; called with node number " << j << std::endl;
          throw OomphLibError(error_message.str(),OOMPH_CURRENT_FUNCTION,OOMPH_EXCEPTION_LOCATION);
      }
      return node_pt((j<3 ? j : j+9)); // vertex nodes are the first 3, then a gap of 9 to skip the mid-edge nodes, then the last 3 vertex nodes at the end of the numbering
    }
    unsigned nnode_on_face() const override { throw_runtime_error("nnode_on_face cannot be implemented for a Wedge, damn."); }
    virtual unsigned nnode_on_face_by_index(const int& face_index) const  { return (face_index<2) ? 6 : 9; }

    // ---------------------------------------------------------------
    // get_bulk_node_number
    // ---------------------------------------------------------------
    // Return the bulk-element node index for local face-node i on
    // face face_index.
    //
    // Triangular faces (0,1) : 6 nodes each.
    // Quadrilateral faces (2,3,4) : 9 nodes each,
    //   ordered as: 4 corners, 4 edge midpoints, 1 centre.
    //
    // The triangular face orderings mirror the C1 element convention:
    //   face 0 (s2=0) uses reversed winding (2,1,0) with matching
    //                  edge midpoints — outward normal points in -s2.
    //   face 1 (s2=1) uses forward  winding (12,13,14) with midpoints
    //                  (15,16,17).
    // ---------------------------------------------------------------
    unsigned get_bulk_node_number(const int& face_index,const unsigned int& i) const;

    void shape(const Vector<double>& s, Shape& psi) const override
    {
        WedgeElementShapeC2::shape(s, psi);
    }

    void dshape_local(const Vector<double>& s,Shape& psi,DShape& dpsids) const override
    {
        WedgeElementShapeC2::dshape_local(s, psi, dpsids);
    }
    

    // ---------------------------------------------------------------
    // local_coordinate_of_node
    // ---------------------------------------------------------------
    // Return the local coordinate s of node j.
    //
    // Layout: layer = j/6 in {0,1,2},  tri_node = j%6 in {0,...,5}.
    //   layer -> s2 :  0 -> 0.0,  1 -> 0.5,  2 -> 1.0
    //   tri_node -> (s0, s1):
    //     0 -> (0,   0  )   1 -> (1,   0  )   2 -> (0,   1  )
    //     3 -> (0.5, 0  )   4 -> (0,   0.5)   5 -> (0.5, 0.5)
    // ---------------------------------------------------------------
    inline void local_coordinate_of_node(const unsigned& j,
                                          Vector<double>& s) const
    {
        if (j >= 18)
        {
            std::ostringstream error_message;
            error_message << "Element only has 18 nodes; called with node "
                          << j << std::endl;
            throw OomphLibError(error_message.str(),
                                OOMPH_CURRENT_FUNCTION,
                                OOMPH_EXCEPTION_LOCATION);
        }

        // s2 from layer index
        s[2] = 0.5 * static_cast<double>(j / 6);

        // (s0, s1) from triangle-node index within the layer
        switch (j % 6)
        {
            case 0: s[0] = 0.0; s[1] = 0.0; break;
            case 1: s[0] = 1.0; s[1] = 0.0; break;
            case 2: s[0] = 0.0; s[1] = 1.0; break;
            case 3: s[0] = 0.5; s[1] = 0.0; break;
            case 4: s[0] = 0.0; s[1] = 0.5; break;
            case 5: s[0] = 0.5; s[1] = 0.5; break;
            default: break;   // unreachable: j%6 is always in [0,5]
        }
    }
};

class PyramidElementC2 : public virtual RefineablePyramidElement
{
 // A 14-node quadratic (rational) pyramid element
 //
 // Node numbering
 // ==============
 // The reference pyramid has a square base (s2 = 0) and a single apex at
 // (s0,s1,s2) = (0,0,1) -- i.e. the apex sits directly above base corner 0,
 // NOT above the centre of the base. Nodes are grouped as:
 //
 //   Vertices        : nodes  0 –  4
 //   Base-edge / apex-
 //     edge midpoints: nodes  5 – 12
 //   Base-face centre: node  13
 //
 //   Node  s0    s1    s2   role
 //    0    0     0     0    vertex (base)
 //    1    1     0     0    vertex (base)
 //    2    1     1     0    vertex (base)
 //    3    0     1     0    vertex (base)
 //    4    0     0     1    vertex (apex)
 //
 //    5    1/2   0     0    midpoint of edge 0–1
 //    6    1     1/2   0    midpoint of edge 1–2
 //    7    1/2   1     0    midpoint of edge 2–3
 //    8    0     1/2   0    midpoint of edge 0–3
 //    9    0     0     1/2  midpoint of edge 0–4
 //   10    1/2   0     1/2  midpoint of edge 1–4
 //   11    1/2   1/2   1/2  midpoint of edge 2–4
 //   12    0     1/2   1/2  midpoint of edge 3–4
 //
 //   13    1/2   1/2   0    centre of base face
 //
 // Schematic (apex drawn centred for clarity; see table above for the
 // true, corner-0-aligned coordinates)
 //
 //                        4
 //                      ,'|`.
 //                  11,'  |  `.10
 //                 ,'    9|    `.
 //               2'-------+-------1
 //               |\      12\      /|
 //               | \        \    / |
 //               |  \        \  /  |
 //               7\  \        \/9  5
 //                 \  3--------+--0    <- base face (s2 = 0)
 //                  \,'       8,'
 //                   `.       ,'
 //                     `-----'
 //
 //   s[i] runs from 0 to 1; the physical pyramid tapers linearly so that
 //   at height s2 the cross-section is the square 0 <= s0,s1 <= 1 - s2.
 //
 // Face associations
 // ==================
 //   Face 0: s[1] = 0,        nodes 0,1,4     (triangle)
 //   Face 1: s[0] = 1-s[2],   nodes 1,2,4     (triangle)
 //   Face 2: s[1] = 1-s[2],   nodes 2,3,4     (triangle)
 //   Face 3: s[0] = 0,        nodes 0,3,4     (triangle)
 //   Face 4: s[2] = 0,        nodes 0,1,2,3   (quad base)
 //
 // Shape functions
 // ===============
 // Unlike the wedge element, these are NOT a simple polynomial tensor
 // product: because the pyramid's cross-section shrinks to a point at the
 // apex, the (s0,s1)-dependence must be rational in s2 to remain a valid
 // 14-dimensional finite element space (see the "V" basis on the
 // DefElement page). All denominators below are (s2 - 1) or (s2 - 1)^2
 // and cancel exactly at the nodes; psi[i] satisfies psi[i](node_j) = delta_ij
 // and sum_i psi[i] = 1 everywhere on the pyramid.
 //
 //   Base vertices (s2 = 0 layer):
 //     psi[0]  =  (s0+s2-1)(s1+s2-1) * (4 s0 s1 + 2 s0 s2 - 2 s0 + 2 s1 s2 - 2 s1 + 2 s2^2 - 3 s2 + 1)             / (s2-1)^2
 //     psi[1]  =  s0 (s1+s2-1)       * (4 s0 s1 + 2 s0 s2 - 2 s0 + 2 s1 s2 - 2 s1 - s2 + 1)                       / (s2-1)^2
 //     psi[3]  =  s1 (s0+s2-1)       * (4 s0 s1 + 2 s0 s2 - 2 s0 + 2 s1 s2 - 2 s1 - s2 + 1)                       / (s2-1)^2
 //     psi[2]  =  s0 s1              * (4 s0 s1 + 2 s0 s2 - 2 s0 + 2 s1 s2 - 2 s1 + 2 s2^2 - 3 s2 + 1)             / (s2-1)^2
 //
 //   Apex vertex:
 //     psi[4]  =  s2 (2 s2 - 1)
 //
 //   Base-edge midpoints (s2 = 0 layer):
 //     psi[5]  = -4 s0 (s0+s2-1)(s1+s2-1)(2 s1+s2-1)   / (s2-1)^2      // edge 0-1
 //     psi[6]  = -4 s0 s1 (2 s0+s2-1)(s1+s2-1)         / (s2-1)^2      // edge 1-2
 //     psi[7] = -4 s0 s1 (s0+s2-1)(2 s1+s2-1)         / (s2-1)^2      // edge 2-3
 //     psi[8]  = -4 s1 (s0+s2-1)(2 s0+s2-1)(s1+s2-1)   / (s2-1)^2      // edge 0-3
 //
 //   Apex-edge midpoints (s2 = 1/2 layer):
 //     psi[9]  =  -4 s2 (s0+s2-1)(s1+s2-1)  / (s2-1)     // edge 0-4
 //     psi[10]  =   4 s0 s2 (s1+s2-1)        / (s2-1)     // edge 1-4
 //     psi[11] =  -4 s0 s1 s2               / (s2-1)     // edge 2-4
 //     psi[12] =   4 s1 s2 (s0+s2-1)        / (s2-1)     // edge 3-4
 //
 //   Base-face centre:
 //     psi[13] =  16 s0 s1 (s0+s2-1)(s1+s2-1)  / (s2-1)^2

 private:
    static PyramidGaussC2 Default_integration_scheme;

 public:
    unsigned nnode_1d() const { return 3; }

    PyramidElementC2() : PyramidElementBase()
    {
        set_n_node(14);
        set_dimension(3);
        set_integration_scheme(&Default_integration_scheme);
    }
    PyramidElementC2(const PyramidElementC2&) = delete;
    ~PyramidElementC2() {}

    Node* vertex_node_pt(const unsigned& j) const override
    {           
      if (j > 4)
      {
          std::ostringstream error_message;
          error_message  << "Element only has five vertex nodes; called with node number " << j << std::endl;
          throw OomphLibError(error_message.str(),OOMPH_CURRENT_FUNCTION,OOMPH_EXCEPTION_LOCATION);
      }
      return node_pt(j); // ordering same as in C1 pyramid
    }
    unsigned nnode_on_face() const override { throw_runtime_error("nnode_on_face cannot be implemented for a Pyramid, damn."); }
    virtual unsigned nnode_on_face_by_index(const int& face_index) const  { return (face_index==4) ? 9 : 6; }

    // ---------------------------------------------------------------
    // get_bulk_node_number
    // ---------------------------------------------------------------
    // Return the bulk-element node index for local face-node i on
    // face face_index.
    //
    // Triangular faces (0,1,2,3) : 6 nodes each.
    // Quadrilateral faces (5) : 9 nodes each,
    //   ordered as: 4 corners, 4 edge midpoints, 1 centre.
    // ---------------------------------------------------------------
    unsigned get_bulk_node_number(const int& face_index,const unsigned int& i) const;

    void shape(const Vector<double>& s, Shape& psi) const override
    {
        PyramidElementShapeC2::shape(s, psi);
    }

    void dshape_local(const Vector<double>& s,Shape& psi,DShape& dpsids) const override
    {
        PyramidElementShapeC2::dshape_local(s, psi, dpsids);
    }
    

    // ---------------------------------------------------------------
    // local_coordinate_of_node
    // ---------------------------------------------------------------
    // Return the local coordinate s of node j.
    inline void local_coordinate_of_node(const unsigned& j,
                                          Vector<double>& s) const
    {
        if (j >= 14)
        {
            std::ostringstream error_message;
            error_message << "Element only has 14 nodes; called with node "
                          << j << std::endl;
            throw OomphLibError(error_message.str(),
                                OOMPH_CURRENT_FUNCTION,
                                OOMPH_EXCEPTION_LOCATION);
        }

        switch (j)
        {
            // Vertices (base)
            case 0: s[0] = 0.0; s[1] = 0.0; s[2] = 0.0; break;
            case 1: s[0] = 1.0; s[1] = 0.0; s[2] = 0.0; break;
            case 2: s[0] = 1.0; s[1] = 1.0; s[2] = 0.0; break;
            case 3: s[0] = 0.0; s[1] = 1.0; s[2] = 0.0; break;

            // Vertex (apex)
            case 4: s[0] = 0.0; s[1] = 0.0; s[2] = 1.0; break;

            // Base-edge midpoints
            case 5: s[0] = 0.5; s[1] = 0.0; s[2] = 0.0; break; // edge 0-1
            case 6: s[0] = 1.0; s[1] = 0.5; s[2] = 0.0; break; // edge 1-2
            case 7: s[0] = 0.5; s[1] = 1.0; s[2] = 0.0; break; // edge 2-3
            case 8: s[0] = 0.0; s[1] = 0.5; s[2] = 0.0; break; // edge 0-3

            // Apex-edge midpoints
            case 9:  s[0] = 0.0; s[1] = 0.0; s[2] = 0.5; break; // edge 0-4
            case 10: s[0] = 0.5; s[1] = 0.0; s[2] = 0.5; break; // edge 1-4
            case 11: s[0] = 0.5; s[1] = 0.5; s[2] = 0.5; break; // edge 2-4
            case 12: s[0] = 0.0; s[1] = 0.5; s[2] = 0.5; break; // edge 3-4

            // Base-face centre
            case 13: s[0] = 0.5; s[1] = 0.5; s[2] = 0.0; break;

            default: break; // unreachable: j is checked against 14 above
        }
    }
};

}